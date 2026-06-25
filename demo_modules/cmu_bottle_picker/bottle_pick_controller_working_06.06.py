"""
Bottle pick controller for RAMMP demo.

Subscribes to /arm/bottle/pose from the bottle detector and executes
a full pick sequence using the arm_driver ROS interface.

Movement mirrors ButtonPushController from last year:
  - Check reachability via /arm/check_reachability before each move
  - Publish PoseStamped to /arm/cmu/cartesian_pose
  - Poll /arm/ee/velocity to wait for arm to fully settle
  - Monitor /arm/ee/force for contact during descent
"""

import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from std_srvs.srv import Trigger, SetBool

from arm_interfaces.srv import CheckReachability
from arm_interfaces.action import ReachPreset
from rclpy.action import ActionClient

# ── Pick geometry ──────────────────────────────────────────────────────────────
APPROACH_HEIGHT = 0.12   # m above bottle centroid for pre-grasp approach
GRASP_Z_OFFSET  = 0.02

# Handoff position — tune for demo
HANDOFF_X = 0.7
HANDOFF_Y = 0.0
HANDOFF_Z = 0.55   # m above bottle centroid for grasp

# Natural EE orientation when arm reaches toward the bottle
EE_QUAT = [0.505, 0.628, 0.389, 0.447]  # [x, y, z, w]

# ── Safety limits ──────────────────────────────────────────────────────────────
MAX_X =  1.0
MAX_Y =  0.5
MAX_Z =  1.2
MIN_Z = -0.5

# ── Motion parameters ─────────────────────────────────────────────────────────
PHASE_TIMEOUT_S  = 4  # max seconds per move
MIN_MOVE_TIME_S  =  3  # wait this long before checking if arm settled
SPEED_THRESHOLD  =  0.12 # m/s — EE considered stopped below this
FORCE_THRESHOLD  = 15.0   # N — contact detection during descent
POLL_RATE_S      =  0.05  # seconds between polls

# ── Stale pose rejection ───────────────────────────────────────────────────────
POSE_MAX_AGE_S = 10.0

# ── Service timeouts ──────────────────────────────────────────────────────────
SERVICE_TIMEOUT_S = 10.0


class BottlePickController(Node):
    def __init__(self):
        super().__init__("bottle_pick_controller")

        self._cb_group = ReentrantCallbackGroup()

        # ── Service clients ────────────────────────────────────────────────────
        self._detection_enable_client = self.create_client(
            SetBool, "/arm/bottle/detection/enable", callback_group=self._cb_group
        )

        self._open_gripper_client = self.create_client(
            Trigger, "/arm/open_gripper", callback_group=self._cb_group
        )
        self._close_gripper_client = self.create_client(
            Trigger, "/arm/close_gripper", callback_group=self._cb_group
        )
        self._check_reachability_client = self.create_client(
            CheckReachability, "/arm/check_reachability", callback_group=self._cb_group
        )

        # ── Publishers ─────────────────────────────────────────────────────────
        self._cartesian_pose_pub = self.create_publisher(
            PoseStamped, "/arm/cmu/cartesian_pose", 10
        )

        # ── Subscribers ────────────────────────────────────────────────────────
        self._latest_ee_velocity = None
        self.create_subscription(
            TwistStamped, "/arm/ee/velocity", self._cb_ee_velocity, 10
        )

        self._latest_ee_force = None
        self.create_subscription(
            Vector3Stamped, "/arm/ee/force", self._cb_ee_force, 10
        )

        self.latest_bottle_pose = None
        self._latest_pose_time = None
        self.create_subscription(
            PoseStamped, "/arm/bottle/pose", self._cb_bottle_pose, 10
        )



        self.get_logger().info("BottlePickController ready")

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _cb_ee_velocity(self, msg: TwistStamped):
        v = msg.twist.linear
        self._latest_ee_velocity = np.array([v.x, v.y, v.z])

    def _cb_ee_force(self, msg: Vector3Stamped):
        self._latest_ee_force = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def _cb_bottle_pose(self, msg: PoseStamped):
        self.latest_bottle_pose = msg
        self._latest_pose_time = self.get_clock().now()

    # ── Sensor helpers ─────────────────────────────────────────────────────────

    def _get_ee_speed(self):
        if self._latest_ee_velocity is None:
            return None
        return float(np.linalg.norm(self._latest_ee_velocity))

    def _get_ee_force(self):
        if self._latest_ee_force is None:
            return 0.0
        return float(np.linalg.norm(self._latest_ee_force))

    # ── Service helpers ────────────────────────────────────────────────────────

    def _wait_for_services(self) -> bool:
        for client in [
            self._open_gripper_client,
            self._close_gripper_client,
            self._check_reachability_client,
        ]:
            if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
                self.get_logger().error(
                    f"Service {client.srv_name} not available after {SERVICE_TIMEOUT_S}s"
                )
                return False
        return True

    def _call_trigger(self, client, label: str) -> bool:
        future = client.call_async(Trigger.Request())
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        if future.result() is None:
            self.get_logger().error(f"{label}: no response")
            return False
        if not future.result().success:
            self.get_logger().error(f"{label} failed: {future.result().message}")
            return False
        self.get_logger().info(f"{label} OK")
        return True

    def _open_gripper(self) -> bool:
        ok = self._call_trigger(self._open_gripper_client, "open_gripper")
        if ok:
            time.sleep(1.5)
        return ok

    def _close_gripper(self) -> bool:
        ok = self._call_trigger(self._close_gripper_client, "close_gripper")
        if ok:
            time.sleep(1.5)
        return ok


    def _partial_close_gripper(self, value=0.6) -> bool:
        """Close gripper to a partial position via direct Kortex command.

        Args:
            value: Gripper position 0.0 (open) to 1.0 (fully closed). Default 0.6.
        """
        try:
            from kortex_api.TCPTransport import TCPTransport
            from kortex_api.RouterClient import RouterClient
            from kortex_api.SessionManager import SessionManager
            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            from kortex_api.autogen.messages import Session_pb2, Base_pb2
            transport = TCPTransport()
            transport.connect('192.168.1.10', 10000)
            router = RouterClient(transport, lambda kx: None)
            session_manager = SessionManager(router)
            session_manager.CreateSession(Session_pb2.CreateSessionInfo(
                username='admin',
                password='admin',
            ))
            base = BaseClient(router)
            cmd = Base_pb2.GripperCommand()
            cmd.mode = Base_pb2.GRIPPER_POSITION
            finger = cmd.gripper.finger.add()
            finger.value = float(value)
            base.SendGripperCommand(cmd)
            time.sleep(1.5)
            session_manager.CloseSession()
            transport.disconnect()
            self.get_logger().info(f"Partial gripper close OK (value={value:.2f})")
            return True
        except Exception as e:
            self.get_logger().error(f"partial_close_gripper failed: {e}")
            return False

    def _check_position_safe(self, x, y, z) -> bool:
        if x > MAX_X or abs(y) > MAX_Y:
            self.get_logger().error(
                f"Position out of safe XY range: x={x:.3f} y={y:.3f}"
            )
            return False
        if z < MIN_Z or z > MAX_Z:
            self.get_logger().error(f"Z out of safe range: z={z:.3f}")
            return False
        return True

    def _check_reachable(self, x, y, z) -> bool:
        req = CheckReachability.Request()
        req.target_pose.position.x = float(x)
        req.target_pose.position.y = float(y)
        req.target_pose.position.z = float(z)
        req.target_pose.orientation.x = EE_QUAT[0]
        req.target_pose.orientation.y = EE_QUAT[1]
        req.target_pose.orientation.z = EE_QUAT[2]
        req.target_pose.orientation.w = EE_QUAT[3]

        future = self._check_reachability_client.call_async(req)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if future.result() is None:
            self.get_logger().error("check_reachability: no response")
            return False
        res = future.result()
        if not res.reachable:
            self.get_logger().error(f"Target not reachable: {res.message}")
            return False

        self.get_logger().info(f"Reachability confirmed for x={x:.3f} y={y:.3f} z={z:.3f}")
        return True

    def _make_pose(self, x, y, z) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.x = EE_QUAT[0]
        msg.pose.orientation.y = EE_QUAT[1]
        msg.pose.orientation.z = EE_QUAT[2]
        msg.pose.orientation.w = EE_QUAT[3]
        return msg

    # ── Movement ───────────────────────────────────────────────────────────────

    def _move_to_xyz(self, x, y, z, check_force=False) -> bool:
        if not self._check_position_safe(x, y, z):
            return False
        if not self._check_reachable(x, y, z):
            return False

        self.get_logger().info(f"Moving to x={x:.3f} y={y:.3f} z={z:.3f}")
        
        pose = self._make_pose(x, y, z)
        for _ in range(5):
            self._cartesian_pose_pub.publish(pose)
            time.sleep(0.05)

        time.sleep(MIN_MOVE_TIME_S)

        deadline = time.monotonic() + PHASE_TIMEOUT_S
        deadline = time.monotonic() + PHASE_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(POLL_RATE_S)

            speed = self._get_ee_speed()
            force = self._get_ee_force()


            if check_force and force > FORCE_THRESHOLD:
                self.get_logger().info(f"Contact detected ({force:.1f} N)")
                return True

            if speed is not None and speed < SPEED_THRESHOLD:
                self.get_logger().info(f"Arm settled (speed={speed:.4f} m/s)")
                return True

        self.get_logger().warn(f"Move timed out after {PHASE_TIMEOUT_S}s — continuing")
        return True

    # ── Pick sequence ──────────────────────────────────────────────────────────

    def _go_home(self):
        """Send arm to RAMMP home preset — no mode change needed."""
        self.get_logger().info("Returning to home...")

        if not hasattr(self, "_reach_preset_client") or self._reach_preset_client is None:
            self._reach_preset_client = ActionClient(self, ReachPreset, "/arm/reach_preset", callback_group=self._cb_group)
        if not self._reach_preset_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("reach_preset action server not available")
            return

        goal = ReachPreset.Goal()
        goal.preset = ReachPreset.Goal.PRESET_HOME
        send_future = self._reach_preset_client.send_goal_async(goal)

        deadline = time.time() + 5.0
        while not send_future.done() and time.time() < deadline:
            time.sleep(0.05)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Home goal rejected")
            return

        self.get_logger().info("Homing — waiting for completion...")
        result_future = goal_handle.get_result_async()
        deadline = time.time() + 30.0
        while not result_future.done() and time.time() < deadline:
            time.sleep(0.05)
        self.get_logger().info("Home complete")

    def _reset_detector(self):
        """Reset the bottle detector for a fresh stable reading."""
        self.get_logger().info("Resetting detector for fresh reading...")
        req_off = SetBool.Request()
        req_off.data = False
        future = self._detection_enable_client.call_async(req_off)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(1.0)
        req_on = SetBool.Request()
        req_on.data = True
        future = self._detection_enable_client.call_async(req_on)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        self.get_logger().info("Detector reset — waiting for stable reading...")
        time.sleep(5.0)

    def pick(self) -> bool:
        self.get_logger().info("=== PICK SEQUENCE START ===")

        if not self._wait_for_services():
            return False

        # Reset detector for fresh stable reading
        self._reset_detector()

        # Validate bottle pose — retry up to 10s for YOLO warmup
        pose = None
        for _ in range(20):
            pose = self.latest_bottle_pose
            if pose is not None:
                x = pose.pose.position.x
                y = pose.pose.position.y
                z = pose.pose.position.z
                if not (x == -1.0 and y == -1.0 and z == -1.0):
                    break
            self.get_logger().info("Waiting for valid bottle detection...")
            time.sleep(0.5)
        if pose is None:
            self.get_logger().error("ABORT: No bottle pose received")
            return False

        if self._latest_pose_time is not None:
            age = (self.get_clock().now() - self._latest_pose_time).nanoseconds / 1e9
            if age > POSE_MAX_AGE_S:
                self.get_logger().error(f"ABORT: Pose stale ({age:.1f}s)")
                return False

        x = pose.pose.position.x
        y = pose.pose.position.y
        z = pose.pose.position.z

        if x == -1.0 and y == -1.0 and z == -1.0:
            self.get_logger().error("ABORT: Detector has no valid bottle detection")
            return False

        self.get_logger().info(f"[STEP 0] Bottle at x={x:.3f} y={y:.3f} z={z:.3f}")

        # Pre-check both approach and grasp are reachable before moving
        self.get_logger().info("[STEP 0] Pre-checking reachability...")
        if not self._check_reachable(x, y, z + APPROACH_HEIGHT):
            self.get_logger().error("ABORT: Approach position not reachable")
            return False
        if not self._check_reachable(x, y, z + GRASP_Z_OFFSET):
            self.get_logger().error("ABORT: Grasp position not reachable — bottle too far or bad angle")
            return False

        # Step 1: open gripper
        self.get_logger().info("[STEP 1] Opening gripper")
        if not self._open_gripper():
            self.get_logger().error("ABORT: Could not open gripper")
            return False

        # Step 3: approach above bottle
        self.get_logger().info("[STEP 3] Approaching above bottle")
        if not self._move_to_xyz(x, y, z + APPROACH_HEIGHT):
            self.get_logger().error("ABORT: Approach failed")
            return False

        # Step 4: descend to grasp with contact detection
        self.get_logger().info("[STEP 4] Descending to grasp")
        if not self._move_to_xyz(x, y, z + GRASP_Z_OFFSET, check_force=True):
            self.get_logger().error("ABORT: Grasp descent failed")
            return False

        # Step 5: close gripper
        # Step 5: close gripper
        self.get_logger().info("[STEP 5] Closing gripper")
        if not self._partial_close_gripper(0.41):
            self.get_logger().error("ABORT: Could not close gripper")
            return False

        # Step 6: lift up, move back, then home
        self.get_logger().info("[STEP 6] Lifting up")
        self._move_to_xyz(x, y, z + APPROACH_HEIGHT)
        self.get_logger().info("[STEP 6] Moving back to safe position")
        self._move_to_xyz(0.4, y, z + APPROACH_HEIGHT)
        self.get_logger().info("[STEP 6] Returning to home")
        self._go_home()

        # Step 7: wait for input then move to handoff and release
        self.get_logger().info("[STEP 7] Press ENTER to release bottle...")
        input()
        self.get_logger().info("[STEP 7] Moving to handoff position")
        self._move_to_xyz(HANDOFF_X, HANDOFF_Y, HANDOFF_Z)
        self.get_logger().info("[STEP 7] Waiting for arm to settle...")
        time.sleep(MIN_MOVE_TIME_S)
        deadline = time.monotonic() + PHASE_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(POLL_RATE_S)
            speed = self._get_ee_speed()
            if speed is not None and speed < SPEED_THRESHOLD:
                break
        self.get_logger().info("[STEP 7] Releasing bottle")
        self._open_gripper()
        self.get_logger().info("[STEP 8] Returning to home...")
        self._go_home()
        self.get_logger().info("=== PICK COMPLETE ===")
        return True


    def _bottle_detected(self) -> bool:
        """Return True if detector is publishing a valid non-sentinel pose."""
        if self.latest_bottle_pose is None:
            return False
        if self._latest_pose_time is None:
            return False
        age = (self.get_clock().now() - self._latest_pose_time).nanoseconds / 1e9
        if age > POSE_MAX_AGE_S:
            return False
        p = self.latest_bottle_pose.pose.position
        if p.x == -1.0 and p.y == -1.0 and p.z == -1.0:
            return False
        return True

    def scan_and_pick(self) -> bool:
        """Slowly sweep camera from top-left to bottom-right looking for bottle.

        Generates a dense grid of waypoints and moves slowly through them,
        checking for bottle detection at each step.
        Stops immediately when bottle is found and triggers pick.
        """
        if not self._wait_for_services():
            return False

        self.get_logger().info("=== SCAN START ===")

        # Reset detector for fresh reading
        self._reset_detector()

        # Check if bottle already visible before sweeping
        if self._bottle_detected():
            self.get_logger().info("Bottle already in frame — skipping sweep")
            return self.pick()

        # Diagonal sweep from top-left to bottom-right
        import numpy as np
        x_fixed = 0.50
        n_steps = 12
        y_vals = np.linspace(-0.35, 0.35, n_steps)   # left to right
        z_vals = np.linspace(0.50, 0.25, n_steps)     # top to bottom

        STEP_WAIT_S = 3.0  # seconds to hold each pose — long enough for YOLO

        scan_poses = [(x_fixed, float(y), float(z)) for y, z in zip(y_vals, z_vals)]

        total = len(scan_poses)
        for i, (sx, sy, sz) in enumerate(scan_poses):
            self.get_logger().info(f"Scan {i+1}/{total}: x={sx:.2f} y={sy:.2f} z={sz:.2f}")

            if not self._check_position_safe(sx, sy, sz):
                continue
            if not self._check_reachable(sx, sy, sz):
                continue

            # Move to pose
            pose = self._make_pose(sx, sy, sz)
            for _ in range(5):
                self._cartesian_pose_pub.publish(pose)
                time.sleep(0.05)

            # Wait for arm to settle
            time.sleep(MIN_MOVE_TIME_S)

            # Hold pose and check for detection every 0.2s
            deadline = time.monotonic() + STEP_WAIT_S
            while time.monotonic() < deadline:
                time.sleep(0.2)
                if self._bottle_detected():
                    self.get_logger().info("Bottle detected! Stopping sweep and picking")
                    return self.pick()

        self.get_logger().error("=== SCAN COMPLETE — No bottle found ===")
        return False


def main():
    rclpy.init()
    node = BottlePickController()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(2.0)
    print(f"Spin thread alive: {spin_thread.is_alive()}")

    try:
        success = node.pick()
        if not success:
            node.get_logger().error("Pick sequence failed")
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
