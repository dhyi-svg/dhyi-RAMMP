#!/usr/bin/env python3
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
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped, Twist
from std_srvs.srv import Trigger

from arm_interfaces.srv import CheckReachability

# Direct Kortex imports for partial gripper control
try:
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.RouterClient import RouterClient
    from kortex_api.SessionManager import SessionManager
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.messages import Session_pb2, Base_pb2
except ModuleNotFoundError:
    pass
from arm_interfaces.action import ReachPreset

# ── Pick geometry ──────────────────────────────────────────────────────────────
APPROACH_HEIGHT = 0.12   # m above bottle centroid for pre-grasp approach
GRASP_Z_OFFSET  = 0.02   # m above bottle centroid for grasp

# Natural EE orientation when arm reaches toward the bottle
EE_QUAT = [0.505, 0.628, 0.389, 0.447]  # [x, y, z, w]

# ── Safety limits ──────────────────────────────────────────────────────────────
MAX_X =  1.0
MAX_Y =  0.5
MAX_Z =  0.7
MIN_Z = -0.5

# ── Motion parameters ─────────────────────────────────────────────────────────
PHASE_TIMEOUT_S  =  4.0   # max seconds per move
MIN_MOVE_TIME_S  =  3.0   # wait this long before checking if arm settled
SPEED_THRESHOLD  =  0.12  # m/s — EE considered stopped below this
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
        self._open_gripper_client = self.create_client(
            Trigger, "/arm/open_gripper", callback_group=self._cb_group
        )
        self._close_gripper_client = self.create_client(
            Trigger, "/arm/close_gripper", callback_group=self._cb_group
        )
        self._check_reachability_client = self.create_client(
            CheckReachability, "/arm/check_reachability", callback_group=self._cb_group
        )

        # ── Action client for homing ───────────────────────────────────────────
        self._reach_preset_client = ActionClient(
            self, ReachPreset, "/arm/reach_preset", callback_group=self._cb_group
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

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

    # ── Home ───────────────────────────────────────────────────────────────────

    def _go_home(self):
        """Send arm to RAMMP home preset."""
        self.get_logger().info("Returning to home...")

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

    # ── Pick sequence ──────────────────────────────────────────────────────────

    def pick(self) -> bool:
        self.get_logger().info("=== PICK SEQUENCE START ===")

        if not self._wait_for_services():
            return False

        # Validate bottle pose
        pose = self.latest_bottle_pose
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

        # Step 1: open gripper
        self.get_logger().info("[STEP 1] Opening gripper")
        if not self._open_gripper():
            self.get_logger().error("ABORT: Could not open gripper")
            return False

        # Step 2: approach above bottle
        self.get_logger().info("[STEP 2] Approaching above bottle")
        if not self._move_to_xyz(x, y, z + APPROACH_HEIGHT):
            self.get_logger().error("ABORT: Approach failed")
            return False

        # Step 3: descend to grasp with contact detection
        self.get_logger().info("[STEP 3] Descending to grasp")
        if not self._move_to_xyz(x, y, z + GRASP_Z_OFFSET, check_force=True):
            self.get_logger().error("ABORT: Grasp descent failed")
            return False

        # Step 4: close gripper (partial to avoid crushing bottle)
        self.get_logger().info("[STEP 4] Closing gripper")
        if not self._partial_close_gripper(0.6):
            self.get_logger().error("ABORT: Could not close gripper")
            return False

        # Step 5: return to home
        self.get_logger().info("[STEP 5] Returning to home")
        self._go_home()

        self.get_logger().info("=== PICK COMPLETE ===")
        return True




def main():
    rclpy.init()
    node = BottlePickController()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("Waiting 5s for bottle pose...")
    time.sleep(5.0)

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
