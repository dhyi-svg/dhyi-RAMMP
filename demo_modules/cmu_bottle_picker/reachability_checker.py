"""Async reachability checker for the RAMMP bottle pick demo.

Fires a non-blocking IK check via /arm/check_reachability each detection
cycle. The result is available one cycle later (~200ms at 5Hz — negligible).

Usage:
    checker = ReachabilityChecker(node, ee_quat)
    # In detection loop:
    checker.check_async(bottle_xyz)
    is_reachable = checker.is_reachable
"""

from arm_interfaces.srv import CheckReachability


class ReachabilityChecker:
    """Non-blocking reachability check via /arm/check_reachability.

    Parameters
    ----------
    node : rclpy.node.Node
        The owning ROS node.
    ee_quat : list[float]
        End-effector orientation quaternion [x, y, z, w] to use for IK.
    """

    def __init__(self, node, ee_quat):
        self._node = node
        self._ee_quat = ee_quat
        self._client = node.create_client(CheckReachability, "/arm/check_reachability")
        self._future = None
        self._reachable = False
        self._service_warned = False

    def check_async(self, xyz):
        """Fire an async reachability check for the given position.

        Parameters
        ----------
        xyz : array-like, shape (3,)
            Target position [x, y, z] in base_link frame (metres).
        """
        self._poll()

        if not self._client.service_is_ready():
            if not self._service_warned:
                self._node.get_logger().warn(
                    "/arm/check_reachability not available — "
                    "is_reachable will be False until arm driver is running"
                )
                self._service_warned = True
            self._reachable = False
            return

        if self._service_warned:
            self._node.get_logger().info("Reachability service now available")
            self._service_warned = False

        # Don't fire a new request while one is in-flight
        if self._future is not None and not self._future.done():
            return

        req = CheckReachability.Request()
        req.target_pose.position.x = float(xyz[0])
        req.target_pose.position.y = float(xyz[1])
        req.target_pose.position.z = float(xyz[2])
        req.target_pose.orientation.x = float(self._ee_quat[0])
        req.target_pose.orientation.y = float(self._ee_quat[1])
        req.target_pose.orientation.z = float(self._ee_quat[2])
        req.target_pose.orientation.w = float(self._ee_quat[3])

        self._future = self._client.call_async(req)

    @property
    def is_reachable(self) -> bool:
        """Return True if the last completed IK check succeeded."""
        self._poll()
        return self._reachable

    def _poll(self):
        """Check if the in-flight future has completed and update reachability."""
        if self._future is None or not self._future.done():
            return

        try:
            result = self._future.result()
            self._reachable = result.reachable if result else False
            if not self._reachable and result and result.message:
                self._node.get_logger().debug(
                    f"Reachability check failed: {result.message}"
                )
        except Exception as e:
            self._node.get_logger().debug(f"Reachability check exception: {e}")
            self._reachable = False

        self._future = None
