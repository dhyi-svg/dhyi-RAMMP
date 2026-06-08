#!/usr/bin/env python3
"""
Bottle detector node for RAMMP demo.

Uses YOLO segmentation + depth image for 3D localization.
Mirrors the approach used in last year's button_detector.py for efficiency
on the Jetson Orin Nano.

Partial detection handling:
  - If the bottle mask touches the bottom of the frame, the bottle extends
    further down than detected. The centroid is shifted down proportionally
    based on how much of the bottle is estimated to be out of frame.
  - If the mask touches any other border, the detection is rejected.

Lifecycle:
  - On launch: initialize subscribers, publishers, TF. YOLO not loaded yet.
  - On /arm/bottle/detection/enable = True: load YOLO, start detection loop.
  - On /arm/bottle/detection/enable = False: stop detection, unload YOLO, free GPU.

Publishes:
  - /arm/bottle/pose (geometry_msgs/PoseStamped): filtered bottle position in base_link.
    Publishes -1/-1/-1 when no valid detection.
"""

import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
import tf2_ros
import tf2_geometry_msgs
import torch
from ultralytics import YOLO
from realsense2_camera_msgs.msg import Extrinsics
from reachability_checker import ReachabilityChecker

MODEL_PATH = '/home/cailyns/yolo11n-seg.pt'
CONF_THRESHOLD = 0.25

# How many pixels from the border counts as "touching"
BORDER_MARGIN_PX = 5

# Must match bottle_pick_controller.py
EE_QUAT = [0.505, 0.628, 0.389, 0.447]  # [x, y, z, w]


def depth_to_meters(depth_cv: np.ndarray) -> np.ndarray:
    if depth_cv.dtype == np.uint16:
        return depth_cv.astype(np.float32) * 0.001
    return depth_cv.astype(np.float32)


class PoseFilter:
    def __init__(self, alpha=0.3, min_samples=3):
        self.alpha = alpha
        self.min_samples = min_samples
        self._count = 0
        self._xyz = np.zeros(3, dtype=np.float64)

    def update(self, xyz):
        if self._count == 0:
            self._xyz = xyz.astype(np.float64)
        else:
            self._xyz = self.alpha * xyz + (1.0 - self.alpha) * self._xyz
        self._count += 1

    @property
    def is_stable(self):
        return self._count >= self.min_samples

    @property
    def xyz(self):
        return self._xyz.copy()

    def reset(self):
        self._count = 0
        self._xyz = np.zeros(3, dtype=np.float64)


class BottleDetector(Node):
    def __init__(self):
        super().__init__('bottle_detector')

        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('color_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('depth_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('extrinsics_topic', '/camera/camera/extrinsics/depth_to_color')
        self.declare_parameter('color_optical_frame', 'camera_color_optical_frame')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('detection_confidence', CONF_THRESHOLD)
        self.declare_parameter('filter_alpha', 0.3)
        self.declare_parameter('filter_min_samples', 3)
        self.declare_parameter('process_rate_hz', 5.0)
        self.declare_parameter('tf_timeout_s', 0.5)
        self.declare_parameter('min_depth_m', 0.10)
        self.declare_parameter('max_depth_m', 3.00)
        self.declare_parameter('min_mask_points', 10)
        self.declare_parameter('depth_stride', 2)

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth_m = None
        self.color_info = None
        self.depth_info = None
        self.depth_to_color_extr = None
        self.color_frame_id = None
        self.last_rgb_t = None
        self._detection_enabled = False
        self.yolo = None
        self.yolo_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self._pose_filter = PoseFilter(
            alpha=float(self.get_parameter('filter_alpha').value),
            min_samples=int(self.get_parameter('filter_min_samples').value),
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._reachability_checker = ReachabilityChecker(self, EE_QUAT)

        self.bottle_pose_pub = self.create_publisher(PoseStamped, '/arm/bottle/pose', 10)
        self.debug_pt_pub = self.create_publisher(PointStamped, '/bottle/debug_point_base', 10)

        self.create_service(SetBool, '/arm/bottle/detection/enable', self._srv_detection_enable)

        self.create_subscription(
            Image, self.get_parameter('rgb_topic').value, self._cb_rgb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self._cb_depth, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, self.get_parameter('color_info_topic').value, self._cb_color_info, 10
        )
        self.create_subscription(
            CameraInfo, self.get_parameter('depth_info_topic').value, self._cb_depth_info, 10
        )

        qos_extr = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Extrinsics, self.get_parameter('extrinsics_topic').value, self._cb_extrinsics, qos_extr
        )

        rate = float(self.get_parameter('process_rate_hz').value)
        self.create_timer(1.0 / max(rate, 0.1), self._process_once)

        self.get_logger().info('BottleDetector started — YOLO not loaded yet')
        self.get_logger().info('Call /arm/bottle/detection/enable (True) to start')

    # ── YOLO load/unload ───────────────────────────────────────────────────────

    def _load_yolo(self):
        if self.yolo is not None:
            return
        self.get_logger().info(f'Loading YOLO: {MODEL_PATH}')
        try:
            self.yolo = YOLO(MODEL_PATH)
            self.yolo.to(self.yolo_device)
            self.get_logger().info(f'YOLO loaded on {self.yolo_device}')
        except Exception as e:
            if self.yolo_device.startswith('cuda'):
                self.get_logger().warn(f'GPU load failed: {e} — trying CPU')
                self.yolo_device = 'cpu'
                self.yolo = YOLO(MODEL_PATH)
                self.yolo.to(self.yolo_device)
            else:
                self.yolo = None
                raise

    def _unload_yolo(self):
        if self.yolo is None:
            return
        self.get_logger().info('Unloading YOLO — freeing GPU memory')
        del self.yolo
        self.yolo = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _srv_detection_enable(self, request, response):
        if request.data:
            try:
                self._load_yolo()
            except Exception as e:
                response.success = False
                response.message = f'Failed to load YOLO: {e}'
                return response
            self._pose_filter.reset()
            self._detection_enabled = True
            self.get_logger().info('Detection ENABLED')
            response.message = 'Detection started'
        else:
            self._detection_enabled = False
            self._pose_filter.reset()
            self._unload_yolo()
            self.get_logger().info('Detection DISABLED')
            response.message = 'Detection stopped'
        response.success = True
        return response

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _cb_rgb(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.last_rgb_t = time.time()
            self.color_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().debug(f'RGB convert failed: {e}')

    def _cb_depth(self, msg):
        try:
            depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_m = depth_to_meters(depth_cv)
        except Exception as e:
            self.get_logger().debug(f'Depth convert failed: {e}')

    def _cb_color_info(self, msg):
        self.color_info = msg
        if msg.header.frame_id:
            self.color_frame_id = msg.header.frame_id

    def _cb_depth_info(self, msg):
        self.depth_info = msg

    def _cb_extrinsics(self, msg):
        self.depth_to_color_extr = msg

    # ── YOLO ──────────────────────────────────────────────────────────────────

    def _run_yolo(self, img):
        """Run YOLO and return (mask, bbox, conf, bottom_cut_fraction).

        bottom_cut_fraction: fraction of bottle estimated to be below frame (0.0 = fully visible).
        Returns (None, None, None, 0.0) if no valid detection.
        """
        if self.yolo is None:
            return None, None, None, 0.0
        try:
            results = self.yolo(
                img,
                conf=float(self.get_parameter('detection_confidence').value),
                verbose=False,
                device=self.yolo_device,
            )
        except Exception as e:
            self.get_logger().debug(f'YOLO failed: {e}')
            return None, None, None, 0.0

        n = len(results[0].boxes) if results[0].boxes is not None else 0
        self.get_logger().info(
            f'YOLO: {n} detections, classes: '
            f'{results[0].boxes.cls.tolist() if results[0].boxes is not None else []}'
        )

        if not results or results[0].masks is None:
            return None, None, None, 0.0
        r = results[0]
        if len(r.boxes) == 0:
            return None, None, None, 0.0

        # Only consider bottle detections (COCO class 39)
        bottle_indices = [
            i for i in range(len(r.boxes))
            if int(r.boxes.cls[i].item()) == 39
        ]
        if len(bottle_indices) == 0:
            self.get_logger().debug('No bottle class (39) detected')
            return None, None, None, 0.0
        best = max(bottle_indices, key=lambda i: float(r.boxes.conf[i].item()))

        mask_poly = r.masks.xy[best]
        conf_val = float(r.boxes.conf[best].item())
        bbox = r.boxes.xyxy[best].cpu().numpy()

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [mask_poly.astype(np.int32)], 255)

        # Check which borders the mask touches
        touches_top    = mask[:BORDER_MARGIN_PX, :].any()
        touches_left   = mask[:, :BORDER_MARGIN_PX].any()
        touches_right  = mask[:, -BORDER_MARGIN_PX:].any()
        touches_bottom = mask[-BORDER_MARGIN_PX:, :].any()

        # Reject if touches top, left, or right — can't estimate correction
        if touches_top or touches_left or touches_right:
            self.get_logger().debug('Rejecting partial detection (touches top/left/right border)')
            return None, None, None, 0.0

        # If touches bottom, estimate how much of the bottle is below frame
        bottom_cut_fraction = 0.0
        if touches_bottom:
            mask_rows = np.where(mask.any(axis=1))[0]
            if len(mask_rows) > 0:
                mask_top_px    = float(mask_rows[0])
                mask_bottom_px = float(mask_rows[-1])
                visible_height = mask_bottom_px - mask_top_px

                if visible_height > 0:
                    # Assume bottle is symmetric — estimate true bottom by mirroring
                    # the visible portion below the detected bottom
                    estimated_true_bottom = 2.0 * mask_bottom_px - mask_top_px
                    pixels_below = max(0.0, estimated_true_bottom - (h - 1))
                    estimated_true_height = visible_height + pixels_below
                    bottom_cut_fraction = pixels_below / estimated_true_height
                    self.get_logger().info(
                        f'Partial bottom detection: {bottom_cut_fraction:.2f} of bottle below frame'
                    )

        return mask, bbox, conf_val, bottom_cut_fraction

    # ── 3D from depth image ────────────────────────────────────────────────────

    def _get_centroid_from_depth(self, mask, bbox):
        """Get 3D centroid using depth image + camera intrinsics."""
        if self.color_info is None or self.depth_info is None or self.depth_to_color_extr is None:
            self.get_logger().debug('Missing camera info or extrinsics')
            return None

        depth_m = self.latest_depth_m
        if depth_m is None:
            self.get_logger().debug('No depth image')
            return None

        Hc, Wc = int(self.color_info.height), int(self.color_info.width)
        Hd, Wd = depth_m.shape[:2]

        Kc = self.color_info.k
        fx_c, fy_c, cx_c, cy_c = float(Kc[0]), float(Kc[4]), float(Kc[2]), float(Kc[5])
        Kd = self.depth_info.k
        fx_d, fy_d, cx_d, cy_d = float(Kd[0]), float(Kd[4]), float(Kd[2]), float(Kd[5])

        R = np.array(self.depth_to_color_extr.rotation, dtype=np.float32).reshape(3, 3)
        t = np.array(self.depth_to_color_extr.translation, dtype=np.float32).reshape(3)

        min_z = float(self.get_parameter('min_depth_m').value)
        max_z = float(self.get_parameter('max_depth_m').value)
        stride = max(1, int(self.get_parameter('depth_stride').value))
        min_pts = int(self.get_parameter('min_mask_points').value)

        x1, y1, x2, y2 = bbox.astype(np.float32)
        margin = 20
        fx_ratio = fx_d / fx_c if fx_c > 0 else 1.0
        fy_ratio = fy_d / fy_c if fy_c > 0 else 1.0
        d_x1 = max(0, int((x1 - cx_c) * fx_ratio + cx_d) - margin)
        d_x2 = min(Wd, int((x2 - cx_c) * fx_ratio + cx_d) + margin)
        d_y1 = max(0, int((y1 - cy_c) * fy_ratio + cy_d) - margin)
        d_y2 = min(Hd, int((y2 - cy_c) * fy_ratio + cy_d) + margin)

        points_color = []
        for v in range(d_y1, d_y2, stride):
            for u in range(d_x1, d_x2, stride):
                z = float(depth_m[v, u])
                if not (min_z < z < max_z):
                    continue
                Xd = (u - cx_d) * z / fx_d
                Yd = (v - cy_d) * z / fy_d
                Pc = R @ np.array([Xd, Yd, z], dtype=np.float32) + t
                Zc = float(Pc[2])
                if Zc <= 0.0:
                    continue
                up = int(round(fx_c * (float(Pc[0]) / Zc) + cx_c))
                vp = int(round(fy_c * (float(Pc[1]) / Zc) + cy_c))
                if up < 0 or up >= Wc or vp < 0 or vp >= Hc:
                    continue
                if mask[vp, up] == 0:
                    continue
                points_color.append(Pc)

        if len(points_color) < min_pts:
            self.get_logger().debug(f'Not enough masked points: {len(points_color)}')
            return None

        return np.median(np.stack(points_color, axis=0), axis=0)

    # ── TF transform ───────────────────────────────────────────────────────────

    def _transform_to_base(self, point_cam):
        cam_frame = self.color_frame_id or self.get_parameter('color_optical_frame').value
        base_frame = self.get_parameter('base_frame').value
        tf_timeout = float(self.get_parameter('tf_timeout_s').value)

        ps = PointStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = cam_frame
        ps.point.x = float(point_cam[0])
        ps.point.y = float(point_cam[1])
        ps.point.z = float(point_cam[2])

        try:
            tf = self.tf_buffer.lookup_transform(
                base_frame, cam_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=tf_timeout),
            )
            pb = tf2_geometry_msgs.do_transform_point(ps, tf)
            return np.array([pb.point.x, pb.point.y, pb.point.z])
        except Exception as e:
            self.get_logger().debug(f'TF failed: {e}')
            return None

    # ── Publishers ─────────────────────────────────────────────────────────────

    def _publish_invalid(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = msg.pose.position.y = msg.pose.position.z = -1.0
        self.bottle_pose_pub.publish(msg)

    def _publish_bottle_pose(self, xyz):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.w = 1.0
        self.bottle_pose_pub.publish(msg)

        pt = PointStamped()
        pt.header = msg.header
        pt.point = msg.pose.position
        self.debug_pt_pub.publish(pt)
        self.get_logger().info(f'Bottle: x={xyz[0]:.3f} y={xyz[1]:.3f} z={xyz[2]:.3f}')

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _process_once(self):
        if not self._detection_enabled:
            return

        if self.latest_rgb is None:
            self.get_logger().debug('Waiting for RGB...')
            self._publish_invalid()
            return
        if self.latest_depth_m is None:
            self.get_logger().debug('Waiting for depth...')
            self._publish_invalid()
            return

        mask, bbox, conf, bottom_cut_fraction = self._run_yolo(self.latest_rgb)
        if mask is None:
            self.get_logger().debug('No bottle detected')
            self._publish_invalid()
            return

        self.get_logger().debug(f'Detected conf={conf:.2f}')

        centroid_cam = self._get_centroid_from_depth(mask, bbox)
        if centroid_cam is None:
            self._publish_invalid()
            return

        centroid_base = self._transform_to_base(centroid_cam)
        if centroid_base is None:
            self._publish_invalid()
            return

        # If bottle is partially below frame, shift the centroid down
        BOTTLE_HEIGHT_M = 0.25
        if bottom_cut_fraction > 0.0:
            z_shift = -bottom_cut_fraction * BOTTLE_HEIGHT_M * 0.5
            centroid_base[2] += z_shift
            self.get_logger().info(f'Applied z correction: {z_shift:.3f}m')

        self._pose_filter.update(centroid_base)
        if not self._pose_filter.is_stable:
            self.get_logger().debug('Filter warming up...')
            self._publish_invalid()
            return

        # Fire async reachability check — result available next cycle
        filtered_xyz = self._pose_filter.xyz
        self._reachability_checker.check_async(filtered_xyz)

        if not self._reachability_checker.is_reachable:
            self.get_logger().debug('Bottle not reachable — publishing invalid')
            self._publish_invalid()
            return

        self._publish_bottle_pose(filtered_xyz)


def main():
    rclpy.init()
    node = BottleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
