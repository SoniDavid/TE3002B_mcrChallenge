#!/usr/bin/env python3
"""
Visual detector node for IBVS.

Detects a target (ArUco marker or colored ball) in the camera image and
publishes a 4-element feature vector [eu, ev, ea, confidence] where:
  eu = u_centroid - cx       (horizontal pixel error, + = target is right of center)
  ev = v_centroid - cy       (vertical pixel error,   + = target is below center)
  ea = sqrt(area) - sqrt(desired_area)  (scale error, + = target is closer than goal)
  confidence = 1.0 if detection valid, 0.0 otherwise

Subscribes:
  /camera/image_compressed  (sensor_msgs/CompressedImage)

Publishes:
  /visual_features           (std_msgs/Float64MultiArray)  [eu, ev, ea, confidence]
  /visual_detector/debug_image (sensor_msgs/Image)          annotated frame
"""

import signal
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64MultiArray

_BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class VisualDetectorNode(Node):

    def __init__(self):
        super().__init__('visual_detector_node')

        self.declare_parameter('detector_type', 'color_blob')  # 'aruco' or 'color_blob'
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('cx', -1.0)   # <0 → use image_width/2
        self.declare_parameter('cy', -1.0)   # <0 → use image_height/2
        self.declare_parameter('desired_area', 8000.0)  # pixels² at goal distance

        # ArUco params
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size_m', 0.10)
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')

        # Color blob params
        self.declare_parameter('hsv_lower', [0, 120, 70])
        self.declare_parameter('hsv_upper', [10, 255, 255])
        self.declare_parameter('ball_diameter_m', 0.065)
        self.declare_parameter('min_contour_area', 500.0)

        self.declare_parameter('confidence_threshold', 0.5)

        # Read params
        self._det_type = self.get_parameter('detector_type').value
        w = self.get_parameter('image_width').value
        h = self.get_parameter('image_height').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        self._cx = cx if cx >= 0 else w / 2.0
        self._cy = cy if cy >= 0 else h / 2.0
        self._desired_area = float(self.get_parameter('desired_area').value)
        self._sqrt_desired_area = np.sqrt(self._desired_area)
        self._min_contour_area = self.get_parameter('min_contour_area').value

        # ArUco detector (lazy-init so it doesn't fail when using color_blob)
        self._aruco_detector = None
        self._target_marker_id = self.get_parameter('marker_id').value
        if self._det_type == 'aruco':
            self._init_aruco_detector()

        # HSV bounds (list params)
        lo = self.get_parameter('hsv_lower').value
        hi = self.get_parameter('hsv_upper').value
        self._hsv_lower = np.array(lo, dtype=np.uint8)
        self._hsv_upper = np.array(hi, dtype=np.uint8)

        # Morphological kernel for noise removal
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Publishers
        self._feat_pub = self.create_publisher(Float64MultiArray, '/visual_features', 10)
        self._dbg_pub = self.create_publisher(Image, '/visual_detector/debug_image', 1)

        # Subscriber
        self.create_subscription(
            CompressedImage,
            '/camera/image_compressed',
            self._image_callback,
            _BEST_EFFORT_QOS,
        )

        self.get_logger().info(
            f'VisualDetectorNode ready — detector={self._det_type}, '
            f'cx={self._cx:.1f}, cy={self._cy:.1f}, '
            f'desired_area={self._desired_area:.1f}'
        )

    # ── ArUco setup ──────────────────────────────────────────────────────────

    def _init_aruco_detector(self):
        dict_name = self.get_parameter('aruco_dict').value
        aruco_dict_id = getattr(cv2.aruco, dict_name, None)
        if aruco_dict_id is None:
            self.get_logger().error(
                f'Unknown aruco_dict "{dict_name}", falling back to DICT_4X4_50'
            )
            aruco_dict_id = cv2.aruco.DICT_4X4_50
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.get_logger().info(
            f'ArUco detector initialised — dict={dict_name}, '
            f'target_id={self._target_marker_id}'
        )

    # ── Image callback ───────────────────────────────────────────────────────

    def _image_callback(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return

        if self._det_type == 'aruco':
            eu, ev, ea, conf, frame = self._detect_aruco(frame)
        else:
            eu, ev, ea, conf, frame = self._detect_color_blob(frame)

        feat = Float64MultiArray()
        feat.data = [eu, ev, ea, conf]
        self._feat_pub.publish(feat)

        # Only publish debug image when someone is watching
        if self._dbg_pub.get_subscription_count() > 0:
            self._publish_debug(frame, msg.header.stamp)

    # ── ArUco detection ──────────────────────────────────────────────────────

    def _detect_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)

        eu, ev, ea, conf = 0.0, 0.0, 0.0, 0.0

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id != self._target_marker_id:
                    continue
                pts = corners[i][0]  # shape (4, 2)
                cx_det = float(np.mean(pts[:, 0]))
                cy_det = float(np.mean(pts[:, 1]))
                # Area from convex hull of corners
                hull = cv2.convexHull(pts.astype(np.float32))
                area = float(cv2.contourArea(hull))

                eu = cx_det - self._cx
                ev = cy_det - self._cy
                ea = np.sqrt(max(area, 0.0)) - self._sqrt_desired_area
                conf = 1.0

                # Draw on frame
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.circle(frame, (int(cx_det), int(cy_det)), 6, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f'eu={eu:.0f} ev={ev:.0f} ea={ea:.1f}',
                    (int(cx_det) + 10, int(cy_det) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
                break  # Use first matching marker only

        self._draw_crosshair(frame)
        return eu, ev, ea, conf, frame

    # ── Color blob detection ─────────────────────────────────────────────────

    def _detect_color_blob(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Handle hue wrap-around for reds (hue near 0/180)
        if self._hsv_lower[0] <= self._hsv_upper[0]:
            mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        else:
            # Hue wraps: e.g. lower=[170,..] upper=[10,..]
            lo2 = self._hsv_lower.copy()
            lo2[0] = 0
            hi2 = self._hsv_upper.copy()
            hi2[0] = 180
            mask1 = cv2.inRange(hsv, self._hsv_lower, hi2)
            mask2 = cv2.inRange(hsv, lo2, self._hsv_upper)
            mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        eu, ev, ea, conf = 0.0, 0.0, 0.0, 0.0

        if contours:
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)

            if area >= self._min_contour_area:
                M = cv2.moments(best)
                if M['m00'] > 0:
                    cx_det = M['m10'] / M['m00']
                    cy_det = M['m01'] / M['m00']
                    eu = cx_det - self._cx
                    ev = cy_det - self._cy
                    ea = np.sqrt(area) - self._sqrt_desired_area
                    conf = 1.0

                    cv2.drawContours(frame, [best], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx_det), int(cy_det)), 6, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f'eu={eu:.0f} ev={ev:.0f} ea={ea:.1f}',
                        (int(cx_det) + 10, int(cy_det) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )

        self._draw_crosshair(frame)
        return eu, ev, ea, conf, frame

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _draw_crosshair(self, frame):
        h, w = frame.shape[:2]
        cx, cy = int(self._cx), int(self._cy)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 0), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 0), 1)

    def _publish_debug(self, frame, stamp):
        h, w, c = frame.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera_optical_frame'
        msg.height = h
        msg.width = w
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = w * c
        msg.data = frame.tobytes()
        self._dbg_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisualDetectorNode()

    def _shutdown(signum, frame):
        rclpy.try_shutdown()

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
