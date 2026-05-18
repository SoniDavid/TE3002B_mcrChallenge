#!/usr/bin/env python3
"""
Visual detector node for IBVS.

Detects a target (ArUco marker or colored blob) in the camera image and
publishes a 4-element feature vector [eu, ev, ea, confidence] where:
  eu = u_centroid - cx       (horizontal pixel error, + = target is right of center)
  ev = v_centroid - cy       (vertical pixel error,   + = target is below center)
  ea = sqrt(area) - sqrt(desired_area)  (scale error, + = target is closer than goal)
  confidence = 1.0 if detection valid, 0.0 otherwise

Subscribes:
  /camera/image_compressed  (sensor_msgs/CompressedImage)
    The CompressedImage.data field is a raw JPEG byte buffer. This node decodes
    it directly with cv2.imdecode — no cv_bridge or image_transport decompression
    step is needed. The camera publisher (usb_camera_publisher.py) sends JPEG
    so this is correct and efficient.

Publishes:
  /visual_features              (std_msgs/Float64MultiArray)  [eu, ev, ea, confidence]
  /visual_detector/debug_image  (sensor_msgs/Image)  annotated BGR frame
  /visual_detector/debug_mask   (sensor_msgs/Image)  binary HSV mask (mono8)

Robustness features (color_blob mode):
  - Gaussian blur before HSV conversion (reduces JPEG compression artifacts)
  - CLAHE on the Value channel (normalises for uneven / changing illumination)
  - Hue-wrap support for red blobs (H near 0/180)
  - Morphological open to remove speckle
  - Solidity filter: rejects non-convex false positives (shadows, table edges)
  - Area computed from convex hull for stable ea under partial occlusion
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

        # ── General params ────────────────────────────────────────────────────
        self.declare_parameter('detector_type', 'color_blob')
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('cx', -1.0)
        self.declare_parameter('cy', -1.0)
        self.declare_parameter('desired_area', 8000.0)
        self.declare_parameter('confidence_threshold', 0.5)

        # ── ArUco params ──────────────────────────────────────────────────────
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size_m', 0.10)
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')

        # ── Color blob params ─────────────────────────────────────────────────
        self.declare_parameter('hsv_lower', [20, 160, 100])
        self.declare_parameter('hsv_upper', [38, 255, 255])
        self.declare_parameter('ball_diameter_m', 0.065)
        self.declare_parameter('min_contour_area', 500.0)
        self.declare_parameter('min_solidity', 0.75)   # reject non-convex blobs
        self.declare_parameter('use_clahe', True)
        self.declare_parameter('clahe_clip_limit', 2.0)

        self.declare_parameter('detection_holdoff_frames', 3)

        # ── Read params ───────────────────────────────────────────────────────
        self._det_type = self.get_parameter('detector_type').value
        _cx_param = self.get_parameter('cx').value
        _cy_param = self.get_parameter('cy').value
        # None means "auto-detect from the first received frame".
        # This handles cameras that negotiate a different resolution than the params say
        # (e.g. USB camera giving 640x480 when image_width=1280 is in the YAML).
        self._cx = float(_cx_param) if _cx_param >= 0 else None
        self._cy = float(_cy_param) if _cy_param >= 0 else None
        self._desired_area = float(self.get_parameter('desired_area').value)
        self._sqrt_desired_area = np.sqrt(self._desired_area)
        self._min_contour_area = self.get_parameter('min_contour_area').value
        self._min_solidity = self.get_parameter('min_solidity').value
        self._holdoff_frames = int(self.get_parameter('detection_holdoff_frames').value)
        self._consecutive_failures = 0
        self._last_valid = (0.0, 0.0, 0.0, 0.0)  # eu, ev, ea, conf

        lo = self.get_parameter('hsv_lower').value
        hi = self.get_parameter('hsv_upper').value
        self._hsv_lower = np.array(lo, dtype=np.uint8)
        self._hsv_upper = np.array(hi, dtype=np.uint8)

        # CLAHE on Value channel — normalises brightness variation between lighting
        # conditions without touching hue or saturation, which carry the color signal.
        use_clahe = self.get_parameter('use_clahe').value
        clip = self.get_parameter('clahe_clip_limit').value
        self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)) if use_clahe else None

        # Fixed 5×5 Gaussian blur before HSV — softens JPEG block artifacts
        self._blur_ksize = (5, 5)

        # Elliptical kernel for morphological open (removes speckle noise)
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # ── ArUco setup ───────────────────────────────────────────────────────
        self._aruco_detector = None
        self._target_marker_id = self.get_parameter('marker_id').value
        if self._det_type == 'aruco':
            self._init_aruco_detector()

        # ── Publishers ────────────────────────────────────────────────────────
        self._feat_pub = self.create_publisher(Float64MultiArray, '/visual_features', 10)
        self._dbg_pub  = self.create_publisher(Image, '/visual_detector/debug_image', 1)
        self._mask_pub = self.create_publisher(Image, '/visual_detector/debug_mask', 1)

        # ── Subscriber ────────────────────────────────────────────────────────
        # CompressedImage carries a raw JPEG buffer in .data. We decode it with
        # cv2.imdecode — equivalent to cv_bridge.compressed_imgmsg_to_cv2 but
        # without the ROS image_transport overhead.
        self.create_subscription(
            CompressedImage,
            '/camera/image_compressed',
            self._image_callback,
            _BEST_EFFORT_QOS,
        )

        cx_str = 'auto' if self._cx is None else f'{self._cx:.1f}'
        cy_str = 'auto' if self._cy is None else f'{self._cy:.1f}'
        self.get_logger().info(
            f'VisualDetectorNode ready — detector={self._det_type}, '
            f'cx={cx_str}, cy={cy_str}, '
            f'desired_area={self._desired_area:.1f}, '
            f'clahe={"on" if self._clahe else "off"}'
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
        # Decode JPEG directly from the CompressedImage byte buffer.
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Auto-set cx/cy from the actual frame on the first callback.
        if self._cx is None:
            frame_h, frame_w = frame.shape[:2]
            self._cx = frame_w / 2.0
            self._cy = frame_h / 2.0
            self.get_logger().info(
                f'Auto-detected frame {frame_w}x{frame_h} — '
                f'cx={self._cx:.1f}, cy={self._cy:.1f}'
            )

        mask = None
        if self._det_type == 'aruco':
            eu, ev, ea, conf, frame = self._detect_aruco(frame)
        else:
            eu, ev, ea, conf, frame, mask = self._detect_color_blob(frame)

        feat = Float64MultiArray()
        feat.data = [eu, ev, ea, conf]
        self._feat_pub.publish(feat)

        stamp = msg.header.stamp
        if self._dbg_pub.get_subscription_count() > 0:
            self._publish_debug(frame, stamp)
        if mask is not None and self._mask_pub.get_subscription_count() > 0:
            self._publish_mask(mask, stamp)

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _to_hsv(self, frame):
        """Blur → CLAHE on V → return HSV image."""
        blurred = cv2.GaussianBlur(frame, self._blur_ksize, 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        if self._clahe is not None:
            h, s, v = cv2.split(hsv)
            v = self._clahe.apply(v)
            hsv = cv2.merge([h, s, v])
        return hsv

    # ── ArUco detection ──────────────────────────────────────────────────────

    def _detect_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)

        eu, ev, ea, conf = 0.0, 0.0, 0.0, 0.0

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id != self._target_marker_id:
                    continue
                pts = corners[i][0]
                cx_det = float(np.mean(pts[:, 0]))
                cy_det = float(np.mean(pts[:, 1]))
                hull = cv2.convexHull(pts.astype(np.float32))
                area = float(cv2.contourArea(hull))

                eu = cx_det - self._cx
                ev = cy_det - self._cy
                ea = np.sqrt(max(area, 0.0)) - self._sqrt_desired_area
                conf = 1.0

                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.circle(frame, (int(cx_det), int(cy_det)), 6, (0, 255, 0), -1)
                self._draw_stats(frame, int(cx_det), int(cy_det), eu, ev, ea, conf)
                break

        self._draw_crosshair(frame)
        return eu, ev, ea, conf, frame

    # ── Color blob detection ─────────────────────────────────────────────────

    def _detect_color_blob(self, frame):
        hsv = self._to_hsv(frame)

        # Support hue wrap-around (e.g. red spans 170–10)
        if self._hsv_lower[0] <= self._hsv_upper[0]:
            mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        else:
            lo2 = self._hsv_lower.copy(); lo2[0] = 0
            hi2 = self._hsv_upper.copy(); hi2[0] = 180
            mask = cv2.bitwise_or(
                cv2.inRange(hsv, self._hsv_lower, hi2),
                cv2.inRange(hsv, lo2, self._hsv_upper),
            )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        eu, ev, ea, conf = 0.0, 0.0, 0.0, 0.0
        detected = False

        best = self._best_contour(contours)
        if best is not None:
            hull = cv2.convexHull(best)
            # Use hull area for ea — more stable than raw contour area under
            # partial occlusion or JPEG block artifacts at the blob boundary.
            hull_area = float(cv2.contourArea(hull))

            M = cv2.moments(hull)
            if M['m00'] > 0:
                cx_det = M['m10'] / M['m00']
                cy_det = M['m01'] / M['m00']
                eu = cx_det - self._cx
                ev = cy_det - self._cy
                ea = np.sqrt(hull_area) - self._sqrt_desired_area
                conf = 1.0
                detected = True

                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                cv2.circle(frame, (int(cx_det), int(cy_det)), 6, (0, 255, 0), -1)
                self._draw_stats(frame, int(cx_det), int(cy_det), eu, ev, ea, conf)

        if detected:
            self._consecutive_failures = 0
            self._last_valid = (eu, ev, ea, conf)
        else:
            self._consecutive_failures += 1
            # Hold last valid detection until holdoff window expires
            if self._consecutive_failures < self._holdoff_frames:
                eu, ev, ea, conf = self._last_valid
                cv2.putText(frame, 'HELD', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        self._draw_crosshair(frame)
        return eu, ev, ea, conf, frame, mask

    def _best_contour(self, contours):
        """Return the largest contour that passes area and solidity filters."""
        if not contours:
            return None
        # Sort by area descending so we short-circuit on the first valid blob
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < self._min_contour_area:
                break  # remaining contours are even smaller
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity >= self._min_solidity:
                return c
        return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _draw_stats(self, frame, x, y, eu, ev, ea, conf):
        for line, txt in enumerate([
            f'eu={eu:.0f}px  ev={ev:.0f}px',
            f'ea={ea:.1f}',
            f'conf={conf:.2f}',
        ]):
            cv2.putText(frame, txt, (x + 10, y - 10 + line * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    def _draw_crosshair(self, frame):
        cx, cy = int(self._cx), int(self._cy)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 0), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 0), 1)

    def _publish_debug(self, frame, stamp):
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._dbg_quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera_optical_frame'
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self._dbg_pub.publish(msg)

    def _publish_mask(self, mask, stamp):
        h, w = mask.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera_optical_frame'
        msg.height = h
        msg.width = w
        msg.encoding = 'mono8'
        msg.is_bigendian = 0
        msg.step = w
        msg.data = mask.tobytes()
        self._mask_pub.publish(msg)


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
