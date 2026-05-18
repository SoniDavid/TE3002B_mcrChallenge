#!/usr/bin/env python3
"""
Color detector node for traffic light detection.

Subscribes:
  /camera/image_compressed  (sensor_msgs/CompressedImage)

Publishes:
  /traffic_light_color         (std_msgs/String)  "red" | "green" | "yellow" | "none"
  /traffic_detector/debug_image (sensor_msgs/Image)  annotated frame
"""

import signal
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

_BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# BGR colors for debug annotation
_DEBUG_COLORS = {
    'red':    (0,   0,   220),
    'green':  (0,   200, 0),
    'yellow': (0,   200, 200),
    'none':   (150, 150, 150),
}


class ColorDetectorNode(Node):

    def __init__(self):
        super().__init__('color_detector_node')

        # HSV range params
        self.declare_parameter('red_h_low1',  0)
        self.declare_parameter('red_h_high1', 10)
        self.declare_parameter('red_h_low2',  165)
        self.declare_parameter('red_h_high2', 180)
        self.declare_parameter('red_s_min',   100)
        self.declare_parameter('red_v_min',   80)

        self.declare_parameter('green_h_low',  45)
        self.declare_parameter('green_h_high', 85)
        self.declare_parameter('green_s_min',  80)
        self.declare_parameter('green_v_min',  80)

        self.declare_parameter('yellow_h_low',  18)
        self.declare_parameter('yellow_h_high', 35)
        self.declare_parameter('yellow_s_min',  120)
        self.declare_parameter('yellow_v_min',  100)

        self.declare_parameter('min_blob_area',  800.0)
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('loop_hz',        30.0)

        self._min_area    = float(self.get_parameter('min_blob_area').value)
        self._confirm_n   = int(self.get_parameter('confirm_frames').value)

        # Confirmation filter state
        self._candidate       = 'none'   # color being accumulated
        self._candidate_count = 0        # consecutive frames of candidate
        self._confirmed_color = 'none'   # last published color

        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.create_subscription(
            CompressedImage,
            '/camera/image_compressed',
            self._image_callback,
            _BEST_EFFORT_QOS,
        )

        self._color_pub = self.create_publisher(String, '/traffic_light_color', 10)
        self._debug_pub = self.create_publisher(Image, '/traffic_detector/debug_image', 1)

        self.get_logger().info(
            f'ColorDetectorNode ready — min_blob_area={self._min_area}, '
            f'confirm_frames={self._confirm_n}'
        )

    # ------------------------------------------------------------------ helpers

    def _hsv_ranges(self):
        """Return per-color list of (lower, upper) HSV numpy arrays."""
        g = self.get_parameter
        red = [
            (np.array([g('red_h_low1').value,  g('red_s_min').value, g('red_v_min').value]),
             np.array([g('red_h_high1').value, 255, 255])),
            (np.array([g('red_h_low2').value,  g('red_s_min').value, g('red_v_min').value]),
             np.array([g('red_h_high2').value, 255, 255])),
        ]
        green = [
            (np.array([g('green_h_low').value,  g('green_s_min').value, g('green_v_min').value]),
             np.array([g('green_h_high').value, 255, 255])),
        ]
        yellow = [
            (np.array([g('yellow_h_low').value,  g('yellow_s_min').value, g('yellow_v_min').value]),
             np.array([g('yellow_h_high').value, 255, 255])),
        ]
        return {'red': red, 'green': green, 'yellow': yellow}

    def _largest_blob_area(self, mask) -> float:
        """Return the area of the largest contour in the binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return float(max(cv2.contourArea(c) for c in contours))

    def _build_mask(self, hsv, ranges):
        """OR together all (lower, upper) range masks and apply morphological open."""
        combined = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, lower, upper)
            combined = m if combined is None else cv2.bitwise_or(combined, m)
        # Morphological open: removes isolated noise pixels
        return cv2.morphologyEx(combined, cv2.MORPH_OPEN, self._morph_kernel, iterations=2)

    def _draw_debug(self, frame, detected_color, areas):
        """Annotate frame with detected color and blob areas."""
        bgr = _DEBUG_COLORS.get(detected_color, (150, 150, 150))
        cv2.putText(frame, f'Color: {detected_color.upper()}',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bgr, 2)
        y = 65
        for name, area in areas.items():
            cv2.putText(frame, f'{name}: {area:.0f}px2',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        _DEBUG_COLORS[name], 1)
            y += 22
        return frame

    # ------------------------------------------------------------------ callback

    def _image_callback(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Bilateral filter: reduces noise while preserving color edges (better than Gaussian for screen glare)
        blurred = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # CLAHE on V channel normalizes brightness variation from screen angle/glare
        h, s, v = cv2.split(hsv)
        v = self._clahe.apply(v)
        hsv = cv2.merge([h, s, v])

        ranges = self._hsv_ranges()
        areas = {}
        for color_name, color_ranges in ranges.items():
            mask = self._build_mask(hsv, color_ranges)
            areas[color_name] = self._largest_blob_area(mask)

        # Winning color: largest area above threshold
        best_color = 'none'
        best_area = self._min_area
        for color_name, area in areas.items():
            if area > best_area:
                best_area = area
                best_color = color_name

        # Confirmation filter: require N consecutive frames
        if best_color == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = best_color
            self._candidate_count = 1

        if self._candidate_count >= self._confirm_n:
            self._confirmed_color = self._candidate

        # Publish confirmed color
        color_msg = String()
        color_msg.data = self._confirmed_color
        self._color_pub.publish(color_msg)

        # Publish debug image if anyone is listening
        if self._debug_pub.get_subscription_count() > 0:
            annotated = self._draw_debug(frame.copy(), self._confirmed_color, areas)
            debug_msg = Image()
            debug_msg.header = msg.header
            debug_msg.height = annotated.shape[0]
            debug_msg.width  = annotated.shape[1]
            debug_msg.encoding = 'bgr8'
            debug_msg.step = annotated.shape[1] * 3
            debug_msg.data = annotated.tobytes()
            self._debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectorNode()

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
