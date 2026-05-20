#!/usr/bin/env python3
"""
Line follower node for Puzzlebot.

Subscribes:
  /camera/image_compressed  (sensor_msgs/CompressedImage)

Publishes:
  /line_follower/cx           (std_msgs/Int32)    detected center x in pixels
  /line_follower/error        (std_msgs/Float32)  error from image center (px)
  /line_follower/line_type    (std_msgs/String)   "solid" | "dashed"
  /line_follower/debug_image  (sensor_msgs/Image) pipeline debug visualization
  /cmd_vel_desired            (geometry_msgs/Twist)
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32, Float32, String
from geometry_msgs.msg import Twist

from pzb_line_follower_scripts.center_line_detector import CenterLineDetector

_BEST_EFFORT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class LineFollowerNode(Node):

    def __init__(self):
        super().__init__('line_follower_node')

        self.declare_parameter('image_width',    320)
        self.declare_parameter('image_height',   240)
        self.declare_parameter('Kp_angular',     0.003)
        self.declare_parameter('dead_band_px',   8)
        self.declare_parameter('linear_speed',   0.10)
        self.declare_parameter('max_angular',    0.8)
        self.declare_parameter('stop_on_dashed', False)
        self.declare_parameter('publish_debug',  True)
        self.declare_parameter('topic_image_in', '/camera/image_compressed')
        self.declare_parameter('topic_cmd_vel',  '/cmd_vel_desired')

        self._img_w        = self.get_parameter('image_width').value
        self._img_h        = self.get_parameter('image_height').value
        self._kp           = float(self.get_parameter('Kp_angular').value)
        self._dead_band    = int(self.get_parameter('dead_band_px').value)
        self._linear_speed = float(self.get_parameter('linear_speed').value)
        self._max_ang      = float(self.get_parameter('max_angular').value)
        self._stop_dashed  = bool(self.get_parameter('stop_on_dashed').value)
        self._pub_debug    = bool(self.get_parameter('publish_debug').value)

        topic_in  = self.get_parameter('topic_image_in').value
        topic_cmd = self.get_parameter('topic_cmd_vel').value

        self._detector = CenterLineDetector(debug=self._pub_debug)

        # Publishers
        self._pub_cx        = self.create_publisher(Int32,   '/line_follower/cx',        _RELIABLE_QOS)
        self._pub_error     = self.create_publisher(Float32, '/line_follower/error',      _RELIABLE_QOS)
        self._pub_line_type = self.create_publisher(String,  '/line_follower/line_type',  _RELIABLE_QOS)
        self._pub_cmd       = self.create_publisher(Twist,   topic_cmd,                   _RELIABLE_QOS)
        self._pub_debug_img = self.create_publisher(Image,   '/line_follower/debug_image', _RELIABLE_QOS)

        # Subscriber — BEST_EFFORT to match camera_publisher output
        self.create_subscription(CompressedImage, topic_in, self._image_cb, _BEST_EFFORT_QOS)

        self.get_logger().info(
            f'Line follower ready  img={self._img_w}x{self._img_h}'
            f'  Kp={self._kp}  dead_band={self._dead_band}px'
            f'  v={self._linear_speed} m/s  debug={self._pub_debug}'
        )

    def _image_cb(self, msg: CompressedImage):
        # Decode JPEG
        buf = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warning('Failed to decode CompressedImage', throttle_duration_sec=2.0)
            return

        # Resize to expected detector resolution
        if img.shape[1] != self._img_w or img.shape[0] != self._img_h:
            img = cv2.resize(img, (self._img_w, self._img_h), interpolation=cv2.INTER_LANCZOS4)

        cx, cy = self._detector.detect_center_line(img)
        line_type = self._detector.line_type

        error = float(cx - self._img_w // 2)

        # Steering control (proportional, same as test_simulator.py)
        if abs(error) <= self._dead_band:
            angular_z = 0.0
        else:
            angular_z = float(np.clip(-self._kp * error, -self._max_ang, self._max_ang))

        # Stop linear speed when dashed line detected (intersection), if configured
        linear_x = 0.0 if (self._stop_dashed and line_type == 'dashed') else self._linear_speed

        # Publish detections
        cx_msg = Int32()
        cx_msg.data = cx
        self._pub_cx.publish(cx_msg)

        err_msg = Float32()
        err_msg.data = error
        self._pub_error.publish(err_msg)

        type_msg = String()
        type_msg.data = line_type
        self._pub_line_type.publish(type_msg)

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x  = linear_x
        cmd.angular.z = angular_z
        self._pub_cmd.publish(cmd)

        # Publish debug image if available
        if self._pub_debug and self._detector.debug_frame is not None:
            dbg = self._detector.debug_frame
            dh, dw = dbg.shape[:2]
            dbg_msg = Image()
            dbg_msg.header.stamp = msg.header.stamp
            dbg_msg.header.frame_id = 'camera_optical_frame'
            dbg_msg.height = dh
            dbg_msg.width  = dw
            dbg_msg.encoding = 'bgr8'
            dbg_msg.is_bigendian = 0
            dbg_msg.step = dw * 3
            dbg_msg.data = dbg.tobytes()
            self._pub_debug_img.publish(dbg_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LineFollowerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
