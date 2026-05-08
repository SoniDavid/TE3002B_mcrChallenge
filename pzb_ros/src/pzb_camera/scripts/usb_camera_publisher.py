#!/usr/bin/env python3
"""
USB Camera publisher using V4L2.
Separated from the CSI camera publisher for better hardware compatibility.

Publishes:
  /camera/image_compressed  (sensor_msgs/CompressedImage)
  /camera/image_raw         (sensor_msgs/Image)
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import Image, CompressedImage

class USBCameraPublisher(Node):

    def __init__(self):
        super().__init__('usb_camera_publisher')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('device_index', 2)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('framerate', 30.0)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('frame_id', 'usb_camera_frame')
        self.declare_parameter('topic_compressed', '/camera/image_compressed')
        self.declare_parameter('topic_raw', '/camera/image_raw')
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('publish_raw', True)

        device_index = self.get_parameter('device_index').value
        self._width = self.get_parameter('width').value
        self._height = self.get_parameter('height').value
        self._fps = self.get_parameter('framerate').value
        self._quality = self.get_parameter('jpeg_quality').value
        self._frame_id = self.get_parameter('frame_id').value
        self._publish_compressed_enabled = self.get_parameter('publish_compressed').value
        self._publish_raw_enabled = self.get_parameter('publish_raw').value
        topic_compressed = self.get_parameter('topic_compressed').value
        topic_raw = self.get_parameter('topic_raw').value

        # ── Publishers ───────────────────────────────────────────────────────
        compressed_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        raw_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub_compressed = self.create_publisher(
            CompressedImage, topic_compressed, compressed_qos)
        self._pub_raw = self.create_publisher(
            Image, topic_raw, raw_qos)

        # ── Camera Initialization ──────────────────────────────────────────
        self.get_logger().info(f'Opening USB camera at /dev/video{device_index}')
        self._cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        
        if not self._cap.isOpened():
            self.get_logger().fatal(f'Failed to open /dev/video{device_index}.')
            raise RuntimeError('USB Camera open failed')

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Attempt to set FPS
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        self.get_logger().info(
            f'USB Camera ready: {self._width}x{self._height} '
            f'publish_compressed={self._publish_compressed_enabled} '
            f'publish_raw={self._publish_raw_enabled}'
        )

        # ── Timer ─────────────────────────────────────────────────────────────
        self._timer = self.create_timer(1.0 / self._fps, self._capture_and_publish)
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        self._frame_count = 0

    def _capture_and_publish(self):
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warning('Frame capture failed — skipping')
            return

        now = self.get_clock().now().to_msg()
        self._frame_count += 1

        if self._publish_compressed_enabled and self._pub_compressed.get_subscription_count() > 0:
            self._publish_compressed(frame, now)

        if self._publish_raw_enabled and self._pub_raw.get_subscription_count() > 0:
            self._publish_raw(frame, now)

    def _publish_compressed(self, frame: np.ndarray, stamp) -> None:
        ok, buf = cv2.imencode('.jpg', frame, self._encode_params)
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        self._pub_compressed.publish(msg)

    def _publish_raw(self, frame: np.ndarray, stamp) -> None:
        h, w, c = frame.shape
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = h
        msg.width = w
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = w * c
        msg.data = frame.tobytes()
        self._pub_raw.publish(msg)

    def destroy_node(self):
        if self._cap.isOpened():
            self._cap.release()
            self.get_logger().info(f'USB Camera released after {self._frame_count} frames.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = USBCameraPublisher()
        rclpy.spin(node)
    except RuntimeError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
