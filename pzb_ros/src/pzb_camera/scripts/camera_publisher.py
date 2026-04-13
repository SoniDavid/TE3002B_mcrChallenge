#!/usr/bin/env python3
"""
IMX219 CSI camera publisher for Jetson Nano.

Publishes:
  /camera/image_compressed  (sensor_msgs/CompressedImage)  -- JPEG, ~30-80 KB/frame
  /camera/image_raw         (sensor_msgs/Image)             -- raw BGR, ~2.7 MB/frame

The compressed topic is the primary one to use over a network.
Raw is kept for local processing nodes (e.g. OpenCV pipelines on the same machine).

GStreamer pipeline used:
  nvarguscamerasrc  ->  nvvidconv  ->  video/x-raw BGRx  ->  videoconvert  ->  BGR appsink
This path is fully hardware-accelerated on the Jetson Nano.
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header


class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        # ── Parameters ───────────────────────────────────────────────────────
        self._declare_params()

        sensor_id    = self.get_parameter('sensor_id').value
        self._width  = self.get_parameter('width').value
        self._height = self.get_parameter('height').value
        self._fps    = self.get_parameter('framerate').value
        self._quality = self.get_parameter('jpeg_quality').value
        self._frame_id = self.get_parameter('frame_id').value
        topic_compressed = self.get_parameter('topic_compressed').value
        topic_raw        = self.get_parameter('topic_raw').value

        # ── Publishers ───────────────────────────────────────────────────────
        self._pub_compressed = self.create_publisher(
            CompressedImage, topic_compressed, 10)
        self._pub_raw = self.create_publisher(
            Image, topic_raw, 10)

        # ── GStreamer pipeline ────────────────────────────────────────────────
        pipeline = self._gstreamer_pipeline(sensor_id)
        self.get_logger().info(f'Opening pipeline:\n  {pipeline}')

        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            self.get_logger().fatal(
                'Failed to open camera. Check that:\n'
                '  1. nvargus-daemon is running  (sudo systemctl start nvargus-daemon)\n'
                '  2. The flex cable is seated correctly\n'
                '  3. No other process has the camera open'
            )
            raise RuntimeError('Camera open failed')

        self.get_logger().info(
            f'Camera ready  {self._width}x{self._height}@{self._fps}fps  '
            f'JPEG quality={self._quality}'
        )

        # ── Timer ─────────────────────────────────────────────────────────────
        # Timer period matches the requested framerate.
        self._timer = self.create_timer(1.0 / self._fps, self._capture_and_publish)

        # ── JPEG encode params ────────────────────────────────────────────────
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]

        self._frame_count = 0

    # ── Parameter declaration ─────────────────────────────────────────────────

    def _declare_params(self):
        self.declare_parameter('sensor_id',        0)
        self.declare_parameter('width',            1280)
        self.declare_parameter('height',           720)
        self.declare_parameter('framerate',        30)
        self.declare_parameter('jpeg_quality',     80)
        self.declare_parameter('frame_id',         'camera_optical_frame')
        self.declare_parameter('topic_compressed', '/camera/image_compressed')
        self.declare_parameter('topic_raw',        '/camera/image_raw')

    # ── GStreamer pipeline builder ────────────────────────────────────────────

    def _gstreamer_pipeline(self, sensor_id: int) -> str:
        return (
            f'nvarguscamerasrc sensor-id={sensor_id} ! '
            f'video/x-raw(memory:NVMM), '
            f'width={self._width}, height={self._height}, '
            f'framerate={self._fps}/1 ! '
            f'nvvidconv ! '
            f'video/x-raw, format=BGRx ! '
            f'videoconvert ! '
            f'video/x-raw, format=BGR ! '
            f'appsink drop=1'
        )

    # ── Capture + publish callback ────────────────────────────────────────────

    def _capture_and_publish(self):
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warning('Frame capture failed — skipping')
            return

        now = self.get_clock().now().to_msg()
        self._frame_count += 1

        # Always publish compressed (primary network topic)
        self._publish_compressed(frame, now)

        # Only publish raw if someone is subscribed (saves CPU when unused)
        if self._pub_raw.get_subscription_count() > 0:
            self._publish_raw(frame, now)

    def _publish_compressed(self, frame: np.ndarray, stamp) -> None:
        ok, buf = cv2.imencode('.jpg', frame, self._encode_params)
        if not ok:
            self.get_logger().warning('JPEG encode failed')
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

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        if self._cap.isOpened():
            self._cap.release()
            self.get_logger().info(
                f'Camera released after {self._frame_count} frames.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraPublisher()
        rclpy.spin(node)
    except RuntimeError as e:
        pass  # Fatal errors already logged inside __init__
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
