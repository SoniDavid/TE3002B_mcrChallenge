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

import threading
import time

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
        self._flip_method = int(self.get_parameter('flip_method').value)
        self._publish_compressed_enabled = bool(self.get_parameter('publish_compressed').value)
        self._publish_raw_enabled = bool(self.get_parameter('publish_raw').value)
        topic_compressed = self.get_parameter('topic_compressed').value
        topic_raw        = self.get_parameter('topic_raw').value

        # ── Publishers ───────────────────────────────────────────────────────
        compressed_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # RViz Image display defaults to RELIABLE; keep raw compatible.
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
            f'Camera ready  {self._width}x{self._height}@{self._sensor_framerate()}fps(sensor)'
            f'  publish@{self._fps}fps'
            f'  JPEG quality={self._quality}  '
            f'publish_compressed={self._publish_compressed_enabled}  '
            f'publish_raw={self._publish_raw_enabled}'
        )

        # ── Shared capture state ──────────────────────────────────────────────
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_stamp = None
        self._capture_thread_running = True
        self._consecutive_failures = 0
        self._reconnect_attempts = 0
        self._MAX_CONSECUTIVE_FAILURES = 10
        self._MAX_RECONNECT_ATTEMPTS = 5
        self._RECONNECT_SLEEP_SEC = 3.0

        self._cap_thread = threading.Thread(
            target=self._capture_loop, name='camera_capture', daemon=True)
        self._cap_thread.start()

        # ── Timer ─────────────────────────────────────────────────────────────
        # Timer period matches the requested publish framerate.
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
        self.declare_parameter('flip_method',      0)
        self.declare_parameter('topic_compressed', '/camera/image_compressed')
        self.declare_parameter('topic_raw',        '/camera/image_raw')
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('publish_raw',      False)

    # ── GStreamer pipeline builder ────────────────────────────────────────────

    def _sensor_framerate(self) -> int:
        """Return the native IMX219 sensor framerate for the configured resolution.

        At 1280x720 the IMX219 has no 30fps mode; its lowest is 60fps.
        Requesting 30fps causes Argus to pick the 120fps mode instead.
        """
        if self._width == 1280 and self._height == 720:
            return 60
        return self._fps

    def _gstreamer_pipeline(self, sensor_id: int) -> str:
        sensor_fps = self._sensor_framerate()
        return (
            f'nvarguscamerasrc sensor-id={sensor_id} ! '
            f'video/x-raw(memory:NVMM), '
            f'width={self._width}, height={self._height}, '
            f'framerate={sensor_fps}/1 ! '
            f'nvvidconv flip-method={self._flip_method} ! '
            f'video/x-raw, width={self._width}, height={self._height}, format=BGRx ! '
            f'videoconvert ! '
            f'video/x-raw, format=BGR ! '
            f'appsink drop=true max-buffers=2 sync=false'
        )

    # ── Background capture loop ───────────────────────────────────────────────

    def _capture_loop(self):
        self.get_logger().info('Capture thread started.')
        while self._capture_thread_running:
            ret, frame = self._cap.read()
            if ret:
                self._consecutive_failures = 0
                stamp = self.get_clock().now().to_msg()
                with self._lock:
                    self._latest_frame = frame
                    self._latest_stamp = stamp
            else:
                self._consecutive_failures += 1
                self.get_logger().warning(
                    f'cap.read() failed (consecutive={self._consecutive_failures})')
                if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                    if self._reconnect_attempts >= self._MAX_RECONNECT_ATTEMPTS:
                        self.get_logger().error(
                            f'Camera failed after {self._MAX_RECONNECT_ATTEMPTS} '
                            'reconnect attempts. Shutting down.')
                        self._capture_thread_running = False
                        rclpy.shutdown()
                        return
                    self._reconnect_attempts += 1
                    self.get_logger().warning(
                        f'Reconnecting {self._reconnect_attempts}/'
                        f'{self._MAX_RECONNECT_ATTEMPTS} ...')
                    self._cap.release()
                    time.sleep(self._RECONNECT_SLEEP_SEC)
                    sensor_id = self.get_parameter('sensor_id').value
                    self._cap = cv2.VideoCapture(
                        self._gstreamer_pipeline(sensor_id), cv2.CAP_GSTREAMER)
                    if self._cap.isOpened():
                        self.get_logger().info('Reconnect succeeded.')
                        self._consecutive_failures = 0
                    else:
                        self.get_logger().error('Reconnect failed — will retry.')
        self.get_logger().info('Capture thread exiting.')

    # ── Publish callback (non-blocking — reads shared frame reference) ────────

    def _capture_and_publish(self):
        with self._lock:
            frame = self._latest_frame
            stamp = self._latest_stamp

        if frame is None:
            return

        self._frame_count += 1

        if self._publish_compressed_enabled and self._pub_compressed.get_subscription_count() > 0:
            self._publish_compressed(frame, stamp)

        if self._publish_raw_enabled and self._pub_raw.get_subscription_count() > 0:
            self._publish_raw(frame, stamp)

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
        self._capture_thread_running = False
        if hasattr(self, '_cap_thread') and self._cap_thread.is_alive():
            self._cap_thread.join(timeout=2.0)
        if hasattr(self, '_cap') and self._cap.isOpened():
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
