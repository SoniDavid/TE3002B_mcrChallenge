#!/usr/bin/env python3
"""
IMX219 CSI camera publisher — raw only.

Publishes:
  /camera/image_raw   (sensor_msgs/Image, BEST_EFFORT)  raw BGR
  /camera/camera_info (sensor_msgs/CameraInfo)           optional intrinsics

Threading model (adapted from Team2's rectify_compress.py):
  - Capture thread: GStreamer → frame → writes to shared slot, notifies Condition.
  - Publish thread: waits on Condition, publishes Image using array.array fast-path.

The array.array fast-path is critical: rclpy iterates Image.data byte-by-byte through
a Python isinstance check when the field is a plain bytes object.  For 1280×720 this
takes ~1.3 s per publish, capping the effective rate at < 1 Hz.
array.array activates the rclpy buffer protocol path and reduces this to ~8 ms.
"""

import array
import threading
import time
import yaml

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, CompressedImage


_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Camera images are a streaming sensor type — BEST_EFFORT avoids ACK backpressure
# from slow subscribers (e.g. ros2 bag record writing to disk),
# which would otherwise throttle the publish rate.
_IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class CameraRawPublisher(Node):

    def __init__(self):
        super().__init__('camera_raw_publisher')

        self.declare_parameter('sensor_id',           0)
        self.declare_parameter('width',               1280)
        self.declare_parameter('height',              720)
        self.declare_parameter('out_width',           1280)
        self.declare_parameter('out_height',          720)
        self.declare_parameter('framerate',           30)
        self.declare_parameter('flip_method',         0)
        self.declare_parameter('frame_id',            'camera_optical_frame')
        self.declare_parameter('topic_raw',           '/camera/image_raw')
        self.declare_parameter('color_cal_file',      '')
        self.declare_parameter('publish_camera_info', False)
        self.declare_parameter('camera_info_file',    '')
        self.declare_parameter('topic_camera_info',   '/camera/camera_info')
        self.declare_parameter('publish_compressed',  False)
        self.declare_parameter('jpeg_quality',        75)
        self.declare_parameter('topic_compressed',    '/camera/image_compressed')

        sensor_id         = self.get_parameter('sensor_id').value
        self._width       = self.get_parameter('width').value
        self._height      = self.get_parameter('height').value
        self._out_w       = self.get_parameter('out_width').value
        self._out_h       = self.get_parameter('out_height').value
        self._fps         = self.get_parameter('framerate').value
        self._flip_method = int(self.get_parameter('flip_method').value)
        self._frame_id    = self.get_parameter('frame_id').value
        topic_raw         = self.get_parameter('topic_raw').value

        # Use all 4 Jetson Nano cores for OpenCV operations (Team2 technique).
        cv2.setNumThreads(4)

        self._pub_raw = self.create_publisher(Image, topic_raw, _IMAGE_QOS)

        # ── Color calibration ─────────────────────────────────────────────────
        self._color_gains = None
        cal_file = self.get_parameter('color_cal_file').value
        if cal_file:
            try:
                cal = np.load(cal_file)
                gains = cal['arr_0'].astype(np.float32)
                self.get_logger().info(
                    f'Color calibration loaded: shape={gains.shape}, file={cal_file}')
                if gains.ndim == 3 and gains.shape != (self._out_h, self._out_w, 3):
                    self.get_logger().warning(
                        f'color_gains shape {gains.shape} != frame shape '
                        f'({self._out_h},{self._out_w},3) — disabling color correction')
                    gains = None
                self._color_gains = gains
            except Exception as e:
                self.get_logger().warning(f'Could not load color_cal_file "{cal_file}": {e}')

        # ── CameraInfo publisher ──────────────────────────────────────────────
        self._pub_camera_info = None
        self._camera_info_msg = None
        if self.get_parameter('publish_camera_info').value:
            info_file = self.get_parameter('camera_info_file').value
            if info_file:
                try:
                    self._camera_info_msg = self._load_camera_info(info_file)
                    self._pub_camera_info = self.create_publisher(
                        CameraInfo,
                        self.get_parameter('topic_camera_info').value,
                        _RELIABLE_QOS,
                    )
                    self.get_logger().info(f'CameraInfo loaded: {info_file}')
                except Exception as e:
                    self.get_logger().warning(f'Could not load camera_info_file "{info_file}": {e}')
            else:
                self.get_logger().warning(
                    'publish_camera_info=true but camera_info_file is empty — skipping')

        # ── Optional compressed publisher ─────────────────────────────────────
        self._pub_compressed = None
        self._jpeg_quality   = self.get_parameter('jpeg_quality').value
        if self.get_parameter('publish_compressed').value:
            topic_comp = self.get_parameter('topic_compressed').value
            self._pub_compressed = self.create_publisher(
                CompressedImage, topic_comp, _IMAGE_QOS)
            self.get_logger().info(
                f'Compressed publisher ready  → {topic_comp}  (JPEG q={self._jpeg_quality})')

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
            f'Camera ready  {self._width}x{self._height}'
            f'  -> published at {self._out_w}x{self._out_h}'
            f'  -> {topic_raw}'
        )

        # ── Shared state (Condition-based, Team2 pattern) ─────────────────────
        # One Condition guards the shared slot.  The capture thread writes into
        # _slot and increments _slot_seq, then notifies.  The publish thread
        # wakes, grabs the reference, and publishes.  If the publish thread is
        # slower than capture, the slot is simply overwritten — newest frame wins.
        self._capture_thread_running = True
        self._slot_cv   = threading.Condition()
        self._slot      = None   # (stamp, frame ndarray) or None
        self._slot_seq  = 0      # incremented on every captured frame
        self._pub_seq   = -1     # sequence last published

        # Reconnect counters (kept from original)
        self._consecutive_failures  = 0
        self._reconnect_attempts    = 0
        self._MAX_CONSECUTIVE_FAILURES = 10
        self._MAX_RECONNECT_ATTEMPTS   = 5
        self._RECONNECT_SLEEP_SEC      = 3.0

        # FPS statistics (Team2 pattern)
        self._cap_frames = 0
        self._pub_frames = 0
        self._t_stats    = self.get_clock().now()

        self._cap_thread = threading.Thread(
            target=self._capture_loop, name='camera_capture', daemon=True)
        self._pub_thread = threading.Thread(
            target=self._publish_loop, name='camera_publish', daemon=True)

        self._cap_thread.start()
        self._pub_thread.start()

        # Stats every 5 s — instant visibility into achieved fps (Team2 pattern).
        self.create_timer(5.0, self._log_stats)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_camera_info(self, path: str) -> CameraInfo:
        with open(path) as f:
            d = yaml.safe_load(f)
        msg = CameraInfo()
        msg.header.frame_id = self._frame_id
        msg.width  = d['image_width']
        msg.height = d['image_height']
        msg.distortion_model = d['distortion_model']
        msg.d = d['distortion_coefficients']['data']
        msg.k = d['camera_matrix']['data']
        msg.r = d['rectification_matrix']['data']
        msg.p = d['projection_matrix']['data']
        return msg

    def _sensor_framerate(self) -> int:
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
            f'video/x-raw, width={self._out_w}, height={self._out_h}, format=BGRx ! '
            f'videoconvert ! '
            f'video/x-raw, format=BGR ! '
            f'appsink drop=true max-buffers=1 sync=false'
        )

    # ── Background capture loop ───────────────────────────────────────────────

    def _capture_loop(self):
        self.get_logger().info('Capture thread started.')
        while self._capture_thread_running:
            ret, frame = self._cap.read()
            if ret:
                self._consecutive_failures = 0
                if self._color_gains is not None:
                    try:
                        frame = (frame.astype(np.float32) * self._color_gains).clip(0, 255).astype(np.uint8)
                    except Exception as e:
                        self.get_logger().warning(f'Color correction failed: {e} — disabling')
                        self._color_gains = None
                stamp = self.get_clock().now().to_msg()
                with self._slot_cv:
                    self._slot = (stamp, frame)
                    self._slot_seq += 1
                    self._slot_cv.notify_all()   # wake the publish thread
                self._cap_frames += 1
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
                        with self._slot_cv:
                            self._slot_cv.notify_all()
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

    # ── Dedicated publish loop (Team2 pattern) ────────────────────────────────

    def _publish_loop(self):
        """Publish thread — wakes on each new captured frame via Condition.

        cv2 and rclpy publish both release the GIL, so this thread can run on
        a separate core while the capture thread continues uninterrupted.
        The array.array fast-path avoids rclpy's byte-by-byte isinstance loop.
        """
        self.get_logger().info('Publish thread started.')
        while self._capture_thread_running:
            # Block until there is a frame we haven't published yet.
            with self._slot_cv:
                while (self._capture_thread_running
                       and self._slot_seq == self._pub_seq):
                    self._slot_cv.wait(timeout=1.0)
                if not self._capture_thread_running:
                    break
                self._pub_seq = self._slot_seq
                stamp, frame = self._slot   # grab reference under lock

            # Build and publish outside the lock.
            h, w, c = frame.shape
            msg = Image()
            msg.header.stamp    = stamp
            msg.header.frame_id = self._frame_id
            msg.height     = h
            msg.width      = w
            msg.encoding   = 'bgr8'
            msg.is_bigendian = 0
            msg.step       = w * c
            # array.array activates rclpy's buffer fast-path — avoids the
            # per-byte isinstance loop that takes ~108 ms for 320×240.
            msg.data = array.array('B', frame.tobytes())
            self._pub_raw.publish(msg)

            if (self._pub_compressed is not None
                    and self._pub_compressed.get_subscription_count() > 0):
                ok, buf = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                if ok:
                    cmsg = CompressedImage()
                    cmsg.header.stamp    = stamp
                    cmsg.header.frame_id = self._frame_id
                    cmsg.format = 'jpeg'
                    cmsg.data   = array.array('B', buf.tobytes())
                    self._pub_compressed.publish(cmsg)

            if self._pub_camera_info is not None and self._camera_info_msg is not None:
                self._camera_info_msg.header.stamp = stamp
                self._pub_camera_info.publish(self._camera_info_msg)

            self._pub_frames += 1

        self.get_logger().info('Publish thread exiting.')

    # ── FPS statistics (Team2 pattern) ────────────────────────────────────────

    def _log_stats(self):
        now = self.get_clock().now()
        dt  = (now - self._t_stats).nanoseconds * 1e-9
        if dt > 0:
            self.get_logger().info(
                f'camera: cap={self._cap_frames / dt:.1f}  '
                f'pub={self._pub_frames / dt:.1f} fps')
        self._cap_frames = self._pub_frames = 0
        self._t_stats = now

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._capture_thread_running = False
        # Wake both threads so they can exit their wait loops.
        with self._slot_cv:
            self._slot_cv.notify_all()
        for t in (getattr(self, '_cap_thread', None),
                  getattr(self, '_pub_thread', None)):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()
            self.get_logger().info(f'Camera released after {self._pub_frames} published frames.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraRawPublisher()
        rclpy.spin(node)
    except RuntimeError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
