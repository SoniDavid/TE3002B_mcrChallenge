#!/usr/bin/env python3
"""
IMX219 CSI camera publisher — DUAL hardware-scaled streams.

Publishes:
  /camera/image_raw    (sensor_msgs/Image, BEST_EFFORT)  FULL-res BGR (1280x720) — for
                       off-board YOLO / recording. NOT undistorted (undistort off-board).
  /camera/image_small  (sensor_msgs/Image, BEST_EFFORT)  hardware-downscaled BGR
                       (default 480x270) — for the on-board line follower. Optionally
                       lens-undistorted (undistort_enabled).
  /camera/camera_info  (sensor_msgs/CameraInfo)          optional intrinsics (full-res K).

Why two streams (CPU-starvation fix):
  The Jetson Nano was CPU-bound (4 cores ~95%, GPU idle) because the node undistorted and
  published a full 1280x720 frame every tick, yet the line follower only uses a 320x80
  ROI. nvvidconv downscales on the VIC HARDWARE (~0 CPU), so a `tee` feeds two
  hardware-scaled appsinks: a full stream for YOLO and a small cheap stream for the
  follower. The expensive full-res cv2.remap (undistort) is removed from the full path
  entirely; the small path's undistort (if enabled) is cheap at low res.

Threading:
  Native GStreamer (gi) pipeline with a tee + two appsinks. Each appsink fires a
  `new-sample` callback on a GStreamer streaming thread; the callback maps the buffer,
  (optionally undistorts the small one), and publishes via the rclpy buffer fast-path.
  A GLib.MainLoop in a daemon thread services the pipeline bus. rclpy.spin runs the node.

The array.array fast-path is critical: rclpy iterates Image.data byte-by-byte through a
Python isinstance check when the field is plain bytes (~1.3 s per 1280x720 publish).
array.array activates the buffer protocol and drops that to ~8 ms.
"""

import array
import json
import queue
import threading

import numpy as np
import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

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
# from slow subscribers (e.g. ros2 bag record writing to disk).
_IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class CameraRawPublisher(Node):

    def __init__(self):
        super().__init__('camera_raw_publisher')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('sensor_id',           0)
        self.declare_parameter('width',               1280)   # sensor capture size
        self.declare_parameter('height',              720)
        self.declare_parameter('out_width',           1280)   # FULL published size
        self.declare_parameter('out_height',          720)
        self.declare_parameter('small_width',         480)    # downscaled (follower) size
        self.declare_parameter('small_height',        270)
        self.declare_parameter('framerate',           30)
        self.declare_parameter('flip_method',         0)
        self.declare_parameter('frame_id',            'camera_optical_frame')
        self.declare_parameter('topic_raw',           '/camera/image_raw')
        self.declare_parameter('topic_small',         '/camera/image_small')
        # Publish the FULL-res raw frame (2.76 MB/msg). Default OFF: it back-pressures
        # the GStreamer pipeline (publish + DDS transport + bag recording on the same
        # streaming thread) and froze the WHOLE graph for 1-18 s on the yolo bags. When
        # YOLO is off-board it consumes the JPEG /camera/image_compressed instead, so the
        # raw Image is not needed on the robot. The full GStreamer branch still runs (the
        # compressed encode + camera_info ride on it) — only the raw Image publish is
        # skipped. Set true only if a subscriber genuinely needs the uncompressed frame.
        self.declare_parameter('publish_raw',         False)
        self.declare_parameter('undistort_file',      '')
        self.declare_parameter('undistort_enabled',   True)   # applies to SMALL stream only
        self.declare_parameter('publish_camera_info', False)
        self.declare_parameter('camera_info_file',    '')
        self.declare_parameter('topic_camera_info',   '/camera/camera_info')
        # Optional JPEG-compressed stream off the FULL frame (for off-board YOLO on the
        # laptop GPU). Default OFF so the on-device line follower pays nothing; enable with
        # publish_compressed:=true. Built from the same GStreamer 'full' appsink frame —
        # no extra camera handle — so it coexists with the raw/small streams (unlike the
        # standalone camera_compressed_publisher, which re-opens the CSI camera).
        self.declare_parameter('publish_compressed',  False)
        self.declare_parameter('jpeg_quality',        75)
        self.declare_parameter('topic_compressed',    '/camera/image_compressed')

        sensor_id          = int(self.get_parameter('sensor_id').value)
        self._width        = int(self.get_parameter('width').value)
        self._height       = int(self.get_parameter('height').value)
        self._out_w        = int(self.get_parameter('out_width').value)
        self._out_h        = int(self.get_parameter('out_height').value)
        self._small_w      = int(self.get_parameter('small_width').value)
        self._small_h      = int(self.get_parameter('small_height').value)
        self._fps          = int(self.get_parameter('framerate').value)
        self._flip_method  = int(self.get_parameter('flip_method').value)
        self._frame_id     = self.get_parameter('frame_id').value
        topic_raw          = self.get_parameter('topic_raw').value
        topic_small        = self.get_parameter('topic_small').value
        self._undist_en    = bool(self.get_parameter('undistort_enabled').value)

        self._publish_raw  = bool(self.get_parameter('publish_raw').value)

        # Use all 4 cores for OpenCV (e.g. the small-stream remap).
        cv2.setNumThreads(4)

        # Only create the raw publisher when raw output is enabled — no idle publisher.
        self._pub_raw   = (self.create_publisher(Image, topic_raw, _IMAGE_QOS)
                           if self._publish_raw else None)
        self._pub_small = self.create_publisher(Image, topic_small, _IMAGE_QOS)
        self.get_logger().info(
            f'Raw full-res publish {"ENABLED" if self._publish_raw else "DISABLED"} '
            f'({topic_raw}) — disabled keeps the heavy 2.76 MB frame off DDS/disk')

        # ── Optional compressed publisher (off the full frame) ────────────────
        self._pub_compressed = None
        self._jpeg_quality   = int(self.get_parameter('jpeg_quality').value)
        if bool(self.get_parameter('publish_compressed').value):
            topic_comp = self.get_parameter('topic_compressed').value
            self._pub_compressed = self.create_publisher(
                CompressedImage, topic_comp, _IMAGE_QOS)
            self.get_logger().info(
                f'Compressed publisher ENABLED → {topic_comp} (JPEG q={self._jpeg_quality})')

        # ── Undistort map for the SMALL stream (scaled K) ─────────────────────
        # The full stream is published RAW (YOLO undistorts off-board), so the costly
        # full-res remap is gone. The small stream optionally undistorts at low res; the
        # intrinsics K are calibrated at (width x height), so scale them to (small_w x
        # small_h). dist coeffs are dimensionless (radial) → unchanged.
        self._small_map1 = self._small_map2 = None
        undistort_file = self.get_parameter('undistort_file').value
        if undistort_file and self._undist_en:
            try:
                with open(undistort_file) as f:
                    cal = json.load(f)
                K    = np.array(cal['K'],    dtype=np.float64).reshape(3, 3)
                dist = np.array(cal['dist'], dtype=np.float64)
                sx = self._small_w / float(self._width)
                sy = self._small_h / float(self._height)
                Ks = K.copy()
                Ks[0, 0] *= sx; Ks[0, 2] *= sx   # fx, cx
                Ks[1, 1] *= sy; Ks[1, 2] *= sy   # fy, cy
                self._small_map1, self._small_map2 = cv2.initUndistortRectifyMap(
                    Ks, dist, None, Ks, (self._small_w, self._small_h), cv2.CV_16SC2)
                self.get_logger().info(
                    f'Undistortion ENABLED on {topic_small} '
                    f'({self._small_w}x{self._small_h}, K scaled by {sx:.3f}x{sy:.3f}, '
                    f'RMSE={cal.get("rmse", "?")} px)')
            except Exception as e:
                self.get_logger().warning(
                    f'Could not load undistort_file "{undistort_file}": {e}')
        else:
            self.get_logger().info(
                f'Undistortion DISABLED on {topic_small} '
                f'(undistort_enabled={self._undist_en}, file={"set" if undistort_file else "empty"})')

        # ── CameraInfo publisher (full-res K) ─────────────────────────────────
        self._pub_camera_info = None
        self._camera_info_msg = None
        if self.get_parameter('publish_camera_info').value:
            info_file = self.get_parameter('camera_info_file').value
            if info_file:
                try:
                    self._camera_info_msg = self._load_camera_info(info_file)
                    self._pub_camera_info = self.create_publisher(
                        CameraInfo, self.get_parameter('topic_camera_info').value,
                        _RELIABLE_QOS)
                    self.get_logger().info(f'CameraInfo loaded: {info_file}')
                except Exception as e:
                    self.get_logger().warning(f'Could not load camera_info_file "{info_file}": {e}')
            else:
                self.get_logger().warning(
                    'publish_camera_info=true but camera_info_file is empty — skipping')

        # ── FPS stats ─────────────────────────────────────────────────────────
        self._n_full = 0
        self._n_small = 0
        self._t_stats = self.get_clock().now()

        # ── Publisher worker threads (stall isolation) ────────────────────────
        # CRITICAL FIX: previously _on_sample published (and JPEG-encoded) INLINE on the
        # GStreamer streaming thread. A slow subscriber / DDS transport / bag-record disk
        # write then back-pressured that thread, which blocked the appsink and froze the
        # WHOLE pipeline — BOTH streams + the rest of the graph (observed 1-18 s stalls on
        # the yolo bags). Now the streaming callback only copies the frame into a depth-1
        # queue (drop-oldest) and returns immediately; a dedicated worker thread per
        # stream does the publish/encode. If a worker blocks, frames are dropped (latest
        # wins) but CAPTURE NEVER STALLS — the small stream the follower depends on keeps
        # flowing. depth-1 + drop-oldest matches the appsink's own max-buffers=1/drop.
        self._run_workers = True
        self._q_small = queue.Queue(maxsize=1)
        self._q_full  = queue.Queue(maxsize=1)
        self._worker_small = threading.Thread(
            target=self._publish_worker, args=('small',), name='pub_small', daemon=True)
        self._worker_full = threading.Thread(
            target=self._publish_worker, args=('full',), name='pub_full', daemon=True)
        self._worker_small.start()
        self._worker_full.start()

        # ── GStreamer pipeline: tee → two hardware-scaled appsinks ────────────
        Gst.init(None)
        self._pipeline = Gst.parse_launch(self._pipeline_str(sensor_id))

        self._sink_full  = self._pipeline.get_by_name('full')
        self._sink_small = self._pipeline.get_by_name('small')
        for sink in (self._sink_full, self._sink_small):
            sink.set_property('emit-signals', True)
            sink.set_property('max-buffers', 1)
            sink.set_property('drop', True)
            sink.set_property('sync', False)
        self._sink_full.connect('new-sample',  self._on_sample, 'full')
        self._sink_small.connect('new-sample', self._on_sample, 'small')

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self._on_bus)

        self._glib_loop = GLib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._glib_loop.run, name='gst_mainloop', daemon=True)
        self._glib_thread.start()

        rc = self._pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info(
            f'Camera pipeline PLAYING ({rc.value_nick})  '
            f'{self._width}x{self._height} -> full {self._out_w}x{self._out_h} '
            f'({topic_raw}) + small {self._small_w}x{self._small_h} ({topic_small})')

        self.create_timer(5.0, self._log_stats)

    # ── GStreamer pipeline string ──────────────────────────────────────────────

    def _sensor_framerate(self) -> int:
        # The 1280x720 sensor mode supports 60 fps; otherwise use the requested fps.
        return 60 if (self._width == 1280 and self._height == 720) else self._fps

    def _pipeline_str(self, sensor_id: int) -> str:
        sfps = self._sensor_framerate()
        # nvarguscamerasrc (ISP) -> NVMM -> tee -> two nvvidconv (VIC hardware) branches.
        # Each branch: flip + scale to its target size in hardware, then a CPU videoconvert
        # to BGR for the appsink. The FULL branch keeps out_w x out_h; the SMALL branch
        # scales to small_w x small_h (the big CPU saving — every downstream op is ~Nx
        # cheaper on the small frame).
        return (
            f'nvarguscamerasrc sensor-id={sensor_id} ! '
            f'video/x-raw(memory:NVMM), width={self._width}, height={self._height}, '
            f'framerate={sfps}/1 ! tee name=t '
            f't. ! queue max-size-buffers=1 leaky=downstream ! '
            f'nvvidconv flip-method={self._flip_method} ! '
            f'video/x-raw, width={self._out_w}, height={self._out_h}, format=BGRx ! '
            f'videoconvert ! video/x-raw, format=BGR ! '
            f'appsink name=full emit-signals=true max-buffers=1 drop=true sync=false '
            f't. ! queue max-size-buffers=1 leaky=downstream ! '
            f'nvvidconv flip-method={self._flip_method} ! '
            f'video/x-raw, width={self._small_w}, height={self._small_h}, format=BGRx ! '
            f'videoconvert ! video/x-raw, format=BGR ! '
            f'appsink name=small emit-signals=true max-buffers=1 drop=true sync=false'
        )

    # ── appsink callback ────────────────────────────────────────────────────────

    def _on_sample(self, sink, which):
        """Pull one frame and hand it to the publisher worker (runs on a GStreamer thread).

        This MUST stay light and non-blocking: it copies the frame out of the mapped
        buffer and drops it into a depth-1 queue (drop-oldest), then returns. All
        publishing/encoding happens on the worker thread, so a slow subscriber or disk
        write can never back-pressure capture and freeze the pipeline.
        """
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK
        buf  = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value('width')
        h = caps.get_value('height')
        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            # Copy out of the mapped buffer (freed on unmap). BGR, tightly packed.
            frame = np.frombuffer(mi.data, np.uint8, w * h * 3).reshape(h, w, 3).copy()
        finally:
            buf.unmap(mi)

        stamp = self.get_clock().now().to_msg()
        q = self._q_small if which == 'small' else self._q_full
        # Drop-oldest: if the worker is still busy, discard the stale queued frame and
        # enqueue the newest so we never block the GStreamer thread.
        try:
            q.put_nowait((frame, stamp))
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait((frame, stamp))
            except queue.Full:
                pass
        return Gst.FlowReturn.OK

    def _publish_worker(self, which):
        """Dedicated per-stream publisher thread — drains the queue and publishes/encodes.

        Isolated from the GStreamer streaming thread so transport/disk stalls drop frames
        (latest-wins) instead of freezing capture.
        """
        q = self._q_small if which == 'small' else self._q_full
        while self._run_workers:
            try:
                frame, stamp = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:                       # shutdown sentinel
                break
            try:
                if which == 'small':
                    if self._small_map1 is not None:
                        frame = cv2.remap(frame, self._small_map1, self._small_map2,
                                          cv2.INTER_LINEAR)
                    self._publish(self._pub_small, frame, stamp)
                    self._n_small += 1
                else:
                    self._n_full += 1
                    # Raw full-res Image publish — only when explicitly enabled (the heavy
                    # 2.76 MB frame that caused the whole-graph stalls). The full GStreamer
                    # branch keeps running regardless so the compressed/camera_info below
                    # still get a frame.
                    if self._pub_raw is not None:
                        self._publish(self._pub_raw, frame, stamp)
                    # JPEG-compress the full frame for off-board YOLO — only when enabled
                    # AND someone is subscribed, so it costs nothing otherwise.
                    if (self._pub_compressed is not None
                            and self._pub_compressed.get_subscription_count() > 0):
                        ok, jbuf = cv2.imencode(
                            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                        if ok:
                            cmsg = CompressedImage()
                            cmsg.header.stamp    = stamp
                            cmsg.header.frame_id = self._frame_id
                            cmsg.format = 'jpeg'
                            cmsg.data   = array.array('B', jbuf.tobytes())
                            self._pub_compressed.publish(cmsg)
                    if (self._pub_camera_info is not None
                            and self._camera_info_msg is not None):
                        self._camera_info_msg.header.stamp = stamp
                        self._pub_camera_info.publish(self._camera_info_msg)
            except Exception as e:
                self.get_logger().warning(f'{which} publish worker error: {e}')

    def _publish(self, pub, frame, stamp):
        h, w, c = frame.shape
        msg = Image()
        msg.header.stamp    = stamp
        msg.header.frame_id = self._frame_id
        msg.height       = h
        msg.width        = w
        msg.encoding     = 'bgr8'
        msg.is_bigendian = 0
        msg.step         = w * c
        # array.array activates rclpy's buffer fast-path (avoids the per-byte loop).
        msg.data = array.array('B', frame.tobytes())
        pub.publish(msg)

    # ── bus + stats + cleanup ───────────────────────────────────────────────────

    def _on_bus(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            self.get_logger().error(f'GStreamer ERROR: {err} | {dbg}')
        elif t == Gst.MessageType.EOS:
            self.get_logger().error('GStreamer end-of-stream (camera dropped).')
        elif t == Gst.MessageType.WARNING:
            warn, dbg = message.parse_warning()
            self.get_logger().warning(f'GStreamer WARN: {warn} | {dbg}')
        return True

    def _load_camera_info(self, path: str) -> CameraInfo:
        import yaml
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

    def _log_stats(self):
        now = self.get_clock().now()
        dt = (now - self._t_stats).nanoseconds * 1e-9
        if dt > 0:
            self.get_logger().info(
                f'camera: full={self._n_full / dt:.1f} fps  '
                f'small={self._n_small / dt:.1f} fps')
        self._n_full = self._n_small = 0
        self._t_stats = now

    def destroy_node(self):
        try:
            if hasattr(self, '_pipeline') and self._pipeline is not None:
                self._pipeline.set_state(Gst.State.NULL)
            if hasattr(self, '_glib_loop') and self._glib_loop is not None:
                self._glib_loop.quit()
            # Stop the publisher workers (sentinel + join).
            if getattr(self, '_run_workers', False):
                self._run_workers = False
                for q in (getattr(self, '_q_small', None), getattr(self, '_q_full', None)):
                    if q is not None:
                        try:
                            q.put_nowait((None, None))
                        except queue.Full:
                            pass
                for w in (getattr(self, '_worker_small', None),
                          getattr(self, '_worker_full', None)):
                    if w is not None:
                        w.join(timeout=1.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraRawPublisher()
        rclpy.spin(node)
    except (RuntimeError, KeyboardInterrupt):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
