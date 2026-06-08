#!/usr/bin/env python3
"""
YOLO object detector — runs on the laptop (GPU).

Subscribes to /camera/image_compressed (JPEG) from the Jetson, decompresses it,
runs a custom YOLOv8 model, and publishes an annotated debug image.

When multiple objects pass the confidence threshold, only the closest one is drawn
— closest is approximated by the largest bounding-box area (more pixels = physically
nearer to the camera).
"""

import array

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


_IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Overlay colours (BGR)
_BOX_COLOR  = (0, 255, 0)
_TEXT_COLOR = (0, 0, 0)


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector_node')

        self.declare_parameter('model_path',     '')
        self.declare_parameter('conf_threshold', 0.55)
        self.declare_parameter('iou_threshold',  0.45)
        self.declare_parameter('blur_threshold', 0.0)
        self.declare_parameter('half',           False)
        self.declare_parameter('device',         'cuda')
        self.declare_parameter('input_topic',    '/camera/image_compressed')
        self.declare_parameter('output_topic',   '/yolo/debug_image')
        self.declare_parameter('sign_topic',     '/yolo/sign')

        model_path       = self.get_parameter('model_path').value
        self._conf_thr   = float(self.get_parameter('conf_threshold').value)
        self._iou_thr    = float(self.get_parameter('iou_threshold').value)
        self._blur_thr   = float(self.get_parameter('blur_threshold').value)
        self._half       = bool(self.get_parameter('half').value)
        device           = self.get_parameter('device').value
        in_topic         = self.get_parameter('input_topic').value
        out_topic        = self.get_parameter('output_topic').value
        sign_topic       = self.get_parameter('sign_topic').value

        if not model_path:
            self.get_logger().fatal(
                'model_path parameter is required. '
                'Pass it with: model_path:=/abs/path/to/model.pt')
            raise RuntimeError('model_path not set')

        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().fatal(
                'ultralytics package not found. Install it with: pip install ultralytics')
            raise

        self.get_logger().info(f'Loading YOLO model: {model_path}  device={device}')
        self._model  = YOLO(model_path)
        self._device = device

        # Warm-up: compile CUDA kernels before the first real frame arrives.
        self._model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=device)
        self.get_logger().info('YOLO model ready.')

        self._pub_debug = self.create_publisher(Image, out_topic, _IMAGE_QOS)
        # Detected sign/class for the behavior layer (traffic FSM + line follower).
        # Published every processed frame as the closest (largest-box) class, or 'none'.
        self._pub_sign  = self.create_publisher(String, sign_topic, 10)
        self._sub = self.create_subscription(
            CompressedImage, in_topic, self._image_cb, _IMAGE_QOS)

        self._frames     = 0
        self._last_class = 'none'
        self._t_stats    = self.get_clock().now()
        self.create_timer(5.0, self._log_stats)

        self.get_logger().info(
            f'YoloDetectorNode ready  conf≥{self._conf_thr}  iou={self._iou_thr}  '
            f'blur_thr={self._blur_thr}  half={self._half}  '
            f'{in_topic} → {out_topic}')

    # ── Image callback ────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage):
        # Run whenever SOMEONE needs us: the sign topic (behavior layer) OR the debug
        # image. The sign topic drives the robot, so don't gate inference on debug alone.
        want_debug = self._pub_debug.get_subscription_count() > 0
        want_sign  = self._pub_sign.get_subscription_count() > 0
        if not (want_debug or want_sign):
            return

        frame = cv2.imdecode(
            np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('cv2.imdecode returned None — skipping frame')
            return

        # Blur pre-filter (disabled when blur_threshold=0.0). Publish 'none' on a skip so
        # a stale sign doesn't linger as if still detected.
        if self._blur_thr > 0.0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < self._blur_thr:
                self._publish_sign('none')
                return

        results = self._model(
            frame, verbose=False, device=self._device,
            conf=self._conf_thr, iou=self._iou_thr, half=self._half,
        )[0]

        best_box   = None
        best_area  = -1
        best_label = ''
        best_class = 'none'

        if results.boxes is not None:
            for i in range(len(results.boxes)):
                conf = float(results.boxes.conf[i])
                if conf < self._conf_thr:
                    continue
                x1, y1, x2, y2 = (int(v) for v in results.boxes.xyxy[i])
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area  = area
                    best_box   = (x1, y1, x2, y2)
                    cls_id     = int(results.boxes.cls[i])
                    cls_name   = results.names[cls_id]
                    best_label = f'{cls_name}  {conf:.2f}'
                    best_class = cls_name

        # Publish the closest detected class (or 'none') for the behavior layer.
        # Append the bbox area fraction (closest sign's pixels / frame pixels) so the
        # follower can fire a turn only when the sign is actually IN FRONT (large box),
        # not on a far-away first detection. Format: "<class>:<area_frac>" (e.g.
        # "letIzquierda:0.082"); 'none' is published bare for backward compatibility.
        self._last_class = best_class
        if best_class != 'none' and best_area > 0:
            fh, fw = frame.shape[0], frame.shape[1]
            area_frac = best_area / float(max(fh * fw, 1))
            self._publish_sign(f'{best_class}:{area_frac:.4f}')
        else:
            self._publish_sign(best_class)

        # Only build + publish the (CPU-costly) debug overlay if someone is watching it.
        if want_debug:
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
                # Filled label background so text is always legible.
                (tw, th), _ = cv2.getTextSize(
                    best_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), _BOX_COLOR, -1)
                cv2.putText(
                    frame, best_label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT_COLOR, 2)
            self._publish_frame(frame, msg.header.stamp)
        self._frames += 1

    # ── Publish helpers ───────────────────────────────────────────────────────

    def _publish_sign(self, cls_name: str):
        msg = String()
        msg.data = cls_name
        self._pub_sign.publish(msg)

    def _publish_frame(self, frame: np.ndarray, stamp):
        h, w, c = frame.shape
        out = Image()
        out.header.stamp    = stamp
        out.header.frame_id = 'camera_optical_frame'
        out.height          = h
        out.width           = w
        out.encoding        = 'bgr8'
        out.is_bigendian    = 0
        out.step            = w * c
        out.data            = array.array('B', frame.tobytes())
        self._pub_debug.publish(out)

    # ── FPS statistics ────────────────────────────────────────────────────────

    def _log_stats(self):
        now = self.get_clock().now()
        dt  = (now - self._t_stats).nanoseconds * 1e-9
        if dt > 0:
            self.get_logger().info(
                f'yolo: {self._frames / dt:.1f} fps  last_class={self._last_class}')
        self._frames  = 0
        self._t_stats = now


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectorNode()
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
