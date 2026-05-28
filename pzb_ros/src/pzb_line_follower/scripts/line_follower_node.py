#!/usr/bin/env python3
"""
Line follower node for Puzzlebot.

Subscribes:
  /camera/image_raw         (sensor_msgs/Image)        raw BGR; node crops bottom 1/3
  /traffic_speed_scale      (std_msgs/Float32)   speed multiplier from traffic FSM

Publishes:
  /line_follower/cx           (std_msgs/Int32)    detected center x in pixels
  /line_follower/error        (std_msgs/Float32)  error from image center (px)
  /line_follower/line_type    (std_msgs/String)   "solid" | "dashed"
  /line_follower/debug_image  (sensor_msgs/Image) pipeline debug visualization
  /cmd_vel_desired_raw        (geometry_msgs/Twist)  — published at fixed 20 Hz timer
                                                        (Team2 pattern: decoupled from camera FPS)
"""

import array
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
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
        self.declare_parameter('Kd_angular',           0.0)
        self.declare_parameter('dead_band_px',         8)
        self.declare_parameter('linear_speed',         0.10)
        self.declare_parameter('max_linear_speed',     0.20)
        self.declare_parameter('max_angular',          0.8)
        self.declare_parameter('curve_speed_reduction', 0.5)
        self.declare_parameter('min_linear_speed',     0.05)
        self.declare_parameter('stop_on_dashed',       False)
        self.declare_parameter('publish_debug',        True)
        self.declare_parameter('topic_image_in',       '/camera/image_raw')
        self.declare_parameter('topic_cmd_vel',        '/cmd_vel_desired_raw')
        self.declare_parameter('sharp_turn_threshold_px', 80)
        self.declare_parameter('sharp_turn_speed',        0.03)
        self.declare_parameter('lost_timeout_s',          0.25)
        self.declare_parameter('lost_speed_scale',        0.50)
        self.declare_parameter('kp_dash_tilt',            0.15)

        self._img_w               = self.get_parameter('image_width').value
        self._img_h               = self.get_parameter('image_height').value
        self._kp                  = float(self.get_parameter('Kp_angular').value)
        self._kd                  = float(self.get_parameter('Kd_angular').value)
        self._dead_band           = int(self.get_parameter('dead_band_px').value)
        self._linear_speed        = float(self.get_parameter('linear_speed').value)
        self._max_lin_speed       = float(self.get_parameter('max_linear_speed').value)
        self._max_ang             = float(self.get_parameter('max_angular').value)
        self._curve_reduction     = float(self.get_parameter('curve_speed_reduction').value)
        self._min_lin_speed       = float(self.get_parameter('min_linear_speed').value)
        self._stop_dashed         = bool(self.get_parameter('stop_on_dashed').value)
        self._pub_debug           = bool(self.get_parameter('publish_debug').value)
        self._sharp_turn_threshold = int(self.get_parameter('sharp_turn_threshold_px').value)
        self._sharp_turn_speed    = float(self.get_parameter('sharp_turn_speed').value)
        self._lost_timeout_s      = float(self.get_parameter('lost_timeout_s').value)
        self._lost_speed_scale    = float(self.get_parameter('lost_speed_scale').value)
        self._kp_dash_tilt        = float(self.get_parameter('kp_dash_tilt').value)

        topic_in  = self.get_parameter('topic_image_in').value
        topic_cmd = self.get_parameter('topic_cmd_vel').value

        # Use all 4 Jetson Nano cores for OpenCV (Team2 technique).
        cv2.setNumThreads(4)

        # D-term state: use actual elapsed time in seconds (Team2 pattern) so the
        # derivative is FPS-invariant instead of "change per frame".
        self._prev_error   = 0.0
        self._prev_error_t = None   # monotonic timestamp of last image callback

        self._speed_scale = 1.0

        # Decoupled cmd_vel: store the latest command; a 20 Hz timer publishes it.
        # This keeps the control loop running even if the camera briefly drops frames.
        self._latest_cmd    = Twist()
        self._last_valid_cmd = Twist()
        self._last_valid_t   = None   # monotonic timestamp of last frame where cx was valid

        # Dashed-state latch: once "dashed" is detected, hold that state for this many
        # seconds so the robot has time to stop before detection briefly reverts to "solid"
        # (e.g. while decelerating past a 1-frame solid-stub glimpse at the intersection edge).
        self._dashed_latch_t = None
        self._dashed_latch_s = 1.0

        self._detector = CenterLineDetector(debug=self._pub_debug)

        # Publishers
        self._pub_cx        = self.create_publisher(Int32,   '/line_follower/cx',        _RELIABLE_QOS)
        self._pub_error     = self.create_publisher(Float32, '/line_follower/error',      _RELIABLE_QOS)
        self._pub_line_type = self.create_publisher(String,  '/line_follower/line_type',  _RELIABLE_QOS)
        self._pub_cmd       = self.create_publisher(Twist,   topic_cmd,                   _RELIABLE_QOS)
        self._pub_debug_img = self.create_publisher(Image,   '/line_follower/debug_image', _RELIABLE_QOS)

        # Subscriber — raw Image; BEST_EFFORT matches camera publisher
        self.create_subscription(Image, topic_in, self._image_cb, _BEST_EFFORT_QOS)

        # Traffic speed scale (optional — defaults to 1.0 if never published)
        self.create_subscription(Float32, '/traffic_speed_scale', self._cb_speed_scale, _RELIABLE_QOS)

        # 20 Hz cmd_vel publish timer — decoupled from camera FPS (Team2 pattern).
        self.create_timer(1.0 / 20.0, self._cmd_publish_cb)

        self.get_logger().info(
            f'Line follower ready  img={self._img_w}x{self._img_h}'
            f'  Kp={self._kp}  Kd={self._kd}  dead_band={self._dead_band}px'
            f'  v={self._linear_speed}/{self._max_lin_speed} m/s'
            f'  curve_reduction={self._curve_reduction}  min_v={self._min_lin_speed}'
            f'  debug={self._pub_debug}'
        )

    def _image_cb(self, msg: Image):
        # Decode raw BGR image — zero-copy view into msg.data
        full = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

        # Adaptive ROI: use bottom half when fewer than 2 lines were visible last frame
        # (lines may have migrated upward during a sharp turn).  Revert to bottom third
        # once all lines are reliably tracked again.
        n_prev_visible = sum(self._detector.line_flags.values())
        roi_start = msg.height // 2 if n_prev_visible < 2 else msg.height * 2 // 3
        roi_crop = full[roi_start:, :]

        target_h = self._img_h // 3
        target_w = self._img_w
        if roi_crop.shape[0] != target_h or roi_crop.shape[1] != target_w:
            img = cv2.resize(roi_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            img = np.ascontiguousarray(roi_crop)

        cx, cy = self._detector.detect_center_line(img, pre_cropped=True)
        line_type = self._detector.line_type

        # Dashed-state latch: hold "dashed" for _dashed_latch_s after the last genuine
        # detection so the robot stops fully before a 1-frame solid-stub glimpse reverts
        # control to line-following mode mid-deceleration.
        _t_latch = time.monotonic()
        if line_type == 'dashed':
            self._dashed_latch_t = _t_latch
        elif (self._dashed_latch_t is not None
              and (_t_latch - self._dashed_latch_t) < self._dashed_latch_s):
            line_type = 'dashed'

        # True loss: all three line slots have no detection this frame.
        # The detector's line_flags dict is True per slot when a real contour was assigned.
        line_lost = not any(self._detector.line_flags.values())

        now = time.monotonic()

        if line_lost:
            # Coast: hold last valid command at reduced speed for up to lost_timeout_s.
            if (self._last_valid_t is not None
                    and (now - self._last_valid_t) < self._lost_timeout_s):
                cmd = Twist()
                cmd.linear.x  = self._last_valid_cmd.linear.x * self._lost_speed_scale
                cmd.angular.z = self._last_valid_cmd.angular.z
            else:
                cmd = Twist()   # timeout — full stop
            self._latest_cmd = cmd

            # Still publish detections so monitors can see the loss
            cx_msg = Int32();  cx_msg.data = cx
            self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = float(cx - self._img_w // 2)
            self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type
            self._pub_line_type.publish(type_msg)
        else:
            self._last_valid_t = now

            error = float(cx - self._img_w // 2)

            # Time-based D-term (Team2 pattern): divide by actual dt in seconds so the
            # derivative gain is FPS-invariant.  At 30 fps, dt≈33 ms; at 5 fps, dt≈200 ms.
            if self._prev_error_t is not None and self._kd != 0.0:
                dt = now - self._prev_error_t
                d_error = (error - self._prev_error) / dt if dt > 0 else 0.0
            else:
                d_error = 0.0
            self._prev_error   = error
            self._prev_error_t = now

            # PD steering — dead-band only suppresses; dashed uses PD + tilt correction
            if abs(error) <= self._dead_band:
                angular_z = 0.0
            else:
                angular_z = float(np.clip(
                    -self._kp * error - self._kd * d_error,
                    -self._max_ang, self._max_ang,
                ))
                if line_type == 'dashed':
                    # Also correct for rotational misalignment: dash_slope_px=0 means
                    # dashes are horizontal (robot is perpendicular to the crossing line).
                    tilt = getattr(self._detector, 'dash_slope_px', 0.0)
                    angular_z = float(np.clip(
                        angular_z - self._kp_dash_tilt * tilt,
                        -self._max_ang, self._max_ang,
                    ))

            # Curve-coupled speed reduction
            if self._max_ang > 0:
                angular_fraction = abs(angular_z) / self._max_ang
            else:
                angular_fraction = 0.0
            speed_scale_curve = 1.0 - self._curve_reduction * angular_fraction
            min_frac  = self._min_lin_speed / self._linear_speed if self._linear_speed > 0 else 0.0
            linear_x  = min(self._max_lin_speed,
                            self._linear_speed * max(min_frac, speed_scale_curve))

            # Visibility-based speed reduction: fewer visible lines → slower
            n_vis = sum(self._detector.line_flags.values())
            if n_vis == 2:
                linear_x = min(linear_x, self._linear_speed * 0.6)
            elif n_vis <= 1:
                linear_x = self._sharp_turn_speed

            # Sharp-turn override (error magnitude) — can only further reduce speed
            if abs(error) > self._sharp_turn_threshold:
                linear_x = self._sharp_turn_speed

            # Dashed-line stop takes priority, then traffic scale
            if self._stop_dashed and line_type == 'dashed':
                linear_x = 0.0
            elif self._speed_scale <= 0.0:
                linear_x = 0.0
            else:
                linear_x *= self._speed_scale

            # Publish detections
            cx_msg = Int32();  cx_msg.data = cx
            self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = error
            self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type
            self._pub_line_type.publish(type_msg)

            # Store latest command — the 20 Hz timer actually publishes it.
            cmd = Twist()
            cmd.linear.x  = linear_x
            cmd.angular.z = angular_z
            self._last_valid_cmd = cmd
            self._latest_cmd     = cmd

        # Debug image
        if self._pub_debug and self._detector.debug_frame is not None:
            dbg = self._detector.debug_frame
            dh, dw = dbg.shape[:2]
            dbg_msg = Image()
            dbg_msg.header.stamp    = msg.header.stamp
            dbg_msg.header.frame_id = msg.header.frame_id
            dbg_msg.height   = dh
            dbg_msg.width    = dw
            dbg_msg.encoding = 'bgr8'
            dbg_msg.is_bigendian = 0
            dbg_msg.step = dw * 3
            dbg_msg.data = array.array('B', dbg.tobytes())
            self._pub_debug_img.publish(dbg_msg)

    def _cmd_publish_cb(self):
        """20 Hz timer — publishes the last computed command.

        Decoupled from image callbacks (Team2 pattern): the wheels keep receiving
        a steady command even when the camera briefly drops frames.
        """
        self._pub_cmd.publish(self._latest_cmd)

    def _cb_speed_scale(self, msg: Float32):
        self._speed_scale = float(msg.data)


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
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
