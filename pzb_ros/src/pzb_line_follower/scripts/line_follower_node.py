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
import math
import time
from collections import deque

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
        self.declare_parameter('publish_debug',        True)
        self.declare_parameter('topic_image_in',       '/camera/image_raw')
        self.declare_parameter('topic_cmd_vel',        '/cmd_vel_desired_raw')
        self.declare_parameter('sharp_turn_threshold_px', 80)
        self.declare_parameter('sharp_turn_speed',        0.03)
        self.declare_parameter('lost_timeout_s',          0.25)
        self.declare_parameter('lost_speed_scale',        0.50)
        self.declare_parameter('dashed_confirm_frames',   3)
        self.declare_parameter('dashed_coast_s',          0.3)
        self.declare_parameter('openloop_speed_mps',      0.15)
        self.declare_parameter('openloop_dist_m',         0.50)
        self.declare_parameter('recovery_angular_z',      0.30)
        self.declare_parameter('turn_approach_delay_s',   0.15)
        # Post-intersection acquisition + stuck-lock watchdog (bag8 fix)
        self.declare_parameter('acquire_guard_s',         0.4)
        self.declare_parameter('stuck_lock_s',            1.0)
        self.declare_parameter('stuck_lock_band_px',      16)
        self.declare_parameter('stuck_lock_var_px',       6)
        # Dashed-crossing behavior (bad_alginment fix)
        self.declare_parameter('dashed_recovery_enabled', False)
        self.declare_parameter('max_error_jump_px',       60)
        # Anti-stutter (10-attempt fix): the cx target jumps frame-to-frame on this
        # multi-line + jigsaw-seam track at ~7 fps, and the PD reversed steering each
        # frame (angular sign-flips 4-8/s). Two damping layers:
        #   angular_smooth_alpha — EMA on the OUTPUT angular.z (0=frozen, 1=no smoothing).
        #   error_median_n       — median-filter the steering error over N frames so a
        #                          single teleport frame can't swing the command.
        self.declare_parameter('angular_smooth_alpha',    0.4)
        self.declare_parameter('error_median_n',          3)
        # Search-then-stop loss recovery (10-attempt fix): on line loss, steer toward
        # the last-seen side at low forward speed for up to search_timeout_s to bring
        # the line back into view, then STOP. Never reverse, never whip out of bounds.
        self.declare_parameter('search_timeout_s',        1.2)
        self.declare_parameter('search_speed_mps',        0.04)
        self.declare_parameter('search_angular_z',        0.25)
        # Fork branch selection (pzb_not_workingcorrectly4). Default OFF: it steers the
        # bag4 fork LEFT correctly, but per-frame geometry cannot yet separate a true
        # fork from a sharp single-lane curve (regresses bad_ignment5/bad_alginment2),
        # so it ships gated until a temporal/exits-based discriminator is added.
        self.declare_parameter('fork_select_enabled',     False)
        # Perpendicular dashed-alignment (bad_alginment2 fix)
        self.declare_parameter('dashed_align_enabled',    True)
        self.declare_parameter('align_deadband_deg',      6.0)
        self.declare_parameter('k_align_z',               0.015)
        self.declare_parameter('align_max_z',             0.30)
        self.declare_parameter('align_sign',             1)
        self.declare_parameter('align_slope_median_n',    5)
        self.declare_parameter('align_timeout_s',         1.2)
        self.declare_parameter('align_window_s',          2.0)
        self.declare_parameter('align_max_tilt_deg',      35.0)

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
        self._pub_debug           = bool(self.get_parameter('publish_debug').value)
        self._sharp_turn_threshold = int(self.get_parameter('sharp_turn_threshold_px').value)
        self._sharp_turn_speed    = float(self.get_parameter('sharp_turn_speed').value)
        self._lost_timeout_s      = float(self.get_parameter('lost_timeout_s').value)
        self._lost_speed_scale    = float(self.get_parameter('lost_speed_scale').value)
        self._dashed_confirm_n    = int(self.get_parameter('dashed_confirm_frames').value)
        self._dashed_coast_s      = float(self.get_parameter('dashed_coast_s').value)
        self._openloop_speed      = float(self.get_parameter('openloop_speed_mps').value)
        self._openloop_dist_m     = float(self.get_parameter('openloop_dist_m').value)
        self._recovery_angular    = float(self.get_parameter('recovery_angular_z').value)
        self._turn_approach_delay = float(self.get_parameter('turn_approach_delay_s').value)
        self._acquire_guard_s     = float(self.get_parameter('acquire_guard_s').value)
        self._stuck_lock_s        = float(self.get_parameter('stuck_lock_s').value)
        self._stuck_lock_band     = float(self.get_parameter('stuck_lock_band_px').value)
        self._stuck_lock_var      = float(self.get_parameter('stuck_lock_var_px').value)
        self._dashed_recovery_en  = bool(self.get_parameter('dashed_recovery_enabled').value)
        self._max_error_jump      = float(self.get_parameter('max_error_jump_px').value)
        self._angular_smooth_a    = float(self.get_parameter('angular_smooth_alpha').value)
        self._error_median_n      = int(self.get_parameter('error_median_n').value)
        self._search_timeout_s    = float(self.get_parameter('search_timeout_s').value)
        self._search_speed        = float(self.get_parameter('search_speed_mps').value)
        self._search_angular      = float(self.get_parameter('search_angular_z').value)
        self._fork_select_enabled = bool(self.get_parameter('fork_select_enabled').value)
        self._align_enabled       = bool(self.get_parameter('dashed_align_enabled').value)
        self._align_deadband_deg  = float(self.get_parameter('align_deadband_deg').value)
        self._k_align_z           = float(self.get_parameter('k_align_z').value)
        self._align_max_z         = float(self.get_parameter('align_max_z').value)
        self._align_sign          = float(self.get_parameter('align_sign').value)
        self._align_median_n      = int(self.get_parameter('align_slope_median_n').value)
        self._align_timeout_s     = float(self.get_parameter('align_timeout_s').value)
        self._align_window_s      = float(self.get_parameter('align_window_s').value)
        self._align_max_tilt_deg  = float(self.get_parameter('align_max_tilt_deg').value)
        # Anisotropic ROI squash factor (vscale/hscale = 0.222/0.250) — converts the
        # ROI dash slope to a true world tilt angle. See project-roi-geometry memory.
        self._roi_aniso           = 0.889

        topic_in  = self.get_parameter('topic_image_in').value
        topic_cmd = self.get_parameter('topic_cmd_vel').value

        # Use all 4 Jetson Nano cores for OpenCV (Team2 technique).
        cv2.setNumThreads(4)

        # D-term state: use actual elapsed time in seconds (Team2 pattern) so the
        # derivative is FPS-invariant instead of "change per frame".
        self._prev_error   = 0.0
        self._prev_error_t = None   # monotonic timestamp of last image callback

        # Approach-speed memory (bad_alginment fix): rolling cruise speed captured
        # in solid PD mode, used so the dashed open-loop crossing continues at the
        # same rate instead of jumping to a fixed openloop_speed_mps.
        self._approach_speed = float(self.get_parameter('linear_speed').value)
        # Last accepted (median-filtered) steering error — also used to choose the
        # search direction (toward the last-seen line side) on line loss.
        self._last_good_error = 0.0

        # Anti-stutter state (10-attempt fix):
        #   _error_hist  — recent raw steering errors, median-filtered before the PD so a
        #                  single cx-teleport frame can't swing the command.
        #   _smoothed_az — EMA of the OUTPUT angular.z, so the command can't reverse
        #                  instantly on one bad frame (the cause of the 4-8/s sign-flips).
        self._error_hist  = deque(maxlen=max(1, self._error_median_n))
        self._smoothed_az = 0.0

        # Search-then-stop loss recovery state:
        #   _search_until — monotonic deadline of the active search window (None=idle).
        #   _search_dir   — +1/-1 steer direction (toward the last-seen line side).
        self._search_until = None
        self._search_dir   = 0.0

        self._speed_scale = 1.0

        # Decoupled cmd_vel: store the latest command; a 20 Hz timer publishes it.
        # This keeps the control loop running even if the camera briefly drops frames.
        self._latest_cmd    = Twist()
        self._last_valid_cmd = Twist()
        self._last_valid_t   = None   # monotonic timestamp of last frame where cx was valid

        # Dashed-state debounce + exit latch.
        # Entry: require _dashed_confirm_n consecutive dashed frames before switching mode.
        # Exit: once confirmed, hold "dashed" for _dashed_latch_s seconds to survive
        #       brief solid-stub glimpses mid-deceleration.
        self._dashed_latch_t  = None
        self._dashed_latch_s  = 1.0
        self._dashed_streak   = 0   # consecutive frames the detector returned "dashed"
        self._dashed_first_t  = None  # monotonic time when dashed was first confirmed
        self._recovery_side   = None  # 'left'|'right'|None — for open-loop boundary recovery

        # Perpendicular dashed-alignment (bad_alginment2 fix): rolling buffer of
        # recent VALID dash-slope samples (median-filtered to reject the noisy
        # per-frame slope) and a latch marking the align sub-phase complete.
        self._slope_buf      = deque(maxlen=self._align_median_n)
        self._aligned_latch  = False  # True once perpendicular → proceed to crossing
        self._align_done_t   = None   # monotonic time the align sub-phase completed

        # Turn approach delay: when the camera first sees a sharp turn, coast at reduced
        # speed for _turn_approach_delay seconds before applying full angular correction.
        # This compensates for the camera looking ahead of the robot body.
        self._turn_first_seen_t = None   # monotonic time when large error first appeared

        # Post-intersection acquisition guard + stuck-lock watchdog (bag8 fix).
        # _was_dashed tracks the latched line_type so we can detect the dashed→solid
        # edge and (a) re-seed the detector's anchors, (b) arm a steering cap.
        # _acq_until caps angular output until this monotonic time after the exit.
        # _err_hist holds recent (t, error) samples so the watchdog can detect a
        # frozen, non-converging off-center lock (robot chasing a wall/seam).
        self._was_dashed = False
        self._acq_until  = None
        self._err_hist   = deque()

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

        # Fixed-height ROI: always crop the SAME source rows (bottom half) so the
        # vertical scale (source_rows → target_h) is constant frame-to-frame.
        #
        # Previously this flipped between bottom-half (when <2 lines were visible,
        # for sharp-turn look-ahead) and bottom-third (normal), then forced BOTH
        # into target_h px. That made the vertical squash factor jump between
        # frames (360→80 = 0.22× vs 240→80 = 0.33×), so the same physical line
        # landed at a different ROI row and a different apparent slope whenever the
        # mode flipped — exactly when tracking was already shaky (≥2 lines lost).
        # Always using the bottom half keeps the sharp-turn look-ahead (the taller
        # view is a superset; the detector's far-field filter discards the top 15%)
        # while removing the geometric discontinuity. Horizontal scale (and thus
        # the cx→steering mapping) is unchanged: full width → target_w as before.
        roi_start = msg.height // 2
        roi_crop = full[roi_start:, :]

        target_h = self._img_h // 3
        target_w = self._img_w
        if roi_crop.shape[0] != target_h or roi_crop.shape[1] != target_w:
            img = cv2.resize(roi_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            img = np.ascontiguousarray(roi_crop)

        cx, cy = self._detector.detect_center_line(img, pre_cropped=True)
        line_type = self._detector.line_type

        # Dashed debounce (entry) + exit latch.
        # Entry: only confirm dashed after _dashed_confirm_n consecutive frames —
        #        prevents a single bad frame during a turn from triggering a stop.
        # Exit:  once confirmed, hold "dashed" for _dashed_latch_s so the robot
        #        stops fully before a brief solid-stub glimpse reverts control.
        _t_latch = time.monotonic()
        if line_type == 'dashed':
            # On the FIRST raw-dashed frame (approach), arm the acquisition cap so
            # the noisy approach frames can't produce a violent steering spike
            # before dashed is confirmed (bad_alginment: cx teleported 155→21 → az
            # +0.714/−0.800). Reuses the same _acq_until clamp as the exit guard.
            if self._dashed_streak == 0:
                self._acq_until = _t_latch + self._acquire_guard_s
            self._dashed_streak += 1
        else:
            self._dashed_streak = 0

        _confirmed_dashed = (self._dashed_streak >= self._dashed_confirm_n)
        if _confirmed_dashed:
            self._dashed_latch_t = _t_latch
            if self._dashed_first_t is None:
                # Entering a fresh dashed crossing — reset alignment state.
                self._dashed_first_t = _t_latch
                self._slope_buf.clear()
                self._aligned_latch = False
                self._align_done_t  = None

        if _confirmed_dashed or (self._dashed_latch_t is not None
                and (_t_latch - self._dashed_latch_t) < self._dashed_latch_s):
            line_type = 'dashed'
        else:
            line_type = 'solid'
            self._dashed_first_t = None  # reset so next encounter starts fresh

        # Feed the dash-slope median buffer ONLY from clean, plausible dash-row
        # readings (bad_ignment4 fix). The raw slope degrades to garbage at the END
        # of a crossing (robot overlaps the dashes / sees the straight lines ahead),
        # producing e.g. a +32° spike when the truth is +1–2°. Guards:
        #   - detector marked the slope valid (≥3 dashes, enough x-span), AND
        #   - the frame is genuinely classified dashed (not a solid-dominated frame), AND
        #   - the implied tilt is physically plausible (|tilt| ≤ align_max_tilt_deg).
        if (self._detector.dash_slope_valid and line_type == 'dashed'):
            _tilt = math.degrees(math.atan(self._detector.dash_slope_px / self._roi_aniso))
            if abs(_tilt) <= self._align_max_tilt_deg:
                self._slope_buf.append(self._detector.dash_slope_px)

        # True loss: all three line slots have no detection this frame.
        # The detector's line_flags dict is True per slot when a real contour was assigned.
        line_lost = not any(self._detector.line_flags.values())

        now = time.monotonic()

        # Dashed→solid edge (intersection exit): re-seed the tracker so the next
        # assignment is anchored at image center, and arm a steering cap so the
        # first large-error frame can't pivot ~90° onto a leftover crossing dash
        # (bag8 Section 1). _was_dashed tracks the previous resolved line_type.
        if self._was_dashed and line_type == 'solid':
            self._detector.reset_tracker_anchors()
            self._acq_until = now + self._acquire_guard_s
            self._err_hist.clear()   # drop the pre-exit error history
        self._was_dashed = (line_type == 'dashed')

        if line_type == 'dashed':
            # Intersection handling — align EARLY, then cross straight, then stop:
            #   Phase A — Align/coast (bad_ignment4 strong-safeguard): drive forward
            #             at the APPROACH speed while squaring up to the dashes, but
            #             ONLY within an early window (align_window_s) when the dash
            #             row is clean and ahead. The robot latches "aligned" the
            #             instant a clean median tilt is within the deadband — so an
            #             already-straight robot latches immediately and NEVER
            #             re-steers (the prior bug: a +32° end-of-crossing outlier
            #             rotated a straight robot). After the window, cross straight.
            #   Phase B — Straight crossing: angular.z = 0 for openloop_dur.
            #   Phase C — Stop: motors off; latch expires → normal PD resumes.
            elapsed = (now - self._dashed_first_t) if self._dashed_first_t is not None else 0.0

            # Cross at the approach speed (clamped), fallback openloop_speed_mps.
            cross_speed = self._approach_speed
            if not (cross_speed > 1e-3):
                cross_speed = self._openloop_speed
            cross_speed = min(self._max_lin_speed,
                              max(self._min_lin_speed, cross_speed))
            openloop_dur = self._openloop_dist_m / max(cross_speed, 0.01)

            align_z, tilt_deg, have_tilt = self._alignment_cmd()
            # Always-straight commitment (10-attempt fix): when alignment is disabled
            # the only required action at a dashed crossing is to cross STRAIGHT, so
            # latch "aligned" IMMEDIATELY on entry — go straight to Phase B (no align
            # steering, no align_window_s wait). This removes the alignment spin as a
            # variable for the simple track (attempts 5/6/7/9 went the wrong way at the
            # junction; signs/YOLO will choose the direction later).
            if not self._align_enabled and not self._aligned_latch:
                self._aligned_latch = True
                self._align_done_t  = now
            # Latch "aligned" the moment a clean median tilt is within the deadband
            # — no coast gate, so an already-perpendicular robot stops correcting
            # immediately and can never be re-rotated by a later bad reading.
            if (not self._aligned_latch and have_tilt
                    and abs(tilt_deg) <= self._align_deadband_deg):
                self._aligned_latch = True
                self._align_done_t  = now
            # Safety cap: give alignment up to align_window_s to converge; past it,
            # cross straight regardless (the dash row degrades once the robot
            # overlaps it, so prolonging only chases garbage).
            if (not self._aligned_latch and elapsed >= self._align_window_s):
                self._aligned_latch = True
                self._align_done_t  = now

            if not self._aligned_latch:
                # Phase A — align/coast: forward + perpendicular alignment steering.
                cmd = Twist()
                cmd.linear.x  = cross_speed * max(self._speed_scale, 1.0)
                cmd.angular.z = align_z
            elif (now - self._align_done_t) < openloop_dur:
                # Phase B — straight crossing at the approach speed.
                cmd = Twist()
                cmd.linear.x = cross_speed * max(self._speed_scale, 1.0)
                cmd.angular.z = 0.0
                if self._dashed_recovery_en:
                    # Optional legacy side-aware boundary recovery (default OFF).
                    lf = self._detector.line_flags
                    only_right = lf['right'] and not lf['left'] and not lf['center']
                    only_left  = lf['left']  and not lf['right'] and not lf['center']
                    if only_right:
                        cmd.angular.z = self._recovery_angular
                        self._recovery_side = 'right'
                    elif only_left:
                        cmd.angular.z = -self._recovery_angular
                        self._recovery_side = 'left'
                    elif line_lost and self._recovery_side == 'right':
                        cmd.angular.z = self._recovery_angular
                    elif line_lost and self._recovery_side == 'left':
                        cmd.angular.z = -self._recovery_angular
                    else:
                        self._recovery_side = None
            else:
                # Phase C — full stop
                cmd = Twist()
                self._last_valid_cmd = cmd

            self._latest_cmd = cmd
            # Keep the anti-stutter smoothers in sync so PD resumes cleanly after the
            # crossing (EMA continues from the issued angular; median buffer not stale).
            self._smoothed_az = cmd.angular.z
            self._error_hist.clear()
            cx_msg = Int32();  cx_msg.data = cx
            self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = float(cx - self._img_w // 2)
            self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type
            self._pub_line_type.publish(type_msg)
        elif line_lost:
            # Search-then-stop loss recovery (10-attempt fix). The old behavior held the
            # last command for lost_timeout_s (~2 frames at 7 fps) then full-stopped and
            # never re-acquired → attempts 4 & 8 went stale forever. Now: on loss, steer
            # toward the LAST-SEEN line side at a low forward speed for up to
            # search_timeout_s to bring the line back into view, then STOP. Never reverse
            # (linear stays ≥ 0), and the search angular is capped so it can't whip the
            # robot out of bounds (attempt 3).
            if self._search_until is None:
                # Entering search: pick the direction from the last accepted error sign.
                # error > 0 ⇒ line was to the RIGHT of center ⇒ steer right (negative az);
                # error < 0 ⇒ line was to the LEFT ⇒ steer left (positive az). Fall back
                # to the recovery side, else don't rotate (just creep straight).
                if self._last_good_error > self._dead_band:
                    self._search_dir = -1.0
                elif self._last_good_error < -self._dead_band:
                    self._search_dir = +1.0
                elif self._recovery_side == 'right':
                    self._search_dir = -1.0
                elif self._recovery_side == 'left':
                    self._search_dir = +1.0
                else:
                    self._search_dir = 0.0
                self._search_until = now + self._search_timeout_s

            if now < self._search_until:
                cmd = Twist()
                cmd.linear.x  = max(0.0, self._search_speed)   # forward only, never reverse
                cmd.angular.z = self._search_dir * self._search_angular
            else:
                cmd = Twist()   # search window expired — safe full stop, hold
            self._latest_cmd = cmd
            # Keep the EMA in sync so PD resumes from the issued angular if the line returns.
            self._smoothed_az = cmd.angular.z

            # Still publish detections so monitors can see the loss
            cx_msg = Int32();  cx_msg.data = cx
            self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = float(cx - self._img_w // 2)
            self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type
            self._pub_line_type.publish(type_msg)
        elif self._is_stuck_lock(now, float(cx - self._img_w // 2)):
            # Stuck-lock watchdog: the detector is reporting a real contour, but the
            # error has been frozen far off-center and is not converging — the robot is
            # "following" a stationary non-line (wall / floor-panel seam, bag8 Section 2).
            # Route it into the SAME search-then-stop recovery as a true loss: steer
            # toward the last-seen side briefly to re-find the real line, then stop.
            if self._search_until is None:
                if self._last_good_error > self._dead_band:
                    self._search_dir = -1.0
                elif self._last_good_error < -self._dead_band:
                    self._search_dir = +1.0
                else:
                    self._search_dir = 0.0
                self._search_until = now + self._search_timeout_s

            if now < self._search_until:
                cmd = Twist()
                cmd.linear.x  = max(0.0, self._search_speed)
                cmd.angular.z = self._search_dir * self._search_angular
            else:
                cmd = Twist()   # search window expired — safe full stop, hold
            self._latest_cmd = cmd
            self._smoothed_az = cmd.angular.z

            cx_msg = Int32();  cx_msg.data = cx
            self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = float(cx - self._img_w // 2)
            self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type
            self._pub_line_type.publish(type_msg)
        else:
            self._last_valid_t = now
            self._search_until = None   # line re-acquired — clear any active search window

            raw_error = float(cx - self._img_w // 2)

            # Error median filter (10-attempt anti-stutter fix): on this multi-line +
            # jigsaw-seam track at ~7 fps the cx target jumps between features frame to
            # frame (e.g. +1 → −57 → +1), and the PD reversed steering each time
            # (sign-flips 4-8/s, az slamming to ±0.8). Steering on the MEDIAN of the last
            # _error_median_n raw errors rejects a single teleport frame while a genuine
            # sustained turn (error persists ≥ ceil(n/2) frames) still gets through. This
            # supersedes the old one-shot >max_error_jump clamp, which only caught a single
            # frame and missed the sustained swinging. Disabled when error_median_n <= 1.
            self._error_hist.append(raw_error)
            if self._error_median_n > 1 and len(self._error_hist) > 0:
                error = float(np.median(self._error_hist))
            else:
                error = raw_error
            self._last_good_error = error

            # Time-based D-term (Team2 pattern): divide by actual dt in seconds so the
            # derivative gain is FPS-invariant.  At 30 fps, dt≈33 ms; at 5 fps, dt≈200 ms.
            if self._prev_error_t is not None and self._kd != 0.0:
                dt = now - self._prev_error_t
                d_error = (error - self._prev_error) / dt if dt > 0 else 0.0
            else:
                d_error = 0.0
            self._prev_error   = error
            self._prev_error_t = now

            # Turn approach delay: camera looks ahead of the robot body.
            # When |error| crosses the sharp-turn threshold, start a timer.
            # During the delay window, cap angular output so the body has time
            # to reach the turn before full steering is applied.
            _sharp = abs(error) > self._sharp_turn_threshold
            if _sharp:
                if self._turn_first_seen_t is None:
                    self._turn_first_seen_t = now
                _in_approach = (now - self._turn_first_seen_t) < self._turn_approach_delay
            else:
                self._turn_first_seen_t = None
                _in_approach = False

            # Acquisition guard (bag8 Section 1): for a short window after the
            # dashed→solid intersection exit, cap angular output the same way the
            # turn-approach delay does, so the tracker has time to settle on the
            # real center line before full steering is applied.
            if self._acq_until is not None and now < self._acq_until:
                _in_approach = True

            # PD steering
            if abs(error) <= self._dead_band:
                angular_z = 0.0
            else:
                angular_z = float(np.clip(
                    -self._kp * error - self._kd * d_error,
                    -self._max_ang, self._max_ang,
                ))
                if _in_approach:
                    approach_cap = self._max_ang * 0.3
                    angular_z = float(np.clip(angular_z, -approach_cap, approach_cap))

            # Output angular EMA (10-attempt anti-stutter fix): low-pass the COMMANDED
            # angular.z so a single bad cx frame can't reverse steering instantly — the
            # direct cause of the 4-8/s sign-flips and the ±0.8 slams. alpha=1 disables
            # smoothing; alpha≈0.4 keeps responsiveness while damping the jitter. The
            # dead-band still forces an exact 0 (no creeping bias when truly centered).
            if abs(error) <= self._dead_band:
                self._smoothed_az = 0.0
                angular_z = 0.0
            else:
                a = self._angular_smooth_a
                self._smoothed_az = (1.0 - a) * self._smoothed_az + a * angular_z
                angular_z = self._smoothed_az

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

            if self._speed_scale <= 0.0:
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

            # Track the steady cruise speed (bad_alginment fix) so the dashed
            # open-loop crossing can continue at the same rate. Only sample when
            # the robot is actually cruising (near-centered, not in a sharp-turn or
            # low-visibility slowdown) so pre-intersection dips don't drag it down.
            if abs(error) <= self._sharp_turn_threshold and n_vis >= 2 \
                    and linear_x > self._min_lin_speed:
                self._approach_speed = 0.7 * self._approach_speed + 0.3 * linear_x

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

        # Gate fork branch selection (pzb_not_workingcorrectly4): never pick a fork
        # branch while the robot is squaring up to a dashed crossing (Phase A of
        # ALIGN→CROSS→STOP) — alignment owns the steering there. Re-enable once the
        # crossing is aligned/over or we're back in solid following. Applies to the
        # NEXT frame's detect_center_line call.
        aligning = (line_type == 'dashed') and (not self._aligned_latch)
        self._detector.branch_select_enabled = self._fork_select_enabled and (not aligning)

    def _is_stuck_lock(self, now, error):
        """True when the tracked error is frozen far off-center (non-converging).

        Records (t, error) into a rolling ~stuck_lock_s window. Returns True only
        once the window is full of samples that are ALL off-center
        (|error| > stuck_lock_band_px) AND span a tiny range
        (max−min < stuck_lock_var_px) — i.e. the robot is chasing a stationary
        non-line it can never center (bag8 Section 2). During the post-intersection
        acquisition guard the watchdog is suppressed so a settling tracker is not
        mistaken for a stuck lock.
        """
        self._err_hist.append((now, error))
        while self._err_hist and (now - self._err_hist[0][0]) > self._stuck_lock_s:
            self._err_hist.popleft()

        if self._acq_until is not None and now < self._acq_until:
            return False
        # Need a full window of samples before judging.
        if len(self._err_hist) < 3 or (now - self._err_hist[0][0]) < self._stuck_lock_s:
            return False

        errs = [e for _, e in self._err_hist]
        if min(abs(e) for e in errs) <= self._stuck_lock_band:
            return False
        if (max(errs) - min(errs)) >= self._stuck_lock_var:
            return False
        return True

    def _alignment_cmd(self):
        """Perpendicular-alignment angular.z from the median dash slope.

        Returns (align_z, tilt_deg, have_estimate). Uses the median of the recent
        VALID dash-slope samples (rejects the noisy per-frame slope), converts to a
        true world tilt via the ROI anisotropy factor, applies a deadband, then a
        capped P term. align_z is 0 inside the deadband (perpendicular enough) or
        when there is no slope estimate yet. align_sign flips rotation direction.
        """
        if not self._align_enabled or len(self._slope_buf) == 0:
            return 0.0, 0.0, False
        med_slope = float(np.median(self._slope_buf))
        tilt_deg  = math.degrees(math.atan(med_slope / self._roi_aniso))
        if abs(tilt_deg) <= self._align_deadband_deg:
            return 0.0, tilt_deg, True
        align_z = self._align_sign * self._k_align_z * tilt_deg
        align_z = float(np.clip(align_z, -self._align_max_z, self._align_max_z))
        return align_z, tilt_deg, True

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
