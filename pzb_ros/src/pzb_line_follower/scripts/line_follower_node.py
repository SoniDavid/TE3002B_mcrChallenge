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
        # Non-terminal rotate-until-found recovery (opt-bags stale fix). The old
        # search-then-STOP latched a permanent (0,0) once the search window expired and
        # never re-acquired (opt6/7/8 sat stale forever). Now, after the Phase-1 forward
        # creep, the robot does an IN-PLACE rotation sweep (linear=0, never drive into a
        # wall) toward the last-seen side, reversing every search_sweep_s so it scans both
        # sides instead of spinning past the line — and keeps doing it until a line is
        # re-acquired. search_rotate_z = capped in-place yaw during the sweep.
        self.declare_parameter('search_rotate_z',         0.30)
        self.declare_parameter('search_sweep_s',          2.0)
        # Blind-stall full stop: frame_stale_s holds steering through a brief drop, but a
        # multi-second camera FREEZE (both streams hang 3-9 s, opt6-9) would let the
        # recovery rotation spin the robot blind. Past frame_blind_s with no fresh frame,
        # publish a full (0,0) stop; the sweep resumes the instant frames return.
        self.declare_parameter('frame_blind_s',           1.0)
        # Curve handling (new-bags curve-understeer fix): on a sharp curve the detector
        # sees only ONE line (the other boundary leaves frame) — that is NORMAL, not a
        # near-loss. The old code floored speed to sharp_turn_speed there, so the robot
        # crawled and fell behind the curve. curve_min_speed keeps it advancing so the
        # line stays in frame while it steers around.
        self.declare_parameter('curve_min_speed',         0.06)
        # Steering slew-rate limit (replaces the pure output EMA): bounds the change in
        # commanded angular.z per detector frame. Caps single-frame reversals (anti-
        # stutter) WITHOUT lagging a sustained same-sign ramp on a real curve (which the
        # EMA did). rad/s of az change allowed per frame. 0 disables (no limit).
        self.declare_parameter('angular_slew_max',        0.12)
        # Sharp-turn slew bypass (newnew-bags fix): the gentle angular_slew_max needed
        # ~5 frames at ~7 fps to ramp steering, so the robot drove PAST tight curves
        # before steering built up. On a genuine sharp turn (|error| ≥ slew_bypass_error_px
        # AND consistent sign — NOT the small-error jitter the slew damps) use the larger
        # angular_slew_max_sharp so steering reaches the PD value in 1-2 frames.
        self.declare_parameter('slew_bypass_error_px',    80)
        self.declare_parameter('angular_slew_max_sharp',  0.45)
        # Curve dashed-suppression (newnew-bags fix): suppress a "dashed" classification
        # while the steering error is past this — a real crossing is approached head-on
        # (small error); a sharp curve (large error) breaks the line into co-linear blobs
        # that falsely read as a dash row and zero steering mid-curve. Measured: real
        # crossings sit at |err| 10-33 px, false-on-curve at 48-129 px → 45 separates them.
        self.declare_parameter('dashed_suppress_error_px', 45)
        # Tight-turn speed drop (adaptive-bags fix): above this |error| the curve is too
        # tight to take at curve speed (the line hit the ROI edge before the robot rounded
        # it), so drop to sharp_turn_speed for turn-radius headroom. Above the normal
        # sharp_turn_threshold (80) so ordinary bends keep curve_min_speed (no crawl).
        self.declare_parameter('tight_turn_error_px',     110)
        # Error-driven early braking (opt-bags turn-overshoot fix): the prior curve
        # slowdown coupled speed to the SLEW-LIMITED angular.z, which lags the turn —
        # at turn entry |error| is already large but angular.z is still ramping, so the
        # robot stayed near full speed exactly when it should brake and the line swept
        # off the squashed ROI before it could round the bend (opt5 t≈15: error 4→159,
        # cx pinned at 319 in ~0.1 s). This brakes off the IMMEDIATE |error| instead, so
        # the robot slows the instant a turn is SEEN. curve_brake_error_px = the |error|
        # at which the full curve_speed_reduction applies.
        self.declare_parameter('curve_brake_error_px',    90)
        # Stale-frame safety (adaptive_new fix): the camera stalls 0.4-1.4 s; if no new
        # frame for frame_stale_s the 20 Hz publisher scales linear speed by
        # stale_speed_scale so the robot doesn't drive blind off a curve. 0 disables.
        self.declare_parameter('frame_stale_s',           0.4)
        self.declare_parameter('stale_speed_scale',       0.0)
        # Direction hysteresis: once steering commits to a side, the error must exceed
        # this opposite-side threshold (px) before the command may reverse — kills the
        # ±jitter around center from the few-px cx wobble that the dead-band alone misses.
        self.declare_parameter('steer_hysteresis_px',     6)
        # Error-saturation junction guard: if |error| stays at/above this for
        # error_sat_frames consecutive frames the detector has lost the real line at a
        # junction (cx pinned at an edge) — route to search-then-stop instead of crawling
        # into the wall. Conservative so a genuine curve (converges < ~1 s) never trips it.
        self.declare_parameter('error_sat_px',            150)
        self.declare_parameter('error_sat_frames',        6)
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
        self._search_rotate_z     = float(self.get_parameter('search_rotate_z').value)
        self._search_sweep_s      = float(self.get_parameter('search_sweep_s').value)
        self._frame_blind_s       = float(self.get_parameter('frame_blind_s').value)
        self._curve_min_speed     = float(self.get_parameter('curve_min_speed').value)
        self._angular_slew_max    = float(self.get_parameter('angular_slew_max').value)
        self._slew_bypass_error_px = float(self.get_parameter('slew_bypass_error_px').value)
        self._angular_slew_max_sharp = float(self.get_parameter('angular_slew_max_sharp').value)
        self._dashed_suppress_error_px = float(self.get_parameter('dashed_suppress_error_px').value)
        self._tight_turn_error_px = float(self.get_parameter('tight_turn_error_px').value)
        self._curve_brake_error_px = float(self.get_parameter('curve_brake_error_px').value)
        self._frame_stale_s       = float(self.get_parameter('frame_stale_s').value)
        self._stale_speed_scale   = float(self.get_parameter('stale_speed_scale').value)
        self._steer_hyst_px       = float(self.get_parameter('steer_hysteresis_px').value)
        self._error_sat_px        = float(self.get_parameter('error_sat_px').value)
        self._error_sat_frames    = int(self.get_parameter('error_sat_frames').value)
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

        # Anti-stutter state:
        #   _error_hist  — recent raw steering errors, median-filtered before the PD so a
        #                  single cx-teleport frame can't swing the command.
        #   _prev_az     — last commanded angular.z; the slew-rate limit bounds the change
        #                  from it per frame (caps reversals without lagging a real ramp).
        #   _steer_side  — sign of the current committed steering direction, for hysteresis.
        self._error_hist  = deque(maxlen=max(1, self._error_median_n))
        self._prev_az     = 0.0
        self._steer_side  = 0.0

        # Search-then-stop loss recovery state:
        #   _search_until — monotonic deadline of the active search window (None=idle).
        #   _search_dir   — +1/-1 steer direction (toward the last-seen line side).
        #   _error_sat_count — consecutive frames |error| has been saturated (junction loss).
        self._search_until = None
        self._search_dir   = 0.0
        #   _search_sweep_dir/_until — Phase-2 in-place rotation sweep state (rotate one
        #   way until _search_sweep_until, then reverse) so the robot scans both sides and
        #   never latches a permanent stop.
        self._search_sweep_dir   = 1.0
        self._search_sweep_until = None
        self._error_sat_count = 0

        self._speed_scale = 1.0

        # Decoupled cmd_vel: store the latest command; a 20 Hz timer publishes it.
        # This keeps the control loop running even if the camera briefly drops frames.
        self._latest_cmd    = Twist()
        self._last_valid_cmd = Twist()
        self._last_valid_t   = None   # monotonic timestamp of last frame where cx was valid
        self._last_frame_t   = None   # monotonic timestamp of the last processed camera frame

        # Dashed-state debounce + exit latch.
        # Entry: require _dashed_confirm_n consecutive dashed frames before switching mode.
        # Exit: once confirmed, hold "dashed" for _dashed_latch_s seconds to survive
        #       brief solid-stub glimpses mid-deceleration.
        self._dashed_latch_t  = None
        self._dashed_latch_s  = 1.0
        self._dashed_streak   = 0   # consecutive frames the detector returned "dashed"
        self._dashed_first_t  = None  # monotonic time when dashed was first confirmed
        self._recovery_side   = None  # 'left'|'right'|None — for open-loop boundary recovery
        # Live-error abort of a false dashed-on-curve (opt-bags drive-off fix):
        # consecutive same-sign frames whose LIVE cx error exceeds the suppress threshold.
        self._dashed_live_break_count = 0
        self._dashed_live_break_sign  = 0

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
        # Stamp every processed camera frame (stale-frame safety): the camera runs at a
        # poor, variable 4.7-9.5 fps with stalls up to 1.4 s, during which the robot would
        # otherwise drive blind off a curve. The 20 Hz publish timer scales speed down if
        # this stamp goes stale (see _cmd_publish_cb).
        self._last_frame_t = time.monotonic()

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

        # Curve dashed-suppression (newnew-bags fix): a real dashed crossing is
        # approached head-on (the robot is going straight, |error| small); a SHARP CURVE
        # produces a large sustained steering error. On a curve the single line breaks
        # into several short, roughly co-linear blobs that mimic a dash row and the
        # classifier falsely returns "dashed" — which then zeroes steering (straight
        # crossing) mid-curve and the robot drives off the bend. Measured separation is
        # clean: real crossings here sit at |error| 10-33 px, false-on-curve ones at
        # 48-129 px. So if the (median-filtered) steering error magnitude is past
        # dashed_suppress_error_px, force solid and keep following the curve. Uses the
        # last accepted error from the previous frame (this frame's PD error is computed
        # later); a curve persists across frames so one-frame lag is harmless.
        if (line_type == 'dashed'
                and abs(self._last_good_error) > self._dashed_suppress_error_px):
            line_type = 'solid'
            self._detector.line_type = 'solid'

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

        # Live-error abort of a false dashed-on-curve (opt-bags drive-off fix). The
        # suppress guard above checks _last_good_error, which is FROZEN during a latched
        # crossing (only updated in the solid branch) — so a curve whose line broke into
        # co-linear blobs while still near center latches dashed at small error, then
        # crosses STRAIGHT (angular.z=0) while the TRUE error balloons off-track (opt4
        # t≈104.2-105.8: entered dashed at err≈11, then cx 250→287 / err→127 driving off
        # the bend). Re-check the LIVE error here: if it exceeds dashed_suppress_error_px
        # for ≥2 consecutive same-sign frames the "dashes" are a curve, not a crossing —
        # break the latch and resume solid PD this frame. A real head-on crossing keeps a
        # small live error and is unaffected.
        if line_type == 'dashed':
            _live_err = float(cx - self._img_w // 2)
            _same = (self._dashed_live_break_sign == 0
                     or (_live_err > 0) == (self._dashed_live_break_sign > 0))
            if abs(_live_err) > self._dashed_suppress_error_px and _same:
                self._dashed_live_break_count += 1
                self._dashed_live_break_sign = 1 if _live_err > 0 else -1
            else:
                self._dashed_live_break_count = 0
                self._dashed_live_break_sign  = 0
            if self._dashed_live_break_count >= 2:
                line_type = 'solid'
                self._detector.line_type = 'solid'
                # Break the dashed latch entirely so PD steering resumes.
                self._dashed_latch_t = None
                self._dashed_streak  = 0
                self._dashed_first_t = None
                self._aligned_latch  = False
                self._dashed_live_break_count = 0
                self._dashed_live_break_sign  = 0
                # We are following a continuous (curving) line, NOT exiting an
                # intersection — suppress the dashed→solid anchor reset below so the
                # tracker isn't re-centered off the curve's offset line.
                self._was_dashed = False
        else:
            self._dashed_live_break_count = 0
            self._dashed_live_break_sign  = 0

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

        # Error-saturation junction guard (new-bags fix): at a hard junction the detector
        # can pin cx at an image edge (cx≈0 or |error|≈half-width) for several frames —
        # the robot then crawls into the wall chasing an error it can never null
        # (complex_new1 t=17.4: error stuck at −160 for ~2 s). Count consecutive saturated
        # frames; once it exceeds error_sat_frames, treat it as a near-loss (handled like
        # line_lost → search-then-stop). Only counts in solid mode (dashed owns its own
        # path) and resets the moment the error comes back in range, so a genuine sharp
        # curve (which converges within ~1 s, well under the threshold) never trips it.
        _sat_now = (line_type == 'solid'
                    and abs(float(cx - self._img_w // 2)) >= self._error_sat_px)
        if _sat_now:
            self._error_sat_count += 1
        else:
            self._error_sat_count = 0
        error_saturated = self._error_sat_count >= self._error_sat_frames

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
            # Keep the anti-stutter state in sync so PD resumes cleanly after the crossing
            # (slew limiter continues from the issued angular; median buffer not stale).
            self._prev_az = cmd.angular.z
            self._error_hist.clear()
            cx_msg = Int32();  cx_msg.data = cx
            self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = float(cx - self._img_w // 2)
            self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type
            self._pub_line_type.publish(type_msg)
        elif line_lost or error_saturated:
            # Non-terminal rotate-until-found recovery (opt-bags stale fix), also entered
            # when |error| has been SATURATED for several frames at a junction
            # (error_saturated). Phase 1 creeps forward toward the last-seen side, Phase 2
            # rotates in place sweeping both sides until the line returns — it never
            # latches a permanent (0,0) stop (the old search-then-STOP sat opt6/7/8 stale
            # forever). See _search_recovery_cmd.
            self._search_recovery_cmd(now)

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
            # Route it into the SAME non-terminal rotate-until-found recovery as a true
            # loss: creep toward the last-seen side, then sweep in place until re-acquired.
            self._search_recovery_cmd(now)

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
            # Steer on the MEDIAN of the last _error_median_n raw errors — on this
            # multi-line/jigsaw track at ~7 fps the detector cx (hence error) is NOISY,
            # bouncing frame to frame (measured at a sharp curve: 84,28,99,113,28,100,92),
            # and the raw PD reversed steering each time (the original 4-8/s stutter). The
            # median rejects the single-frame dips/teleports while a sustained turn (error
            # large for ≥ ceil(n/2) frames) still passes through.
            if self._error_median_n > 1 and len(self._error_hist) > 0:
                error = float(np.median(self._error_hist))
            else:
                error = raw_error
            self._last_good_error = error

            # Sharp-turn detection (newnew-bags fix). A real sharp turn is a LARGE,
            # consistent-sign error — distinct from the small-error sign-flipping jitter
            # the slew limit damps. Detect it from the MEDIAN magnitude (noise-robust, so
            # the per-frame dips above don't drop us out of sharp mode) plus a same-sign
            # check. In sharp mode the steering slew limit below is raised so the command
            # reaches the PD value in 1-2 frames instead of ~5 — the robot was driving
            # PAST tight curves because the gentle slew couldn't build steering in time.
            # We keep steering on the MEDIAN (not raw) even in sharp mode, so the curve's
            # detector noise is still rejected while the ramp is fast.
            _same_sign = all((e > 0) == (error > 0)
                             for e in self._error_hist if abs(e) > self._dead_band)
            sharp_turn = (abs(error) >= self._slew_bypass_error_px and _same_sign)

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
                # The approach/acquisition cap throttles angular to 0.3*max_ang while the
                # body catches up to a just-seen turn or settles after an intersection.
                # But it must NOT fight a CONFIRMED sharp turn (large, consistent-sign
                # error) — that is exactly when full steering is needed NOW or the line
                # leaves the ROI (adaptive bags: this cap alternated node_az 0.24/0.48 on a
                # 159 px curve, halving the steering every other frame). Skip the cap when
                # sharp_turn is active.
                if _in_approach and not sharp_turn:
                    approach_cap = self._max_ang * 0.3
                    angular_z = float(np.clip(angular_z, -approach_cap, approach_cap))

            # Direction hysteresis + steering slew-rate limit (curve-understeer fix,
            # replaces the prior output EMA). The EMA killed single-frame jitter but it
            # ALSO lagged a legitimate sustained curve ramp (the robot under-steered and
            # fell behind). A slew-rate limit caps the per-frame CHANGE in angular.z —
            # so a one-frame reversal is bounded (anti-stutter) while a steady same-sign
            # ramp passes through at full rate (good curve tracking). Hysteresis stops the
            # command flipping sign on the few-px cx wobble around center.
            if abs(error) <= self._dead_band:
                angular_z = 0.0
                self._steer_side = 0.0
            else:
                # Hysteresis: to reverse to the opposite side, the error must first exceed
                # steer_hysteresis_px on that side; otherwise hold the committed side at 0+.
                want_side = -1.0 if angular_z < 0 else 1.0   # az sign = steer direction
                if (self._steer_side != 0.0 and want_side != self._steer_side
                        and abs(error) <= self._steer_hyst_px):
                    angular_z = 0.0
                else:
                    self._steer_side = want_side
            # Node-side steering slew-rate limit. On a genuine sharp turn (large,
            # consistent-sign error) the node slew is BYPASSED entirely so the full PD
            # value is emitted at once — the downstream twist_slew_limiter (now at
            # max_angular_accel=4.0 rad/s²) does the smoothing fast enough (~1 frame), so a
            # second node-side limit only re-adds the lag that made the robot drive PAST
            # tight curves (adaptive bags: full steering reached the motors ~0.4 s late).
            # Bypassing here can't reintroduce stutter: stutter is small-error sign-FLIPPING,
            # which the same-sign sharp_turn gate excludes. Small errors keep the gentle
            # angular_slew_max for anti-stutter (the downstream node smooths those too).
            if sharp_turn:
                pass                                   # emit full PD value; downstream node ramps it
            elif self._angular_slew_max > 0:
                delta = float(np.clip(angular_z - self._prev_az,
                                      -self._angular_slew_max, self._angular_slew_max))
                angular_z = self._prev_az + delta
            self._prev_az = angular_z

            # Curve-coupled speed reduction
            if self._max_ang > 0:
                angular_fraction = abs(angular_z) / self._max_ang
            else:
                angular_fraction = 0.0
            speed_scale_curve = 1.0 - self._curve_reduction * angular_fraction
            # Error-driven early braking (opt-bags turn-overshoot fix): brake off the
            # IMMEDIATE |error| (the look-ahead signal) instead of waiting for the
            # slew-limited angular.z to ramp. error_frac ramps 0→1 as |error| grows from
            # the dead band to curve_brake_error_px, so the robot is already slowing the
            # instant a turn appears in the ROI — before the line can sweep off the edge.
            # Take whichever of the two scales brakes more so neither under-slows a turn.
            _brake_span = max(1.0, self._curve_brake_error_px - self._dead_band)
            error_frac  = float(np.clip((abs(error) - self._dead_band) / _brake_span, 0.0, 1.0))
            speed_scale_error = 1.0 - self._curve_reduction * error_frac
            speed_scale = min(speed_scale_curve, speed_scale_error)
            min_frac  = self._min_lin_speed / self._linear_speed if self._linear_speed > 0 else 0.0
            linear_x  = min(self._max_lin_speed,
                            self._linear_speed * max(min_frac, speed_scale))

            # Visibility-based speed reduction: fewer visible lines → slower.
            # CURVE-UNDERSTEER FIX: seeing only ONE line is NORMAL on a sharp curve (the
            # other boundary leaves frame); the old code floored to sharp_turn_speed here
            # so the robot crawled and fell behind. Use curve_min_speed instead — fast
            # enough to keep advancing around the bend so the line stays in frame.
            n_vis = sum(self._detector.line_flags.values())
            if n_vis == 2:
                linear_x = min(linear_x, self._linear_speed * 0.6)
            elif n_vis <= 1:
                linear_x = max(self._sharp_turn_speed, self._curve_min_speed)

            # Sharp-turn override (error magnitude): a large error means a real bend — slow
            # but DON'T crawl, or the robot can't advance through it (curve-understeer fix).
            # Floor at curve_min_speed for a moderate bend.
            if abs(error) > self._sharp_turn_threshold:
                linear_x = max(self._sharp_turn_speed, self._curve_min_speed)
            # TIGHT-turn override (adaptive-bags fix): on the very tightest curves
            # (|error| past tight_turn_error_px) even fast full steering couldn't hold the
            # line at curve speed — the turn radius was too large and the line hit the ROI
            # edge. Drop to the genuine slowest sharp_turn_speed there to shrink the radius
            # (turn-radius headroom). Only the tightest regime, so normal curves keep
            # curve_min_speed and don't crawl. Paired with the now-fast steering ramp so
            # the robot turns hard AND slow exactly when a curve is too tight to take fast.
            if abs(error) > self._tight_turn_error_px:
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

    def _search_recovery_cmd(self, now):
        """Non-terminal loss recovery (opt-bags stale fix): rotate until found.

        Phase 1 (search_timeout_s): creep FORWARD toward the last-seen side at
        search_speed to bring the line back with minimal motion. Phase 2 (until a line
        is re-acquired): IN-PLACE rotation sweep — linear=0 (never drive blind into a
        wall), rotating search_rotate_z toward the last-seen side and reversing every
        search_sweep_s so it scans both sides instead of spinning past the line. Never
        latches a permanent stop. The normal branch clears _search_until on re-acquire,
        which re-seeds this FSM next loss.
        """
        if self._search_until is None:
            # Entering search: initial direction from the last accepted error sign.
            # error > 0 ⇒ line was RIGHT of center ⇒ steer/rotate right (negative az);
            # error < 0 ⇒ line was LEFT ⇒ positive az. Fall back to the recovery side.
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
            # Phase-2 sweep starts toward the same side (default right if unknown).
            self._search_sweep_dir   = self._search_dir if self._search_dir != 0.0 else 1.0
            self._search_sweep_until = self._search_until + self._search_sweep_s

        cmd = Twist()
        if now < self._search_until:
            # Phase 1 — forward creep toward the last-seen side (capped angular).
            cmd.linear.x  = max(0.0, self._search_speed)
            cmd.angular.z = self._search_dir * self._search_angular
        else:
            # Phase 2 — in-place rotation sweep, reversing every search_sweep_s. The
            # first half-segment scans the entry side; subsequent segments are doubled so
            # the sweep covers symmetric ±search_sweep_s arc around the entry heading.
            if now >= self._search_sweep_until:
                self._search_sweep_dir   = -self._search_sweep_dir
                self._search_sweep_until = now + 2.0 * self._search_sweep_s
            cmd.linear.x  = 0.0
            cmd.angular.z = self._search_sweep_dir * self._search_rotate_z
        self._latest_cmd = cmd
        # Keep the slew limiter in sync so PD resumes from the issued angular on re-acquire.
        self._prev_az = cmd.angular.z
        return cmd

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

        Stale-frame safety (adaptive_new fix): the camera stalls 0.4-1.4 s; driving the
        last command forward at full speed through a blind gap runs the robot off a curve.
        If no new frame has arrived for frame_stale_s, scale the published LINEAR speed by
        stale_speed_scale (steering is held — turning in place toward the last-seen line is
        safer than translating blind). Resumes full speed the moment a fresh frame lands.
        This is a mitigation; the real fix for the low fps is on the camera/compute side.
        """
        cmd = self._latest_cmd
        blind_for = (time.monotonic() - self._last_frame_t) if self._last_frame_t is not None else 0.0
        if (self._frame_blind_s > 0 and self._last_frame_t is not None
                and blind_for > self._frame_blind_s):
            # Multi-second camera FREEZE (opt6-9: both streams hang 3-9 s): fully stop so
            # the recovery rotation can't spin the robot blind. Resumes the instant a
            # fresh frame lands (_last_frame_t updates in the image callback).
            self._pub_cmd.publish(Twist())
        elif (self._frame_stale_s > 0 and self._last_frame_t is not None
                and blind_for > self._frame_stale_s):
            stale = Twist()
            stale.linear.x  = cmd.linear.x * self._stale_speed_scale
            stale.angular.z = cmd.angular.z
            self._pub_cmd.publish(stale)
        else:
            self._pub_cmd.publish(cmd)

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
