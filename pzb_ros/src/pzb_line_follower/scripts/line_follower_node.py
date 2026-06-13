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
                                                        (decoupled from camera FPS)
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
        # Post-intersection acquisition + stuck-lock watchdog
        self.declare_parameter('acquire_guard_s',         0.4)
        self.declare_parameter('stuck_lock_s',            1.0)
        self.declare_parameter('stuck_lock_band_px',      16)
        self.declare_parameter('stuck_lock_var_px',       6)
        # Dashed-crossing behavior
        self.declare_parameter('dashed_recovery_enabled', False)
        self.declare_parameter('max_error_jump_px',       60)
        # Anti-stutter: angular_smooth_alpha EMAs the output angular.z (0=frozen, 1=off);
        # error_median_n median-filters the steering error so a single cx-teleport frame
        # can't swing the command.
        self.declare_parameter('angular_smooth_alpha',    0.4)
        self.declare_parameter('error_median_n',          3)
        # Search-then-stop loss recovery: on line loss, steer toward the last-seen side at
        # low forward speed for up to search_timeout_s, then stop. Never reverses.
        self.declare_parameter('search_timeout_s',        1.2)
        self.declare_parameter('search_speed_mps',        0.04)
        self.declare_parameter('search_angular_z',        0.25)
        # Rotate-until-found recovery: after the forward creep, do an in-place rotation
        # sweep toward the last-seen side (linear=0), reversing every search_sweep_s so it
        # scans both sides, until a line is re-acquired. search_rotate_z caps the sweep yaw.
        self.declare_parameter('search_rotate_z',         0.30)
        self.declare_parameter('search_sweep_s',          2.0)
        # Blind-stall full stop: past frame_blind_s with no fresh frame, publish (0,0) so a
        # multi-second camera freeze can't spin the robot blind; resumes when frames return.
        self.declare_parameter('frame_blind_s',           1.0)
        # Curve handling: on a sharp curve only one line is visible (normal, not a loss).
        # curve_min_speed keeps the robot advancing so the line stays in frame while steering.
        self.declare_parameter('curve_min_speed',         0.06)
        # Steering slew-rate limit: max angular.z change per frame (rad/s). Caps single-frame
        # reversals without lagging a sustained ramp on a real curve. 0 disables.
        self.declare_parameter('angular_slew_max',        0.12)
        # Sharp-turn slew bypass: on a genuine sharp turn (|error| >= slew_bypass_error_px,
        # consistent sign) use the larger angular_slew_max_sharp so steering reaches the PD
        # value in 1-2 frames instead of ramping past the curve.
        self.declare_parameter('slew_bypass_error_px',    80)
        self.declare_parameter('angular_slew_max_sharp',  0.45)
        # Medium-curve gain boost: on a confirmed turn (sustained same-sign |err| >=
        # slew_bypass_error_px) multiply Kp by curve_gain so medium curves don't under-steer.
        # Only on a confirmed same-sign turn, so it can't amplify small-error jitter.
        self.declare_parameter('curve_gain',              1.5)
        # Open-loop "turn until aligned" latch (opt-in, default off). On a confirmed turn,
        # latch a fixed yaw (turn_latch_z) at reduced speed and hold until the camera says
        # aligned (|err| < turn_latch_exit_px for turn_latch_exit_frames) or turn_latch_max_s.
        self.declare_parameter('turn_latch_enabled',      False)
        self.declare_parameter('turn_latch_error_px',     55)
        self.declare_parameter('turn_latch_frames',       3)
        self.declare_parameter('turn_latch_z',            0.55)
        self.declare_parameter('turn_latch_exit_px',      25)
        self.declare_parameter('turn_latch_exit_frames',  3)
        self.declare_parameter('turn_latch_max_s',        2.5)
        self.declare_parameter('turn_latch_speed',        0.05)
        # YOLO dashed-gated turn signs. At a confirmed dashed crossing, consume the latest
        # turn class: letRecto = cross straight; left/right = open-loop arc into that lane.
        # Only turn classes are latched here; slow/stop classes go to the traffic FSM.
        self.declare_parameter('turn_sign_left_class',    'letIzquierda')
        self.declare_parameter('turn_sign_right_class',   'letDerecha')
        self.declare_parameter('turn_sign_straight_class','letRecto')
        self.declare_parameter('turn_sign_stale_s',       4.0)   # ignore a sign older than this
        self.declare_parameter('cross_turn_z',            0.6)   # yaw rate during the lane turn
        self.declare_parameter('cross_turn_s',            1.6)   # turn duration (s) into the lane
        self.declare_parameter('cross_turn_speed',        0.06)  # forward speed during the turn
        # Crossing-state gap handling: after the turn/straight phase, coast forward at this
        # speed (no line steering) through the inter-intersection gap until a real line
        # returns (>=2 line slots for crossing_exit_frames frames), then resume centering.
        self.declare_parameter('crossing_coast_speed',    0.06)
        self.declare_parameter('crossing_exit_frames',    3)
        # YOLO turn arrow → turn at the sign's closest point (area peak-then-drop). The arrow
        # (own /yolo/turn_sign topic) latches a direction and tracks a running-max bbox area
        # while in view. The turn fires when the max has been close (>= turn_peak_min_area)
        # and the area has dropped below turn_peak_drop_frac x that max (just past closest).
        # letRecto / no arrow → cross straight. turn_sign_only_enabled is the master switch;
        # turn_sign_min_area_frac is a freshness floor, not the trigger.
        self.declare_parameter('turn_sign_only_enabled',  True)
        self.declare_parameter('turn_sign_topic',         '/yolo/turn_sign')
        self.declare_parameter('turn_sign_min_area_frac', 0.0)   # freshness floor only (0=off)
        self.declare_parameter('turn_sign_rearm_gap_s',   4.0)
        self.declare_parameter('turn_peak_min_area',      0.03)  # arrow must get this close before a turn
        self.declare_parameter('turn_peak_drop_frac',     0.7)   # fire when area < this × running-max
        # Post-fire cooldown: after a sign turn fires, block any new sign turn for this long
        # (one sign = one turn) so the robot won't re-fire on the same sign mid-maneuver.
        self.declare_parameter('turn_fire_cooldown_s',    8.0)
        # Teach-by-demonstration sign actions: replay a recorded /cmd_vel maneuver
        # (config/sign_actions/<sign>.csv) open-loop when a fresh turn sign is latched at a
        # dashed crossing, then resume line-following.
        self.declare_parameter('sign_action_enabled',     True)
        self.declare_parameter('sign_action_dir',         '')
        # Commit-nudge sign turn: at a dashed crossing + fresh sign, stop+center then a short
        # slow nudge, then resume line-following (it latches the new branch).
        self.declare_parameter('commit_speed',            0.04)
        self.declare_parameter('commit_w',                0.30)
        self.declare_parameter('commit_s',                2.9)
        self.declare_parameter('commit_center_s',         0.4)
        self.declare_parameter('commit_forward_s',        3.0)
        # Curve lockout: while cornering on the solid line (|median error| past
        # curve_lockout_error_px for curve_lockout_frames frames) hold the latched sign
        # instead of consuming it, so the open-loop arc can't fight the PD mid-curve.
        self.declare_parameter('curve_lockout_error_px',  70.0)
        self.declare_parameter('curve_lockout_frames',    3)
        # Reference line-follow control. control_mode='ref' uses the center-pick + soft-dir
        # control law; 'pd' = the original cx-PD path.
        self.declare_parameter('control_mode',            'ref')
        self.declare_parameter('ref_kp',                  1.5)
        self.declare_parameter('ref_max_w',               2.0)
        self.declare_parameter('ref_direction_alpha',     0.35)
        self.declare_parameter('ref_direction_slew_rate', 10.0)
        self.declare_parameter('ref_omega_alpha',         1.0)
        self.declare_parameter('ref_omega_slew_rate',     10.0)
        self.declare_parameter('ref_soft_dir_exp',        0.75)
        self.declare_parameter('ref_curve_scale_k',       0.75)
        self.declare_parameter('ref_angular_scale_k',     0.45)
        self.declare_parameter('ref_lost_speed_scale',    0.65)
        self.declare_parameter('ref_deadband',            0.0)
        # Curve dashed-suppression: suppress a "dashed" classification while |steering error|
        # is past this. A real crossing is approached head-on (small error); a sharp curve
        # (large error) breaks the line into co-linear blobs that falsely read as a dash row.
        self.declare_parameter('dashed_suppress_error_px', 45)
        # Tight-turn speed drop: above this |error| the curve is too tight to take at curve
        # speed, so drop to sharp_turn_speed. Above sharp_turn_threshold so ordinary bends
        # keep curve_min_speed.
        self.declare_parameter('tight_turn_error_px',     110)
        # Error-driven early braking: brake off the immediate |error| (not the slew-limited
        # angular.z, which lags the turn) so the robot slows the instant a turn is seen.
        # curve_brake_error_px = the |error| at which the full curve_speed_reduction applies.
        self.declare_parameter('curve_brake_error_px',    90)
        # Stale-frame safety: if no new frame for frame_stale_s, the 20 Hz publisher scales
        # linear speed by stale_speed_scale so the robot doesn't drive blind. 0 disables.
        self.declare_parameter('frame_stale_s',           0.4)
        self.declare_parameter('stale_speed_scale',       0.0)
        # Direction hysteresis: once steering commits to a side, the error must exceed
        # this opposite-side threshold (px) before the command may reverse — kills the
        # ±jitter around center from the few-px cx wobble that the dead-band alone misses.
        self.declare_parameter('steer_hysteresis_px',     6)
        # Error-saturation junction guard: if |error| stays at/above this for
        # error_sat_frames frames, the detector has lost the line at a junction (cx pinned at
        # an edge) — route to search-then-stop instead of crawling into the wall.
        self.declare_parameter('error_sat_px',            150)
        self.declare_parameter('error_sat_frames',        6)
        # Perpendicular dashed-alignment
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
        self._curve_gain          = float(self.get_parameter('curve_gain').value)
        self._tl_en       = bool(self.get_parameter('turn_latch_enabled').value)
        self._tl_err      = float(self.get_parameter('turn_latch_error_px').value)
        self._tl_frames   = int(self.get_parameter('turn_latch_frames').value)
        self._tl_z        = float(self.get_parameter('turn_latch_z').value)
        self._tl_exit     = float(self.get_parameter('turn_latch_exit_px').value)
        self._tl_exit_n   = int(self.get_parameter('turn_latch_exit_frames').value)
        self._tl_max_s    = float(self.get_parameter('turn_latch_max_s').value)
        self._tl_speed    = float(self.get_parameter('turn_latch_speed').value)
        # Turn-latch runtime state
        self._tl_active   = False
        self._tl_sign     = 0.0
        self._tl_start    = None
        self._tl_arm      = 0
        self._tl_exit_c   = 0
        # YOLO turn-sign params + latch state
        self._sign_left   = str(self.get_parameter('turn_sign_left_class').value)
        self._sign_right  = str(self.get_parameter('turn_sign_right_class').value)
        self._sign_straight = str(self.get_parameter('turn_sign_straight_class').value)
        self._sign_stale_s = float(self.get_parameter('turn_sign_stale_s').value)
        self._cross_turn_z = float(self.get_parameter('cross_turn_z').value)
        self._cross_turn_s = float(self.get_parameter('cross_turn_s').value)
        self._cross_turn_speed = float(self.get_parameter('cross_turn_speed').value)
        self._crossing_coast_speed = float(self.get_parameter('crossing_coast_speed').value)
        self._crossing_exit_frames = int(self.get_parameter('crossing_exit_frames').value)
        self._turn_sign   = None     # 'left' | 'right' | 'straight' | None (latched)
        self._turn_sign_t = None     # monotonic time the turn sign was last seen
        self._cross_turn_start = None  # set when the in-crossing turn arc begins
        # YOLO turn arrow → turn at the area peak (see _cb_yolo_sign + the peak-turn block)
        self._turn_sign_only = bool(self.get_parameter('turn_sign_only_enabled').value)
        self._turn_sign_topic = str(self.get_parameter('turn_sign_topic').value)
        self._turn_sign_min_area = float(self.get_parameter('turn_sign_min_area_frac').value)
        self._turn_sign_rearm_gap_s = float(self.get_parameter('turn_sign_rearm_gap_s').value)
        self._turn_peak_min_area = float(self.get_parameter('turn_peak_min_area').value)
        self._turn_peak_drop_frac = float(self.get_parameter('turn_peak_drop_frac').value)
        # Teach-by-demonstration sign actions
        self._sign_action_enabled = bool(self.get_parameter('sign_action_enabled').value)
        self._sign_action_dir = str(self.get_parameter('sign_action_dir').value)
        self._sign_actions = self._load_sign_actions() if self._sign_action_enabled else {}
        self._action_replay_active = False
        self._action_replay_start = None
        self._action_replay_sign = None
        # Commit-nudge sign turn — the active turn model
        self._commit_speed = float(self.get_parameter('commit_speed').value)
        self._commit_w = float(self.get_parameter('commit_w').value)
        self._commit_s = float(self.get_parameter('commit_s').value)
        self._commit_center_s = float(self.get_parameter('commit_center_s').value)
        self._commit_forward_s = float(self.get_parameter('commit_forward_s').value)
        self._commit_phase = 0       # 0 idle | 1 center | 2 fwd-enter | 3 arc-nudge
        self._commit_phase_t = None
        self._commit_dir = None
        self._curve_lockout_error_px = float(self.get_parameter('curve_lockout_error_px').value)
        self._curve_lockout_frames = int(self.get_parameter('curve_lockout_frames').value)
        self._turn_sign_area = 0.0       # latest bbox area fraction of the latched turn arrow
        self._turn_peak_max = 0.0        # running max area while the arrow is in view
        self._turn_sign_last_seen = None # last time a turn arrow was in view
        self._turn_peak_reached = False  # running-max has crossed turn_peak_min_area this approach
        self._curve_lockout_streak = 0   # consecutive cornering-on-solid frames (sign lockout)
        self._turn_fire_cooldown_s = float(self.get_parameter('turn_fire_cooldown_s').value)
        self._last_turn_fire_t = None    # monotonic time the last sign turn fired (cooldown)
        # Reference control state + params
        self._control_mode = str(self.get_parameter('control_mode').value)
        self._ref_kp = float(self.get_parameter('ref_kp').value)
        self._ref_max_w = float(self.get_parameter('ref_max_w').value)
        self._ref_dir_alpha = float(self.get_parameter('ref_direction_alpha').value)
        self._ref_dir_slew = float(self.get_parameter('ref_direction_slew_rate').value)
        self._ref_omega_alpha = float(self.get_parameter('ref_omega_alpha').value)
        self._ref_omega_slew = float(self.get_parameter('ref_omega_slew_rate').value)
        self._ref_soft_exp = float(self.get_parameter('ref_soft_dir_exp').value)
        self._ref_curve_k = float(self.get_parameter('ref_curve_scale_k').value)
        self._ref_ang_k = float(self.get_parameter('ref_angular_scale_k').value)
        self._ref_lost_scale = float(self.get_parameter('ref_lost_speed_scale').value)
        self._ref_deadband = float(self.get_parameter('ref_deadband').value)
        self._ref_raw_dir = 0.0
        self._ref_filtered_dir = 0.0
        self._ref_omega_filt = 0.0
        self._ref_prev_dir = 0.0
        self._ref_last_ctrl_t = None
        # peak-turn open-loop arc state
        self._so_turn_active = False
        self._so_turn_dir = 0.0
        self._so_turn_start = None
        self._so_armed = True
        self._so_open_loop = False   # True while the arc is being published direct to /cmd_vel
        self._dashed_suppress_error_px = float(self.get_parameter('dashed_suppress_error_px').value)
        self._tight_turn_error_px = float(self.get_parameter('tight_turn_error_px').value)
        self._curve_brake_error_px = float(self.get_parameter('curve_brake_error_px').value)
        self._frame_stale_s       = float(self.get_parameter('frame_stale_s').value)
        self._stale_speed_scale   = float(self.get_parameter('stale_speed_scale').value)
        self._steer_hyst_px       = float(self.get_parameter('steer_hysteresis_px').value)
        self._error_sat_px        = float(self.get_parameter('error_sat_px').value)
        self._error_sat_frames    = int(self.get_parameter('error_sat_frames').value)
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
        # ROI dash slope to a true world tilt angle.
        self._roi_aniso           = 0.889

        topic_in  = self.get_parameter('topic_image_in').value
        topic_cmd = self.get_parameter('topic_cmd_vel').value

        # Use all 4 Jetson Nano cores for OpenCV.
        cv2.setNumThreads(4)

        # D-term state: use actual elapsed time in seconds so the derivative is
        # FPS-invariant instead of "change per frame".
        self._prev_error   = 0.0
        self._prev_error_t = None   # monotonic timestamp of last image callback

        # Approach-speed memory: rolling cruise speed captured in solid PD mode, so the
        # dashed open-loop crossing continues at the same rate instead of a fixed speed.
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
        # Live-error abort of a false dashed-on-curve: consecutive same-sign frames whose
        # live cx error exceeds the suppress threshold.
        self._dashed_live_break_count = 0
        self._dashed_live_break_sign  = 0

        # Perpendicular dashed-alignment: rolling buffer of recent valid dash-slope samples
        # (median-filtered to reject the noisy per-frame slope) and a latch marking the
        # align sub-phase complete.
        self._slope_buf      = deque(maxlen=self._align_median_n)
        self._aligned_latch  = False  # True once perpendicular → proceed to crossing
        self._align_done_t   = None   # monotonic time the align sub-phase completed

        # Crossing-state ownership (inter-intersection gap fix). Once a dashed crossing is
        # confirmed the crossing FSM OWNS control (align → turn/straight → coast) and does
        # NOT revert to line-following while the robot is between intersections — in that
        # gap there is no continuous line, so the detector's (L+R)/2 fallback would steer
        # to noise. The FSM exits only when a REAL continuous line returns: ≥2 line slots
        # for _crossing_exit_frames consecutive frames. The post-cross "coast" drives at
        # the approach speed (straight) until then, instead of the old full-stop Phase C.
        self._crossing_active   = False
        self._crossing_done_t   = None   # monotonic time the turn/straight phase finished
        self._real_line_streak  = 0      # consecutive frames with ≥2 line slots

        # Turn approach delay: when the camera first sees a sharp turn, coast at reduced
        # speed for _turn_approach_delay seconds before applying full angular correction.
        # This compensates for the camera looking ahead of the robot body.
        self._turn_first_seen_t = None   # monotonic time when large error first appeared

        # Post-intersection acquisition guard + stuck-lock watchdog.
        # _was_dashed tracks the latched line_type to detect the dashed→solid edge and
        # re-seed anchors + arm a steering cap. _acq_until caps angular output until this
        # time after the exit. _err_hist holds recent (t, error) samples so the watchdog
        # can detect a frozen, non-converging off-center lock (chasing a wall/seam).
        self._was_dashed = False
        self._acq_until  = None
        self._err_hist   = deque()

        self._detector = CenterLineDetector(debug=self._pub_debug)

        # Publishers
        self._pub_cx        = self.create_publisher(Int32,   '/line_follower/cx',        _RELIABLE_QOS)
        self._pub_error     = self.create_publisher(Float32, '/line_follower/error',      _RELIABLE_QOS)
        self._pub_line_type = self.create_publisher(String,  '/line_follower/line_type',  _RELIABLE_QOS)
        self._pub_cmd       = self.create_publisher(Twist,   topic_cmd,                   _RELIABLE_QOS)
        # Open-loop turn bypass: during the sign arc the normal command path
        # (→ /cmd_vel_desired_raw → slew_limiter → velocity_controller PI → /cmd_vel) mangles
        # the arc (PI on the noisy /robot_vel.w adds a spurious correction, slew ramps it).
        # So while _so_turn_active we send zero to topic_cmd (the chain idles to zero) and
        # emit the arc Twist straight onto /cmd_vel from this publisher.
        self._pub_cmd_direct = self.create_publisher(Twist,  '/cmd_vel',                  _RELIABLE_QOS)
        # Debug image is large (~0.5 MB/msg). BEST_EFFORT so a slow subscriber / bag
        # recorder can never ACK-back-pressure the follower (a RELIABLE 0.5 MB topic
        # contributed to multi-second whole-graph stalls when recorded on the Jetson).
        self._pub_debug_img = self.create_publisher(Image,   '/line_follower/debug_image', _BEST_EFFORT_QOS)

        # Subscriber — raw Image; BEST_EFFORT matches camera publisher
        self.create_subscription(Image, topic_in, self._image_cb, _BEST_EFFORT_QOS)

        # Traffic speed scale (optional — defaults to 1.0 if never published)
        self.create_subscription(Float32, '/traffic_speed_scale', self._cb_speed_scale, _RELIABLE_QOS)

        # YOLO turn arrow (optional) — its OWN channel (/yolo/turn_sign) so a nearer
        # non-turn sign can't mask it on /yolo/sign. Latches the most-recent turn
        # direction + bbox area; the turn fires when the area is large enough (close).
        # Non-turn (slow/stop/give-way) classes stay on /yolo/sign for the traffic FSM.
        self.create_subscription(
            String, self._turn_sign_topic, self._cb_yolo_sign, 10)

        # 20 Hz cmd_vel publish timer — decoupled from camera FPS.
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
        # variable, stall-prone rate. The 20 Hz publish timer scales speed down if this
        # stamp goes stale (see _cmd_publish_cb).
        self._last_frame_t = time.monotonic()

        # Decode raw BGR image — zero-copy view into msg.data
        full = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

        # Fixed-height ROI: always crop the same source rows (bottom half) so the vertical
        # scale (source_rows → target_h) is constant frame-to-frame. A variable crop made
        # the squash factor jump between frames, shifting the same physical line to a
        # different ROI row and apparent slope. The bottom half keeps the sharp-turn
        # look-ahead (detector's far-field filter discards the top 15%); horizontal scale
        # (and the cx→steering mapping) is unchanged: full width → target_w.
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

        # Curve dashed-suppression: a real crossing is approached head-on (|error| small);
        # a sharp curve breaks the single line into co-linear blobs that mimic a dash row,
        # so the classifier falsely returns "dashed" and zeroes steering mid-curve. If the
        # (median-filtered) error magnitude is past dashed_suppress_error_px, force solid and
        # keep following the curve. Uses last frame's accepted error (one-frame lag is
        # harmless since a curve persists).
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
            # On the first raw-dashed frame (approach), arm the acquisition cap so the noisy
            # approach frames can't produce a violent steering spike before dashed is
            # confirmed. Reuses the same _acq_until clamp as the exit guard.
            if self._dashed_streak == 0:
                self._acq_until = _t_latch + self._acquire_guard_s
            self._dashed_streak += 1
        else:
            self._dashed_streak = 0

        _confirmed_dashed = (self._dashed_streak >= self._dashed_confirm_n)
        if _confirmed_dashed:
            self._dashed_latch_t = _t_latch
            if self._dashed_first_t is None:
                # Entering a fresh dashed crossing — reset alignment + turn-arc state and
                # take crossing-state ownership (held across the gap until a real line).
                self._dashed_first_t = _t_latch
                self._slope_buf.clear()
                self._aligned_latch = False
                self._align_done_t  = None
                self._cross_turn_start = None   # turn arc starts fresh per crossing
                self._crossing_active  = True
                self._crossing_done_t  = None
                self._real_line_streak = 0

        if _confirmed_dashed or (self._dashed_latch_t is not None
                and (_t_latch - self._dashed_latch_t) < self._dashed_latch_s):
            line_type = 'dashed'
        else:
            line_type = 'solid'
            self._dashed_first_t = None  # reset so next encounter starts fresh

        # Live-error abort of a false dashed-on-curve. The suppress guard above checks
        # _last_good_error, which is frozen during a latched crossing — so a curve that
        # latched dashed at small error then crosses straight while the true error balloons
        # off-track. Re-check the live error here: if it exceeds dashed_suppress_error_px for
        # >=2 consecutive same-sign frames the "dashes" are a curve, not a crossing — break
        # the latch and resume solid PD. A real head-on crossing keeps a small live error.
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
                # This was a false dashed-on-curve, not a real crossing — release
                # crossing-state ownership so normal centering resumes immediately.
                self._crossing_active  = False
                self._real_line_streak = 0
        else:
            self._dashed_live_break_count = 0
            self._dashed_live_break_sign  = 0

        # Feed the dash-slope median buffer only from clean, plausible dash-row readings.
        # The raw slope degrades to garbage at the end of a crossing (robot overlaps the
        # dashes / sees the straight lines ahead), producing large false spikes. Guards:
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

        # Error-saturation junction guard: at a hard junction the detector can pin cx at an
        # image edge for several frames, so the robot crawls into the wall chasing an error
        # it can never null. Count consecutive saturated frames; once past error_sat_frames,
        # treat it as a near-loss (→ search-then-stop). Only counts in solid mode and resets
        # when the error returns in range, so a genuine sharp curve never trips it.
        _sat_now = (line_type == 'solid'
                    and abs(float(cx - self._img_w // 2)) >= self._error_sat_px)
        if _sat_now:
            self._error_sat_count += 1
        else:
            self._error_sat_count = 0
        error_saturated = self._error_sat_count >= self._error_sat_frames

        now = time.monotonic()

        # Dashed→solid edge (intersection exit): re-seed the tracker so the next assignment
        # is anchored at image center, and arm a steering cap so the first large-error frame
        # can't pivot ~90° onto a leftover crossing dash. _was_dashed tracks the previous
        # resolved line_type.
        if self._was_dashed and line_type == 'solid':
            self._detector.reset_tracker_anchors()
            self._acq_until = now + self._acquire_guard_s
            self._err_hist.clear()   # drop the pre-exit error history
        self._was_dashed = (line_type == 'dashed')

        # Crossing-state ownership across the inter-intersection gap. Once a crossing is
        # active the FSM keeps control until a real continuous line returns (>=2 line slots
        # for _crossing_exit_frames frames). In the gap the detector may briefly report
        # "solid" with a spurious (L+R)/2 target over noise; holding crossing-state prevents
        # steering to it. The live-error dashed-break above already clears _crossing_active
        # for a genuine curving line, so a real curve is not trapped here.
        _n_vis_now = sum(self._detector.line_flags.values())
        if self._crossing_active:
            if _n_vis_now >= 2:
                self._real_line_streak += 1
            else:
                self._real_line_streak = 0
            if self._real_line_streak >= self._crossing_exit_frames:
                # A real lane is back — exit crossing-state and resume normal centering.
                self._crossing_active = False
                self._detector.reset_tracker_anchors()
                self._acq_until = now + self._acquire_guard_s
                self._err_hist.clear()

        # ── Peak-area turn (primary trigger) ──────────────────────────────────────
        # The arrow fires the turn at the sign's closest point (max area = closest): the
        # area rises to a peak as the robot approaches, then falls. Fire when either
        # (a) the running-max has been close (>= turn_peak_min_area) and the current area has
        # dropped below turn_peak_drop_frac x that max (peak-drop while in view), or
        # (b) the arrow leaves view having reached turn_peak_min_area (past closest).
        # Direction is the last in-view latch. The dashed-crossing FSM below is the fallback.
        # While _so_turn_active the command is open-loop and bypasses the control chain.
        _so_cmd = None
        self._so_open_loop = False   # default: normal chain; set True only when an arc fires

        # ── Commit-nudge sign turn (the active turn model) ─────────────────────────
        # At a dashed crossing with a fresh sign: stop+center for commit_center_s, then a
        # short slow nudge (v=commit_speed, w=±commit_w / 0 straight) for commit_s, then
        # resume line-following (it latches the perpendicular branch). Published via the
        # normal chain. Mirrors FollowerCore (C++).
        # _commit_phase: 0 idle | 1 centering | 2 forward-creep | 3 nudge.
        def _publish_diag():
            cx_msg = Int32();  cx_msg.data = cx;  self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = float(cx - self._img_w // 2);  self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type;  self._pub_line_type.publish(type_msg)

        if self._commit_phase == 1:   # STOP-AND-CENTER
            if (now - self._commit_phase_t) < self._commit_center_s:
                err = float(cx - self._img_w // 2)
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = max(-self._commit_w, min(self._commit_w, -self._kp * err))
                self._latest_cmd = cmd;  self._prev_az = cmd.angular.z;  self._error_hist.clear()
                _publish_diag();  return
            self._commit_phase = 2;  self._commit_phase_t = now
        if self._commit_phase == 2:   # FORWARD ENTER — advance into the intersection
            if (now - self._commit_phase_t) < self._commit_forward_s:
                cmd = Twist()
                cmd.linear.x  = self._commit_speed * max(self._speed_scale, 0.0)
                cmd.angular.z = 0.0
                self._latest_cmd = cmd;  self._prev_az = 0.0;  self._error_hist.clear()
                _publish_diag();  return
            self._commit_phase = 3;  self._commit_phase_t = now
        if self._commit_phase == 3:   # ARC NUDGE — forward WHILE turning into the branch
            if (now - self._commit_phase_t) < self._commit_s:
                cmd = Twist()
                cmd.linear.x  = self._commit_speed * max(self._speed_scale, 0.0)
                cmd.angular.z = (self._commit_w if self._commit_dir == 'left'
                                 else -self._commit_w if self._commit_dir == 'right' else 0.0)
                self._latest_cmd = cmd;  self._prev_az = cmd.angular.z;  self._error_hist.clear()
                _publish_diag();  return
            # done → resume normal line-following (latches the new branch)
            self._commit_phase = 0;  self._commit_dir = None
            self._detector.reset_tracker_anchors();  self._err_hist.clear()
            self._acq_until = now + self._acquire_guard_s
            self._crossing_active = False
        # TRIGGER: dashed crossing + fresh sign + not in cooldown → begin stop-and-center
        if self._commit_phase == 0 and self._turn_sign_only:
            _at_dashed = (self._crossing_active or line_type == 'dashed')
            _fs = self._fresh_turn_sign()
            _in_cooldown = (self._last_turn_fire_t is not None
                            and (now - self._last_turn_fire_t) < self._turn_fire_cooldown_s)
            if _at_dashed and not _in_cooldown and _fs in ('left', 'right'):
                self._commit_phase = 1;  self._commit_phase_t = now;  self._commit_dir = _fs
                self._last_turn_fire_t = now
                self._turn_sign = None;  self._turn_sign_t = None     # consume
                self.get_logger().info(f'Commit-nudge turn {_fs.upper()}.')
                cmd = Twist()
                self._latest_cmd = cmd;  self._prev_az = 0.0;  self._error_hist.clear()
                _publish_diag();  return

        # The synthetic peak-area arc below is superseded by the commit-nudge turn (above).
        # Disabled (gated False), kept parked for reference.
        if False and self._turn_sign_only:
            _in_view = (self._turn_sign_last_seen is not None
                        and (now - self._turn_sign_last_seen) <= self._turn_sign_rearm_gap_s)
            # Curve lockout: if the robot is actively cornering on the REAL solid line
            # (large sustained steering error), DON'T let a sign arc preempt it — finish
            # the corner first, then fire. We hold the latch (don't consume it) while
            # locked out. Uses last frame's accepted error (this frame's PD error is
            # computed later); a corner persists across frames so the one-frame lag is
            # harmless — same pattern as the dashed-suppress guard above.
            if line_type == 'solid' and abs(self._last_good_error) >= self._curve_lockout_error_px:
                self._curve_lockout_streak += 1
            else:
                self._curve_lockout_streak = 0
            _curve_locked = self._curve_lockout_streak >= self._curve_lockout_frames

            # Arm once a fresh direction is latched (no longer gated on "out of view").
            if not self._so_turn_active and self._fresh_turn_sign() in ('left', 'right'):
                self._so_armed = True
            if self._so_turn_active:
                if (now - self._so_turn_start) < self._cross_turn_s:
                    _so_cmd = Twist()
                    _so_cmd.linear.x  = self._cross_turn_speed * max(self._speed_scale, 1.0)
                    _so_cmd.angular.z = self._so_turn_dir * self._cross_turn_z
                else:
                    # Arc done → don't resume PD immediately (it thrashes between line
                    # fragments at the busy intersection and goes out of bounds). Hand off to
                    # the dashed-FSM Phase-C coast: drive straight, no line steering, until
                    # >=2 line slots are stable for crossing_exit_frames, then resume
                    # centering. Enter crossing-state already past the turn/straight phases so
                    # the next frame lands in Phase C coast directly.
                    self._so_turn_active = False
                    self._so_armed = False
                    self._turn_sign = None        # consumed → _do_turn=False, no re-fire
                    self._turn_sign_t = None
                    self._turn_peak_max = 0.0
                    self._turn_peak_reached = False
                    self._detector.reset_tracker_anchors()
                    self._err_hist.clear()
                    self._crossing_active = True
                    self._aligned_latch   = True
                    self._align_done_t    = now - 999.0   # straight-cross window already elapsed
                    self._cross_turn_start = now - 999.0  # Phase B' already elapsed
                    self._crossing_done_t = None          # Phase C will stamp it
                    self._real_line_streak = 0
            _in_cooldown = (self._last_turn_fire_t is not None
                            and (now - self._last_turn_fire_t) < self._turn_fire_cooldown_s)
            if (_so_cmd is None and not self._so_turn_active
                    and self._so_armed and not _curve_locked and not _in_cooldown):
                _td = self._turn_sign if self._turn_sign in ('left', 'right') else None
                # (a) clean peak-drop while still in view
                _fire_peak = (_in_view and _td is not None
                              and self._turn_peak_reached
                              and self._turn_sign_area
                              <= self._turn_peak_drop_frac * self._turn_peak_max)
                # (b) arrow left view after getting close (exited frame past closest)
                _fire_leave = ((not _in_view) and _td is not None
                               and self._turn_peak_reached)
                if _fire_peak or _fire_leave:
                    self._so_turn_active = True
                    self._so_turn_dir = 1.0 if _td == 'left' else -1.0
                    self._so_turn_start = now
                    self._last_turn_fire_t = now   # start the one-turn-per-sign cooldown
                    self.get_logger().info(
                        f'Sign turn {_td.upper()} '
                        f'({"peak-drop" if _fire_peak else "left-view"}: '
                        f'max={self._turn_peak_max:.3f}, now={self._turn_sign_area:.3f}).')
                    _so_cmd = Twist()
                    _so_cmd.linear.x  = self._cross_turn_speed * max(self._speed_scale, 1.0)
                    _so_cmd.angular.z = self._so_turn_dir * self._cross_turn_z

        if _so_cmd is not None:
            # OPEN-LOOP turn: route the arc DIRECTLY to /cmd_vel and send zero down the
            # normal chain so the slew_limiter + velocity_controller idle to zero (no PI
            # correction, no slew on the arc). _cmd_publish_cb reads _so_open_loop.
            self._so_open_loop = True
            self._latest_cmd = _so_cmd
            self._prev_az = _so_cmd.angular.z
            self._error_hist.clear()
            cx_msg = Int32();  cx_msg.data = cx
            self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = float(cx - self._img_w // 2)
            self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type
            self._pub_line_type.publish(type_msg)
        # The crossing FSM runs while a crossing is active or the current frame is dashed.
        # It is the fallback turn path (turn-into-lane) when the peak trigger didn't fire.
        elif self._crossing_active or line_type == 'dashed':
            # Intersection handling — align early, then cross straight, then coast:
            #   Phase A — Align/coast: drive forward at the approach speed while squaring up
            #             to the dashes, only within an early window (align_window_s) when
            #             the dash row is clean and ahead. Latches "aligned" the instant a
            #             clean median tilt is within the deadband — so an already-straight
            #             robot latches immediately and never re-steers. After the window,
            #             cross straight.
            #   Phase B — Straight crossing: angular.z = 0 for openloop_dur.
            #   Phase C — Coast: drive straight through the gap until a real lane returns.
            elapsed = (now - self._dashed_first_t) if self._dashed_first_t is not None else 0.0

            # Cross at the approach speed (clamped), fallback openloop_speed_mps.
            cross_speed = self._approach_speed
            if not (cross_speed > 1e-3):
                cross_speed = self._openloop_speed
            cross_speed = min(self._max_lin_speed,
                              max(self._min_lin_speed, cross_speed))
            openloop_dur = self._openloop_dist_m / max(cross_speed, 0.01)

            align_z, tilt_deg, have_tilt = self._alignment_cmd()
            # Always-straight commitment: when alignment is disabled the only required action
            # at a dashed crossing is to cross straight, so latch "aligned" immediately on
            # entry — go straight to Phase B (no align steering, no align_window_s wait).
            # Signs/YOLO choose the direction later.
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

            # The dashed FSM no longer turns — the commit-nudge block above owns all sign
            # turns. Here we only align → cross straight → coast.
            _do_turn = False

            if not self._aligned_latch:
                # Phase A — align/coast: forward + perpendicular alignment steering.
                cmd = Twist()
                cmd.linear.x  = cross_speed * max(self._speed_scale, 1.0)
                cmd.angular.z = align_z
            elif not _do_turn and (now - self._align_done_t) < openloop_dur:
                # Phase B — straight crossing at the approach speed (letRecto / no sign).
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
                # Phase C — coast through the inter-intersection gap. The turn/straight phase
                # is done but there is no continuous line yet; drive straight at the coast
                # speed (no line steering) and keep crossing-state until a real lane returns
                # (the >=2-slot exit gate above clears _crossing_active).
                if self._crossing_done_t is None:
                    self._crossing_done_t = now
                    self._turn_sign = None     # consume the sign so it can't re-fire
                    self._turn_sign_t = None
                cmd = Twist()
                cmd.linear.x  = self._crossing_coast_speed * max(self._speed_scale, 1.0)
                cmd.angular.z = 0.0
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
            # Rotate-until-found recovery, also entered when |error| has been saturated for
            # several frames at a junction. Phase 1 creeps forward toward the last-seen side,
            # Phase 2 rotates in place sweeping both sides until the line returns — never
            # latches a permanent stop. See _search_recovery_cmd.
            self._search_recovery_cmd(now)

            # Still publish detections so monitors can see the loss
            cx_msg = Int32();  cx_msg.data = cx
            self._pub_cx.publish(cx_msg)
            err_msg = Float32();  err_msg.data = float(cx - self._img_w // 2)
            self._pub_error.publish(err_msg)
            type_msg = String();  type_msg.data = line_type
            self._pub_line_type.publish(type_msg)
        elif self._is_stuck_lock(now, float(cx - self._img_w // 2)):
            # Stuck-lock watchdog: the detector reports a real contour, but the error has
            # been frozen far off-center and is not converging — the robot is "following" a
            # stationary non-line (wall / floor-panel seam). Route into the same
            # rotate-until-found recovery as a true loss.
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

            # ── Reference control ──────────────────────────────────────────────
            # Smooth normalized-direction following on the solid line. Runs the reference
            # center-pick on the same ROI, then the soft-dir control law. The dashed / sign /
            # open-loop paths above already returned, so this only governs solid following.
            # control_mode='pd' falls back to the original cx-PD below.
            if self._control_mode == 'ref':
                direction, ref_found = self._detector.ref_center_line(img, self._ref_prev_dir)
                self._ref_prev_dir = direction if ref_found else 0.0
                dt = 0.05 if self._ref_last_ctrl_t is None else max(0.015, min(0.12, now - self._ref_last_ctrl_t))
                speed_scale = 1.0
                if ref_found:
                    self._ref_raw_dir = ((1.0 - self._ref_dir_alpha) * self._ref_raw_dir
                                         + self._ref_dir_alpha * direction)
                    target = self._ref_raw_dir if abs(self._ref_raw_dir) >= self._ref_deadband else 0.0
                    max_dd = self._ref_dir_slew * dt
                    self._ref_filtered_dir = max(self._ref_filtered_dir - max_dd,
                                                 min(self._ref_filtered_dir + max_dd, target))
                    if abs(self._ref_filtered_dir) < self._ref_deadband:
                        self._ref_filtered_dir = 0.0
                else:
                    self._ref_filtered_dir = 0.0
                    speed_scale = self._ref_lost_scale
                self._ref_last_ctrl_t = now
                soft = self._ref_filtered_dir * abs(self._ref_filtered_dir) ** self._ref_soft_exp
                w_t = max(-self._ref_max_w, min(self._ref_max_w, -self._ref_kp * soft))
                w_sm = (1.0 - self._ref_omega_alpha) * self._ref_omega_filt + self._ref_omega_alpha * w_t
                maxd = self._ref_omega_slew * dt
                w = max(self._ref_omega_filt - maxd, min(self._ref_omega_filt + maxd, w_sm))
                w = max(-self._ref_max_w, min(self._ref_max_w, w))
                self._ref_omega_filt = w
                curve_s = 1.0 - self._ref_curve_k * min(1.0, abs(self._ref_filtered_dir))
                ang_s = 1.0 - self._ref_ang_k * min(1.0, abs(w) / max(1e-6, self._ref_max_w))
                v = self._linear_speed * speed_scale * curve_s * ang_s * max(self._speed_scale, 0.0)
                self._last_good_error = direction * (self._img_w // 2)
                self._prev_az = w
                cmd = Twist();  cmd.linear.x = float(v);  cmd.angular.z = float(w)
                self._latest_cmd = cmd
                cx_ref = int(round((direction + 1.0) * (self._img_w // 2)))
                cx_msg = Int32();  cx_msg.data = cx_ref;  self._pub_cx.publish(cx_msg)
                err_msg = Float32();  err_msg.data = float(direction * (self._img_w // 2));  self._pub_error.publish(err_msg)
                type_msg = String();  type_msg.data = line_type;  self._pub_line_type.publish(type_msg)
                return

            raw_error = float(cx - self._img_w // 2)

            # Error median filter (anti-stutter): the detector cx (hence error) is noisy and
            # jumps between features frame to frame, and the raw PD reversed steering each
            # time. Steering on the median of the last _error_median_n raw errors rejects a
            # single teleport frame while a sustained turn (error persists >= ceil(n/2)
            # frames) still passes through. Disabled when error_median_n <= 1.
            self._error_hist.append(raw_error)
            if self._error_median_n > 1 and len(self._error_hist) > 0:
                error = float(np.median(self._error_hist))
            else:
                error = raw_error
            self._last_good_error = error

            # Sharp-turn detection: a real sharp turn is a large, consistent-sign error,
            # distinct from the small-error sign-flipping jitter the slew limit damps. Detect
            # it from the median magnitude (noise-robust) plus a same-sign check. In sharp
            # mode the steering slew limit below is raised so the command reaches the PD value
            # in 1-2 frames instead of ramping past the curve.
            _same_sign = all((e > 0) == (error > 0)
                             for e in self._error_hist if abs(e) > self._dead_band)
            sharp_turn = (abs(error) >= self._slew_bypass_error_px and _same_sign)

            # ── Open-loop "turn until aligned" latch (opt-in, default off) ─────────
            # On a confirmed turn, rotate at a fixed yaw toward the line until the camera
            # says re-centered (or a safety cap). Bypasses PD entirely while latched.
            if self._tl_en:
                tl_cmd = self._turn_latch_step(error, _same_sign, now)
                if tl_cmd is not None:
                    self._latest_cmd     = tl_cmd
                    self._last_valid_cmd = tl_cmd
                    self._last_valid_t   = now
                    self._prev_az        = tl_cmd.angular.z
                    cx_msg = Int32();   cx_msg.data = cx;                self._pub_cx.publish(cx_msg)
                    err_msg = Float32(); err_msg.data = error;           self._pub_error.publish(err_msg)
                    type_msg = String(); type_msg.data = line_type;      self._pub_line_type.publish(type_msg)
                    return

            # Time-based D-term: divide by actual dt in seconds so the derivative gain is
            # FPS-invariant. At 30 fps, dt≈33 ms; at 5 fps, dt≈200 ms.
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

            # Acquisition guard: for a short window after the dashed→solid intersection exit,
            # cap angular output (like the turn-approach delay) so the tracker has time to
            # settle on the real center line before full steering is applied.
            if self._acq_until is not None and now < self._acq_until:
                _in_approach = True

            # PD steering. On a confirmed turn (sharp_turn) apply the medium-curve gain boost
            # so the same error commands a decisive correction. Gated on sharp_turn
            # (sustained same-sign |err|), so it never amplifies small-error jitter.
            if abs(error) <= self._dead_band:
                angular_z = 0.0
            else:
                kp = self._kp * (self._curve_gain if sharp_turn else 1.0)
                angular_z = float(np.clip(
                    -kp * error - self._kd * d_error,
                    -self._max_ang, self._max_ang,
                ))
                # The approach/acquisition cap throttles angular to 0.3*max_ang while the
                # body catches up to a just-seen turn or settles after an intersection. It
                # must not fight a confirmed sharp turn (full steering is needed then or the
                # line leaves the ROI), so skip the cap when sharp_turn is active.
                if _in_approach and not sharp_turn:
                    approach_cap = self._max_ang * 0.3
                    angular_z = float(np.clip(angular_z, -approach_cap, approach_cap))

            # Direction hysteresis + steering slew-rate limit. The slew-rate limit caps the
            # per-frame change in angular.z, so a one-frame reversal is bounded (anti-stutter)
            # while a steady same-sign ramp passes through at full rate (good curve tracking).
            # Hysteresis stops the command flipping sign on the few-px cx wobble around center.
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
            # Node-side steering slew-rate limit. On a genuine sharp turn the node slew is
            # bypassed so the full PD value is emitted at once — the downstream
            # twist_slew_limiter does the smoothing fast enough, and a second node-side limit
            # only re-adds lag. Bypassing can't reintroduce stutter (that's small-error
            # sign-flipping, which the same-sign sharp_turn gate excludes). Small errors keep
            # the gentle angular_slew_max for anti-stutter.
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
            # Error-driven early braking: brake off the immediate |error| (the look-ahead
            # signal) instead of waiting for the slew-limited angular.z to ramp. error_frac
            # ramps 0→1 as |error| grows from the dead band to curve_brake_error_px, so the
            # robot is already slowing the instant a turn appears in the ROI. Take whichever
            # of the two scales brakes more.
            _brake_span = max(1.0, self._curve_brake_error_px - self._dead_band)
            error_frac  = float(np.clip((abs(error) - self._dead_band) / _brake_span, 0.0, 1.0))
            speed_scale_error = 1.0 - self._curve_reduction * error_frac
            speed_scale = min(speed_scale_curve, speed_scale_error)
            min_frac  = self._min_lin_speed / self._linear_speed if self._linear_speed > 0 else 0.0
            linear_x  = min(self._max_lin_speed,
                            self._linear_speed * max(min_frac, speed_scale))

            # Visibility-based speed reduction: fewer visible lines → slower. Seeing only one
            # line is normal on a sharp curve (the other boundary leaves frame), so floor at
            # curve_min_speed — fast enough to keep advancing so the line stays in frame.
            n_vis = sum(self._detector.line_flags.values())
            if n_vis == 2:
                linear_x = min(linear_x, self._linear_speed * 0.6)
            elif n_vis <= 1:
                linear_x = max(self._sharp_turn_speed, self._curve_min_speed)

            # Sharp-turn override (error magnitude): a large error means a real bend — slow
            # but don't crawl, or the robot can't advance through it. Floor at curve_min_speed.
            if abs(error) > self._sharp_turn_threshold:
                linear_x = max(self._sharp_turn_speed, self._curve_min_speed)
            # Tight-turn override: on the tightest curves (|error| past tight_turn_error_px)
            # even fast full steering can't hold the line at curve speed, so drop to the
            # slowest sharp_turn_speed to shrink the turn radius. Only the tightest regime, so
            # normal curves keep curve_min_speed.
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

            # Track the steady cruise speed so the dashed open-loop crossing can continue at
            # the same rate. Only sample when actually cruising (near-centered, not in a
            # sharp-turn or low-visibility slowdown) so pre-intersection dips don't drag it down.
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

    def _turn_latch_step(self, error, same_sign, now):
        """Open-loop turn-until-aligned. Returns a Twist while latched/firing, else None.

        Enter: sustained same-sign |error| >= turn_latch_error_px for turn_latch_frames.
        Hold:  fixed yaw toward the line at turn_latch_speed.
        Exit:  |error| < turn_latch_exit_px for turn_latch_exit_frames (camera aligned),
               or turn_latch_max_s elapsed (safety cap).
        """
        if self._tl_active:
            if abs(error) < self._tl_exit:
                self._tl_exit_c += 1
            else:
                self._tl_exit_c = 0
            timed_out = (self._tl_start is not None
                         and (now - self._tl_start) >= self._tl_max_s)
            if self._tl_exit_c >= self._tl_exit_n or timed_out:
                self._tl_active = False
                self._tl_arm    = 0
                return None                       # aligned → hand back to PD
            cmd = Twist()
            cmd.linear.x  = self._tl_speed * max(self._speed_scale, 0.0)
            cmd.angular.z = self._tl_sign * self._tl_z
            return cmd

        # not latched — arm on a sustained confirmed turn
        if abs(error) >= self._tl_err and same_sign:
            self._tl_arm += 1
        else:
            self._tl_arm = 0
        if self._tl_arm >= self._tl_frames:
            self._tl_active = True
            # PD convention: az = -kp*error, so error>0 (line right) → steer right (az<0).
            self._tl_sign   = -1.0 if error > 0 else 1.0
            self._tl_start  = now
            self._tl_exit_c = 0
            cmd = Twist()
            cmd.linear.x  = self._tl_speed * max(self._speed_scale, 0.0)
            cmd.angular.z = self._tl_sign * self._tl_z
            return cmd
        return None

    def _search_recovery_cmd(self, now):
        """Non-terminal loss recovery: rotate until found.

        Phase 1 (search_timeout_s): creep forward toward the last-seen side at search_speed
        to bring the line back with minimal motion. Phase 2 (until re-acquired): in-place
        rotation sweep — linear=0, rotating search_rotate_z toward the last-seen side and
        reversing every search_sweep_s so it scans both sides instead of spinning past the
        line. Never latches a permanent stop. The normal branch clears _search_until on
        re-acquire, which re-seeds this FSM next loss.
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
        non-line it can never center. During the post-intersection acquisition guard
        the watchdog is suppressed so a settling tracker is not mistaken for a stuck lock.
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

        Decoupled from image callbacks: the wheels keep receiving a steady command even
        when the camera briefly drops frames.

        Stale-frame safety: driving the last command forward at full speed through a blind
        gap runs the robot off a curve. If no new frame has arrived for frame_stale_s, scale
        the published linear speed by stale_speed_scale (steering is held — turning in place
        toward the last-seen line is safer than translating blind). Resumes full speed the
        moment a fresh frame lands.
        """
        cmd = self._latest_cmd

        # Open-loop sign turn: the arc is a timed, camera-independent motion. Bypass the
        # whole control chain so nothing fights it — publish zero down the normal path
        # (topic_cmd → slew_limiter → velocity_controller idles its /cmd_vel to zero) and
        # send the arc Twist straight to /cmd_vel from this 20 Hz timer (no slew, no PI). The
        # stale/blind camera safety is skipped here — the arc runs its full duration.
        if self._so_open_loop and self._so_turn_active:
            self._pub_cmd.publish(Twist())     # kill the chain's output
            self._pub_cmd_direct.publish(cmd)  # open-loop arc direct to /cmd_vel
            return

        blind_for = (time.monotonic() - self._last_frame_t) if self._last_frame_t is not None else 0.0
        if (self._frame_blind_s > 0 and self._last_frame_t is not None
                and blind_for > self._frame_blind_s):
            # Multi-second camera freeze: fully stop so the recovery rotation can't spin the
            # robot blind. Resumes the instant a fresh frame lands.
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

    def _cb_yolo_sign(self, msg: String):
        """Latch the most-recent turn arrow (left/right/straight) from /yolo/turn_sign and
        track its running-max area while in view.

        The arrow is an activation flag: the peak-turn block fires the turn when the area
        peaks then drops. The running-max resets when the arrow has left view (a new
        approach). The optional area floor rejects a far-off speck.
        """
        raw = msg.data.strip()
        c, _, area_s = raw.partition(':')
        try:
            area = float(area_s) if area_s else 0.0
        except ValueError:
            area = 0.0
        # Freshness floor: a too-small (far) arrow doesn't (re)latch a direction.
        if self._turn_sign_min_area > 0.0 and area < self._turn_sign_min_area:
            return
        now = time.monotonic()
        # Reset the running-max if the arrow had left view (a NEW approach starts).
        if (self._turn_sign_last_seen is None
                or (now - self._turn_sign_last_seen) > self._turn_sign_rearm_gap_s):
            self._turn_peak_max = 0.0
            self._turn_peak_reached = False
        # letRecto / "straight" is ignored: treat it as if no sign was seen — don't latch a
        # turn intent, so the commit-turn FSM never fires on it (the robot keeps
        # line-following; the dashed FSM crosses straight on its own).
        if c == self._sign_left:
            self._turn_sign, self._turn_sign_t = 'left', now
        elif c == self._sign_right:
            self._turn_sign, self._turn_sign_t = 'right', now
        else:
            return  # straight / slow signs / none don't change the latched turn intent
        self._turn_sign_area = area
        self._turn_peak_max = max(self._turn_peak_max, area)
        # Latch "got genuinely close" once the running-max crosses the min area. The
        # peak-turn block fires on the subsequent drop OR when the arrow leaves view —
        # both require this flag, so a far-off speck never triggers a turn.
        if self._turn_peak_max >= self._turn_peak_min_area:
            self._turn_peak_reached = True
        self._turn_sign_last_seen = now

    def _fresh_turn_sign(self):
        """Return the latched turn sign if still fresh, else None."""
        if (self._turn_sign is not None and self._turn_sign_t is not None
                and (time.monotonic() - self._turn_sign_t) <= self._sign_stale_s):
            return self._turn_sign
        return None

    def _load_sign_actions(self):
        """Load sign_action_dir/<sign>.csv (rows t,vx,wz) for left/right/straight.

        Mirrors FollowerCore::load_sign_actions (C++). Missing files are skipped.
        Returns {sign: [(t, vx, wz), ...]}.
        """
        import os
        actions = {}
        d = self._sign_action_dir
        if not d:
            self.get_logger().warn('sign_action_dir empty — sign-action replay disabled')
            return actions
        for sign in ('left', 'right', 'straight'):
            path = os.path.join(d, f'{sign}.csv')
            try:
                rows = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line[0] in ('t', '#'):
                            continue
                        a, b, c = line.split(',')[:3]
                        rows.append((float(a), float(b), float(c)))
                if rows:
                    actions[sign] = rows
            except FileNotFoundError:
                continue
            except Exception as e:
                self.get_logger().warn(f'sign action {path}: {e}')
        self.get_logger().info(
            f'sign actions: loaded {sorted(actions)} from {d}')
        return actions


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
