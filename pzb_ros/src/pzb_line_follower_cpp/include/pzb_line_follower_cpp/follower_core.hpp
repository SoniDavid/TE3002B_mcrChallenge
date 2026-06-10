// ROS-free core of line_follower_node.py: parameters + the full steering state machine.
// Drives the C++ CenterLineDetector and returns a (linear.x, angular.z) command per frame.
// Used by both the rclcpp node (line_follower_node.cpp) and the offline comparator
// (replay_follower_main.cpp). FAITHFUL port — same params, order, thresholds.
#pragma once

#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "pzb_line_follower_cpp/center_line_detector.hpp"

namespace pzb {

struct Twist2 { double v = 0.0; double w = 0.0; };

// All node parameters (defaults mirror line_follower_node.py declare_parameter values +
// the YAML, which the ROS node overrides at construction).
struct FollowerParams {
  int    image_width = 320, image_height = 240;
  double Kp_angular = 0.0045, Kd_angular = 0.0;
  int    dead_band_px = 12;
  double linear_speed = 0.06, max_linear_speed = 0.20, max_angular = 0.8;
  double curve_speed_reduction = 0.5, min_linear_speed = 0.05;
  int    sharp_turn_threshold_px = 60;
  double sharp_turn_speed = 0.03;
  double lost_timeout_s = 0.35, lost_speed_scale = 0.50;
  int    dashed_confirm_frames = 3;
  double dashed_coast_s = 1.5;
  double openloop_speed_mps = 0.15, openloop_dist_m = 0.50;
  double recovery_angular_z = 0.30;
  double turn_approach_delay_s = 0.15, acquire_guard_s = 0.4;
  double stuck_lock_s = 2.5; double stuck_lock_band_px = 20, stuck_lock_var_px = 4;
  bool   dashed_recovery_enabled = false;
  double max_error_jump_px = 60;
  int    error_median_n = 2;
  double search_timeout_s = 1.2, search_speed_mps = 0.04, search_angular_z = 0.25;
  double search_rotate_z = 0.30, search_sweep_s = 2.0, frame_blind_s = 1.0;
  double curve_min_speed = 0.05, angular_slew_max = 0.20;
  double slew_bypass_error_px = 40, angular_slew_max_sharp = 0.45;
  double curve_gain = 1.5;
  // B2 turn latch (opt-in)
  bool   turn_latch_enabled = false;
  double turn_latch_error_px = 55; int turn_latch_frames = 3;
  double turn_latch_z = 0.55, turn_latch_exit_px = 25; int turn_latch_exit_frames = 3;
  double turn_latch_max_s = 2.5, turn_latch_speed = 0.05;
  double dashed_suppress_error_px = 45, tight_turn_error_px = 110, curve_brake_error_px = 90;
  double frame_stale_s = 0.4, stale_speed_scale = 0.0, steer_hysteresis_px = 6;
  double error_sat_px = 150; int error_sat_frames = 6;
  // Perpendicular dashed-alignment: rotate to square up to the dash row before crossing.
  bool   dashed_align_enabled = true;
  double align_deadband_deg = 6.0, k_align_z = 0.015, align_max_z = 0.30, align_sign = 1.0;
  int    align_slope_median_n = 5;
  double align_window_s = 2.0, align_max_tilt_deg = 35.0;
  double roi_aniso = 0.889;   // ROI anisotropic-resize factor (dy/dx) for tilt recovery
  // YOLO turn signs
  std::string turn_sign_left_class = "letIzquierda", turn_sign_right_class = "letDerecha",
              turn_sign_straight_class = "letRecto";
  double turn_sign_stale_s = 4.0, cross_turn_z = 0.6, cross_turn_s = 2.9, cross_turn_speed = 0.06;
  // Sign turn — turn at the arrow's CLOSEST point (area peak-then-drop). The arrow (own
  // /yolo/turn_sign channel) is an ACTIVATION FLAG: latch a direction + track running-max
  // area. FIRE when running-max >= turn_peak_min_area (genuinely close) AND current area
  // < turn_peak_drop_frac × max (past closest). Validated on dashed_turn_v1/v2 (fires at
  // the area peak, not early). The dashed-crossing FSM is the FALLBACK. turn_sign_min_area_frac
  // is a tiny freshness floor (0=off), NOT the trigger.
  bool   turn_sign_only_enabled = true;
  double turn_sign_min_area_frac = 0.0;     // freshness floor only (0=off)
  // Arrow gone this long → reset running-max + re-arm (a new approach).
  double turn_sign_rearm_gap_s = 4.0;
  // After a turn fires, block any new sign turn for this long (one sign = one turn).
  double turn_fire_cooldown_s = 8.0;
  double turn_peak_min_area = 0.03;         // arrow must get this close before a turn can fire
  double turn_peak_drop_frac = 0.7;         // fire when current area < this × running-max
  // Curve lockout: don't let a sign arc preempt a real corner on the solid line — hold the
  // latch while |error| is past curve_lockout_error_px on solid for curve_lockout_frames
  // consecutive frames; fire only after the corner settles back near center.
  double curve_lockout_error_px = 70.0; int curve_lockout_frames = 3;
  // Crossing-state gap handling: coast through the inter-intersection gap until a real
  // line (>=2 slots for crossing_exit_frames frames) returns, instead of stopping.
  double crossing_coast_speed = 0.06; int crossing_exit_frames = 3;

  // ── miniretoS8 reference line-follow control (ROUND 8) ────────────────────────
  // control_mode: "ref" = use the reference detector center-pick + soft-dir control law
  // for solid-line following (smooth, no slot-swap thrash; validated offline); "pd" = the
  // original cx-PD path. The dashed-FSM / sign-turn / open-loop paths are unchanged either
  // way. The reference normalizes the line offset to `direction` ∈ [-1,1]:
  //   raw EMA(α) → direction slew(±rate·dt) → soft = dir·|dir|^exp → ω = -kp·soft (clamp
  //   ±max_w) → ω EMA + ω slew → v = linear·(1-cs_k·|dir|)·(1-as_k·|ω|/max_w).
  std::string control_mode = "ref";
  double ref_kp = 1.5, ref_max_w = 2.0;
  double ref_direction_alpha = 0.35, ref_direction_slew_rate = 10.0;
  double ref_omega_alpha = 1.0, ref_omega_slew_rate = 10.0;
  double ref_soft_dir_exp = 0.75;
  double ref_curve_scale_k = 0.75, ref_angular_scale_k = 0.45;
  double ref_lost_speed_scale = 0.65, ref_deadband = 0.0;

  // ── Teach-by-demonstration sign actions (ROUND 9) ─────────────────────────────
  // When enabled, a fresh turn sign latched AT a dashed crossing triggers an OPEN-LOOP
  // replay of a recorded /cmd_vel sequence (one CSV per sign: left/right/straight) instead
  // of the synthetic cross_turn arc — the maneuver the user demonstrated. After the
  // sequence ends, hand off to the Phase-C coast-reacquire. sign_action_dir holds the CSVs
  // (t,vx,wz rows). If disabled or no CSV for the latched sign, the synthetic arc / dashed
  // FSM is used (backward compatible).
  bool   sign_action_enabled = true;
  std::string sign_action_dir = "";   // set by the node from the package share dir

  // ── COMMIT turn FSM (ROUND 9.2b): center → forward(enter) → ARC nudge → resume follow ──
  // At a dashed crossing with a fresh sign: stop+center (commit_center_s), drive forward to
  // enter the intersection (commit_forward_s), then a short ARC nudge (v=commit_speed,
  // w=±commit_w / 0 straight, for commit_s; swept ≈ commit_w×commit_s) into the branch, then
  // resume normal line-following which completes the corner closed-loop.
  //   Tune: commit_forward_s = enter distance; commit_w×commit_s = nudge angle (raise the
  //   product to turn more; lower commit_w + raise commit_s for slower-same-angle).
  double commit_speed = 0.04, commit_w = 0.30, commit_s = 2.9, commit_center_s = 0.4;
  double commit_forward_s = 3.0;
};

class FollowerCore {
 public:
  explicit FollowerCore(const FollowerParams& p, std::function<double()> clock_fn = {});

  // Process one camera frame (already the bgr8 small image). Runs ROI crop+resize +
  // detector + the full steering decision. Returns the command the 20 Hz timer would
  // publish (here returned directly per frame for the offline comparator). Also exposes
  // cx/error/line_type via the out-params for the cx/error/line_type publishers.
  Twist2 process_frame(const cv::Mat& small_bgr, double now_s,
                       int& cx_out, double& error_out, std::string& line_type_out);

  // external inputs
  void set_speed_scale(double s) { speed_scale_ = s; }
  void set_yolo_sign(const std::string& cls, double now_s);

  // Load per-sign demonstrated actions from CSVs (sign_action_dir/<sign>.csv, rows t,vx,wz)
  // for sign ∈ {left,right,straight}. Missing files are simply skipped. Returns #loaded.
  int load_sign_actions();

  // True while the last process_frame returned an OPEN-LOOP sign-turn arc command — the
  // node routes it DIRECTLY to /cmd_vel (zero down the normal chain) so the slew_limiter +
  // velocity_controller PI can't fight the timed arc. Matches the Python node's _so_open_loop.
  bool open_loop() const { return so_open_loop_; }

  CenterLineDetector detector;

 private:
  FollowerParams p_;
  std::function<double()> clock_;
  double speed_scale_ = 1.0;

  // ── steering / smoothing state ──
  std::deque<double> error_hist_;
  double prev_az_ = 0.0, steer_side_ = 0.0;
  double prev_error_ = 0.0, prev_error_t_ = NAN, turn_first_seen_t_ = NAN;
  double last_good_error_ = 0.0, last_valid_t_ = NAN;
  double approach_speed_ = 0.0;

  // ── B2 turn latch ──
  bool tl_active_ = false; double tl_sign_ = 0.0, tl_start_ = NAN; int tl_arm_ = 0, tl_exit_c_ = 0;

  // ── dashed crossing ──
  int dashed_streak_ = 0; double dashed_latch_t_ = NAN, dashed_first_t_ = NAN;
  bool aligned_latch_ = false; double align_done_t_ = NAN; double cross_turn_start_ = NAN;
  std::deque<double> slope_buf_;   // dash-slope samples for alignment median
  bool was_dashed_ = false; double acq_until_ = NAN;
  int dashed_live_break_count_ = 0, dashed_live_break_sign_ = 0;
  bool crossing_active_ = false; double crossing_done_t_ = NAN; int real_line_streak_ = 0;

  // ── YOLO turn arrow (peak-area turn) ──
  double turn_sign_area_ = 0.0;            // latest bbox area fraction of the latched arrow
  double turn_peak_max_ = 0.0;             // running max area while the arrow is in view
  double turn_sign_last_seen_t_ = NAN;     // last time a turn arrow was in view
  bool turn_peak_reached_ = false;         // running-max crossed turn_peak_min_area this approach
  int curve_lockout_streak_ = 0;           // consecutive cornering-on-solid frames (sign lockout)
  double last_turn_fire_t_ = NAN;          // time the last sign turn fired (post-fire cooldown)
  // peak-turn open-loop arc state
  bool so_turn_active_ = false;
  double so_turn_dir_ = 0.0;
  double so_turn_start_ = NAN;
  bool so_armed_ = true;
  bool so_open_loop_ = false;              // last frame returned an open-loop arc command

  // ── teach-by-demonstration replay (ROUND 9 — dormant, superseded by commit-nudge) ──
  struct ActionSample { double t, vx, wz; };
  std::map<std::string, std::vector<ActionSample>> sign_actions_;  // "left"/"right"/"straight"
  bool action_replay_active_ = false;
  double action_replay_start_ = NAN;       // monotonic time the replay began
  std::string action_replay_sign_;         // which sign is replaying

  // ── commit-nudge turn FSM (ROUND 9.2) ──
  int commit_phase_ = 0;                   // 0 idle | 1 centering | 2 committing
  double commit_phase_t_ = NAN;            // time the current phase began
  std::string commit_dir_;                 // "left"/"right"/"straight"

  // ── miniretoS8 reference control-law state (ROUND 8) ──
  double ref_raw_dir_ = 0.0, ref_filtered_dir_ = 0.0, ref_prev_filtered_dir_ = 0.0;
  double ref_omega_filtered_ = 0.0, ref_prev_dir_ = 0.0;
  double ref_last_ctrl_t_ = NAN;

  // ── recovery / guards ──
  double search_until_ = NAN, search_dir_ = 0.0, search_sweep_dir_ = 1.0, search_sweep_until_ = NAN;
  std::string recovery_side_;
  int error_sat_count_ = 0;
  std::deque<std::pair<double,double>> stuck_hist_;

  // ── YOLO turn sign latch ──
  std::string turn_sign_;          // "left"|"right"|"straight"|""
  double turn_sign_t_ = NAN;

  double approach_speed_clamped() const;
  std::string fresh_turn_sign(double now) const;
  // alignment: returns align_z; sets tilt_deg + have_estimate.
  double alignment_cmd(double& tilt_deg, bool& have_estimate);
  bool is_stuck_lock(double now, double error);
  Twist2 search_recovery_cmd(double now);
};

}  // namespace pzb
