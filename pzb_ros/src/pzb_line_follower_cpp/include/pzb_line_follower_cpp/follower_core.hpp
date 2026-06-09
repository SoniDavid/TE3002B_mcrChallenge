// ROS-free core of line_follower_node.py: parameters + the full steering state machine.
// Drives the C++ CenterLineDetector and returns a (linear.x, angular.z) command per frame.
// Used by both the rclcpp node (line_follower_node.cpp) and the offline comparator
// (replay_follower_main.cpp). FAITHFUL port — same params, order, thresholds.
#pragma once

#include <deque>
#include <functional>
#include <string>

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
  // Sign turn (no dashed cue needed): the turn ARROW (on its OWN /yolo/turn_sign channel,
  // so a nearer non-turn sign can't mask it) LATCHES a direction; the open-loop arc FIRES
  // when the arrow's bbox area fraction >= turn_sign_min_area_frac (the arrow is close /
  // robot is AT the intersection). The 5 slowww bags proved a well-driven turn keeps
  // |error| LOW so an error gate can't fire, and that the arrow is masked on /yolo/sign —
  // hence the separate channel + area trigger. One arrow = one turn (re-arm after the
  // arrow leaves view for turn_sign_rearm_gap_s).
  bool   turn_sign_only_enabled = true;
  double turn_sign_min_area_frac = 0.06;    // turn-arrow bbox/frame at which to fire (tuned on slowww)
  // After a turn fires, do NOT turn again until the arrow has LEFT view for this long —
  // one physical arrow = one turn, even though it stays visible for seconds.
  double turn_sign_rearm_gap_s = 1.0;
  // Crossing-state gap handling: coast through the inter-intersection gap until a real
  // line (>=2 slots for crossing_exit_frames frames) returns, instead of stopping.
  double crossing_coast_speed = 0.06; int crossing_exit_frames = 3;
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

  // ── YOLO turn arrow (acted on at the dashed crossing) ──
  double turn_sign_area_ = 0.0;            // latched bbox area fraction of the last turn arrow
  double turn_sign_last_seen_t_ = NAN;     // last time a turn arrow was in view

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
