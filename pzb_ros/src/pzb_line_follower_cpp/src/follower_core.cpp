// ROS-free port of line_follower_node.py steering. See header. FAITHFUL to the Python
// _image_cb flow: dashed-suppress -> debounce/latch -> live-break -> dashed phase machine
// (align disabled -> immediate latch -> turn/straight cross -> stop) OR line_lost/error_sat
// /stuck-lock -> search recovery OR the solid PD path. Alignment trig is omitted because
// dashed_align_enabled defaults false (the node then latches aligned immediately).
#include "pzb_line_follower_cpp/follower_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

namespace pzb {
namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
double steady_now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
double median_even(std::deque<double> d) {  // numpy.median (even -> mean of two middle)
  std::vector<double> v(d.begin(), d.end());
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n == 0) return 0.0;
  if (n % 2 == 1) return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double clampd(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }
}  // namespace

FollowerCore::FollowerCore(const FollowerParams& p, std::function<double()> clock_fn)
    : detector(clock_fn ? clock_fn : steady_now), p_(p),
      clock_(clock_fn ? std::move(clock_fn) : steady_now) {}

int FollowerCore::load_sign_actions() {
  // Load sign_action_dir/<sign>.csv (rows: t,vx,wz; header line skipped) for each turn sign.
  sign_actions_.clear();
  if (p_.sign_action_dir.empty()) return 0;
  int n = 0;
  for (const std::string& sign : {std::string("left"), std::string("right"), std::string("straight")}) {
    std::string path = p_.sign_action_dir + "/" + sign + ".csv";
    std::ifstream f(path);
    if (!f.is_open()) continue;
    std::vector<ActionSample> samples;
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == 't' || line[0] == '#') continue;  // header/comment
      std::stringstream ss(line);
      std::string a, b, c;
      if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',') || !std::getline(ss, c, ','))
        continue;
      try {
        samples.push_back({std::stod(a), std::stod(b), std::stod(c)});
      } catch (...) { continue; }
    }
    if (!samples.empty()) { sign_actions_[sign] = std::move(samples); ++n; }
  }
  return n;
}

void FollowerCore::set_yolo_sign(const std::string& cls_in, double now_s) {
  // Accept "<class>" or "<class>:<area_frac>" (the detector appends the bbox area
  // fraction so we can tell when the sign is IN FRONT). Parse the optional suffix.
  std::string cls = cls_in;
  double area = 0.0;
  auto colon = cls_in.find(':');
  if (colon != std::string::npos) {
    cls = cls_in.substr(0, colon);
    try { area = std::stod(cls_in.substr(colon + 1)); } catch (...) { area = 0.0; }
  }
  // Freshness floor: a too-small (far) arrow doesn't (re)latch a direction. 0 = off.
  if (p_.turn_sign_min_area_frac > 0.0 && area < p_.turn_sign_min_area_frac) return;
  // Reset the running-max if the arrow had left view (a NEW approach starts).
  if (std::isnan(turn_sign_last_seen_t_) ||
      (now_s - turn_sign_last_seen_t_) > p_.turn_sign_rearm_gap_s) {
    turn_peak_max_ = 0.0;
    turn_peak_reached_ = false;
  }
  // letRecto / "straight" is ignored: treat it as if no sign was seen — don't latch a turn
  // intent, so the commit-turn FSM never fires on it and the robot keeps line-following
  // (the dashed FSM crosses straight on its own anyway).
  if (cls == p_.turn_sign_left_class)      { turn_sign_ = "left";     turn_sign_t_ = now_s; turn_sign_area_ = area; turn_sign_last_seen_t_ = now_s; }
  else if (cls == p_.turn_sign_right_class){ turn_sign_ = "right";    turn_sign_t_ = now_s; turn_sign_area_ = area; turn_sign_last_seen_t_ = now_s; }
  else return;  // straight / none / other → no-op
  turn_peak_max_ = std::max(turn_peak_max_, area);
  // Latch "got genuinely close" once the running-max crosses the min area (peak block
  // fires on the subsequent drop OR when the arrow leaves view — both require this).
  if (turn_peak_max_ >= p_.turn_peak_min_area) turn_peak_reached_ = true;
}

std::string FollowerCore::fresh_turn_sign(double now) const {
  if (!turn_sign_.empty() && !std::isnan(turn_sign_t_) &&
      (now - turn_sign_t_) <= p_.turn_sign_stale_s)
    return turn_sign_;
  return "";
}

double FollowerCore::approach_speed_clamped() const {
  double cs = approach_speed_;
  if (!(cs > 1e-3)) cs = p_.openloop_speed_mps;
  return std::min(p_.max_linear_speed, std::max(p_.min_linear_speed, cs));
}

bool FollowerCore::is_stuck_lock(double now, double error) {
  stuck_hist_.push_back({now, error});
  while (!stuck_hist_.empty() && (now - stuck_hist_.front().first) > p_.stuck_lock_s)
    stuck_hist_.pop_front();
  if (!std::isnan(acq_until_) && now < acq_until_) return false;
  if (stuck_hist_.size() < 3 || (now - stuck_hist_.front().first) < p_.stuck_lock_s) return false;
  double mn = 1e18, mx = -1e18, min_abs = 1e18;
  for (auto& [t, e] : stuck_hist_) { mn = std::min(mn, e); mx = std::max(mx, e); min_abs = std::min(min_abs, std::abs(e)); }
  if (min_abs <= p_.stuck_lock_band_px) return false;
  if ((mx - mn) >= p_.stuck_lock_var_px) return false;
  return true;
}

double FollowerCore::alignment_cmd(double& tilt_deg, bool& have_estimate) {
  tilt_deg = 0.0; have_estimate = false;
  if (!p_.dashed_align_enabled || slope_buf_.empty()) return 0.0;
  double med_slope = median_even(slope_buf_);
  tilt_deg = std::atan(med_slope / p_.roi_aniso) * 180.0 / M_PI;
  have_estimate = true;
  if (std::abs(tilt_deg) <= p_.align_deadband_deg) return 0.0;
  double az = p_.align_sign * p_.k_align_z * tilt_deg;
  return clampd(az, -p_.align_max_z, p_.align_max_z);
}

Twist2 FollowerCore::search_recovery_cmd(double now) {
  if (std::isnan(search_until_)) {
    if (last_good_error_ > p_.dead_band_px)        search_dir_ = -1.0;
    else if (last_good_error_ < -p_.dead_band_px)  search_dir_ = +1.0;
    else if (recovery_side_ == "right")            search_dir_ = -1.0;
    else if (recovery_side_ == "left")             search_dir_ = +1.0;
    else                                            search_dir_ = 0.0;
    search_until_ = now + p_.search_timeout_s;
    search_sweep_dir_ = (search_dir_ != 0.0) ? search_dir_ : 1.0;
    search_sweep_until_ = search_until_ + p_.search_sweep_s;
  }
  Twist2 cmd;
  if (now < search_until_) {
    cmd.v = std::max(0.0, p_.search_speed_mps);
    cmd.w = search_dir_ * p_.search_angular_z;
  } else {
    if (now >= search_sweep_until_) {
      search_sweep_dir_ = -search_sweep_dir_;
      search_sweep_until_ = now + 2.0 * p_.search_sweep_s;
    }
    cmd.v = 0.0;
    cmd.w = search_sweep_dir_ * p_.search_rotate_z;
  }
  prev_az_ = cmd.w;
  return cmd;
}

Twist2 FollowerCore::process_frame(const cv::Mat& small_bgr, double now_s,
                                   int& cx_out, double& error_out, std::string& line_type_out) {
  // ROI: bottom half -> resize (img_w, img_h//3)
  int rs = small_bgr.rows / 2;
  cv::Mat roi = small_bgr(cv::Range(rs, small_bgr.rows), cv::Range::all());
  int th = p_.image_height / 3, tw = p_.image_width;
  cv::Mat img;
  if (roi.rows != th || roi.cols != tw) cv::resize(roi, img, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
  else img = roi.clone();

  auto [cx, cy] = detector.detect_center_line(img, true);
  (void)cy;
  std::string line_type = detector.line_type;
  double now = now_s;
  int half = p_.image_width / 2;

  // dashed-suppression by last good error
  if (line_type == "dashed" && std::abs(last_good_error_) > p_.dashed_suppress_error_px)
    line_type = "solid";

  // debounce entry + acquisition cap arm
  if (line_type == "dashed") {
    if (dashed_streak_ == 0) acq_until_ = now + p_.acquire_guard_s;
    dashed_streak_ += 1;
  } else {
    dashed_streak_ = 0;
  }
  bool confirmed = dashed_streak_ >= p_.dashed_confirm_frames;
  if (confirmed) {
    dashed_latch_t_ = now;
    if (std::isnan(dashed_first_t_)) {
      dashed_first_t_ = now;
      slope_buf_.clear();
      aligned_latch_ = false;
      align_done_t_ = kNaN;
      cross_turn_start_ = kNaN;
      crossing_active_ = true;       // own control across the gap until a real line returns
      crossing_done_t_ = kNaN;
      real_line_streak_ = 0;
    }
  }
  double dashed_latch_s = p_.dashed_coast_s;
  if (confirmed || (!std::isnan(dashed_latch_t_) && (now - dashed_latch_t_) < dashed_latch_s)) {
    line_type = "dashed";
  } else {
    line_type = "solid";
    dashed_first_t_ = kNaN;
  }

  // live-error abort of a false dashed-on-curve
  if (line_type == "dashed") {
    double live = static_cast<double>(cx - half);
    bool same = (dashed_live_break_sign_ == 0) || ((live > 0) == (dashed_live_break_sign_ > 0));
    if (std::abs(live) > p_.dashed_suppress_error_px && same) {
      dashed_live_break_count_ += 1;
      dashed_live_break_sign_ = live > 0 ? 1 : -1;
    } else {
      dashed_live_break_count_ = 0;
      dashed_live_break_sign_ = 0;
    }
    if (dashed_live_break_count_ >= 2) {
      line_type = "solid";
      dashed_latch_t_ = kNaN; dashed_streak_ = 0; dashed_first_t_ = kNaN;
      aligned_latch_ = false; dashed_live_break_count_ = 0; dashed_live_break_sign_ = 0;
      was_dashed_ = false;
      crossing_active_ = false; real_line_streak_ = 0;  // false dashed-on-curve → release
    }
  } else {
    dashed_live_break_count_ = 0; dashed_live_break_sign_ = 0;
  }

  // Feed the dash-slope median buffer from clean, plausible dash-row readings only.
  if (detector.dash_slope_valid && line_type == "dashed") {
    double tilt = std::atan(detector.dash_slope_px / p_.roi_aniso) * 180.0 / M_PI;
    if (std::abs(tilt) <= p_.align_max_tilt_deg) {
      slope_buf_.push_back(detector.dash_slope_px);
      while ((int)slope_buf_.size() > p_.align_slope_median_n) slope_buf_.pop_front();
    }
  }

  bool line_lost = !(detector.line_flags["left"] || detector.line_flags["center"] ||
                     detector.line_flags["right"]);

  bool sat_now = (line_type == "solid" && std::abs(static_cast<double>(cx - half)) >= p_.error_sat_px);
  if (sat_now) error_sat_count_ += 1; else error_sat_count_ = 0;
  bool error_saturated = error_sat_count_ >= p_.error_sat_frames;

  // dashed->solid edge: re-seed tracker + arm acquisition cap
  if (was_dashed_ && line_type == "solid") {
    detector.reset_tracker_anchors();
    acq_until_ = now + p_.acquire_guard_s;
    error_hist_.clear();
  }
  was_dashed_ = (line_type == "dashed");

  int n_vis = (detector.line_flags["left"] ? 1 : 0) + (detector.line_flags["center"] ? 1 : 0) +
              (detector.line_flags["right"] ? 1 : 0);

  // Crossing-state ownership across the gap: hold control until a real lane returns
  // (>=2 slots for crossing_exit_frames frames), then exit to normal centering.
  if (crossing_active_) {
    real_line_streak_ = (n_vis >= 2) ? real_line_streak_ + 1 : 0;
    if (real_line_streak_ >= p_.crossing_exit_frames) {
      crossing_active_ = false;
      detector.reset_tracker_anchors();
      acq_until_ = now + p_.acquire_guard_s;
      error_hist_.clear();
    }
  }

  Twist2 cmd;
  so_open_loop_ = false;   // default: normal chain; set true only when an arc fires

  // ── Commit-nudge sign turn ──────────────────────────────────────────────────
  // At a dashed crossing with a fresh sign latched: stop+center briefly, then a short slow
  // directional nudge (v=commit_speed, w=±commit_w / 0 for straight) for commit_s, then hand
  // back to normal line-following, which latches the perpendicular branch and completes the
  // turn closed-loop. Published via the normal chain.
  //   commit_phase_: 0 idle | 1 center | 2 forward-enter | 3 arc-nudge (forward while turning)
  if (commit_phase_ == 1) {  // STOP-AND-CENTER: hold still, steer to center the dashed entry
    if ((now - commit_phase_t_) < p_.commit_center_s) {
      double err = static_cast<double>(cx - half);
      cmd.v = 0.0;
      cmd.w = clampd(-p_.Kp_angular * err, -p_.commit_w, p_.commit_w);
      prev_az_ = cmd.w; error_hist_.clear();
      cx_out = cx; error_out = err; line_type_out = line_type;
      return cmd;
    }
    commit_phase_ = 2; commit_phase_t_ = now;   // → forward enter
  }
  if (commit_phase_ == 2) {  // FORWARD ENTER: advance into the intersection before turning
    if ((now - commit_phase_t_) < p_.commit_forward_s) {
      cmd.v = p_.commit_speed * std::max(speed_scale_, 0.0);
      cmd.w = 0.0;
      prev_az_ = cmd.w; error_hist_.clear();
      cx_out = cx; error_out = static_cast<double>(cx - half); line_type_out = line_type;
      return cmd;
    }
    commit_phase_ = 3; commit_phase_t_ = now;   // → arc nudge
  }
  if (commit_phase_ == 3) {  // ARC NUDGE: forward WHILE turning into the chosen branch
    if ((now - commit_phase_t_) < p_.commit_s) {
      cmd.v = p_.commit_speed * std::max(speed_scale_, 0.0);
      cmd.w = (commit_dir_ == "left") ? p_.commit_w
            : (commit_dir_ == "right") ? -p_.commit_w : 0.0;
      prev_az_ = cmd.w; error_hist_.clear();
      cx_out = cx; error_out = static_cast<double>(cx - half); line_type_out = line_type;
      return cmd;
    }
    // done → resume normal line-following (it latches the new branch). Reset trackers + take
    // a short acquisition guard so the first frames don't snap to a crossing dash.
    commit_phase_ = 0; commit_dir_.clear();
    detector.reset_tracker_anchors(); error_hist_.clear();
    acq_until_ = now + p_.acquire_guard_s;
    crossing_active_ = false;
  }
  // TRIGGER: at a dashed crossing + a FRESH turn sign + not in the one-turn cooldown →
  // begin the stop-and-center (phase 1). Consumes the sign (one turn per crossing).
  if (commit_phase_ == 0 && p_.turn_sign_only_enabled) {
    bool at_dashed = (crossing_active_ || line_type == "dashed");
    std::string fs = fresh_turn_sign(now);   // "left"/"right"/"straight"/""
    bool in_cooldown = !std::isnan(last_turn_fire_t_) &&
                       (now - last_turn_fire_t_) < p_.turn_fire_cooldown_s;
    if (at_dashed && !in_cooldown && !fs.empty()) {
      commit_phase_ = 1; commit_phase_t_ = now; commit_dir_ = fs;
      last_turn_fire_t_ = now;
      turn_sign_.clear(); turn_sign_t_ = kNaN;   // consume
      cmd.v = 0.0; cmd.w = 0.0; prev_az_ = 0.0; error_hist_.clear();
      cx_out = cx; error_out = static_cast<double>(cx - half); line_type_out = line_type;
      return cmd;
    }
  }

  // The dashed-crossing FSM (now WITHOUT a turn — the commit-nudge owns turns) still handles
  // a plain straight crossing / crossing-state coast when no sign turn fired.
  if (crossing_active_ || line_type == "dashed") {
    // Phase A — align perpendicular to the dash row, then turn-or-straight cross, then stop.
    double elapsed = !std::isnan(dashed_first_t_) ? (now - dashed_first_t_) : 0.0;
    double cross_speed = approach_speed_clamped();
    double openloop_dur = p_.openloop_dist_m / std::max(cross_speed, 0.01);

    double tilt_deg; bool have_tilt;
    double align_z = alignment_cmd(tilt_deg, have_tilt);
    if (!p_.dashed_align_enabled && !aligned_latch_) { aligned_latch_ = true; align_done_t_ = now; }
    if (!aligned_latch_ && have_tilt && std::abs(tilt_deg) <= p_.align_deadband_deg) {
      aligned_latch_ = true; align_done_t_ = now;
    }
    if (!aligned_latch_ && elapsed >= p_.align_window_s) { aligned_latch_ = true; align_done_t_ = now; }

    // The dashed FSM no longer turns — the commit-nudge block above owns all sign turns.
    // Here we only align → cross straight → coast (a no-sign or straight/letRecto crossing).
    // A turn sign is handled before this block ever runs.
    if (!aligned_latch_) {
      // Phase A — forward at approach speed + perpendicular alignment steering.
      cmd.v = cross_speed * std::max(speed_scale_, 1.0);
      cmd.w = align_z;
    } else if ((now - align_done_t_) < openloop_dur) {
      cmd.v = cross_speed * std::max(speed_scale_, 1.0);
      cmd.w = 0.0;
    } else {
      // Phase C — coast straight through the gap until a real line returns (the exit gate
      // above clears crossing_active_). Replaces the old full stop.
      if (std::isnan(crossing_done_t_)) { crossing_done_t_ = now; turn_sign_.clear(); turn_sign_t_ = kNaN; }
      cmd.v = p_.crossing_coast_speed * std::max(speed_scale_, 1.0);
      cmd.w = 0.0;
    }
    prev_az_ = cmd.w;
    error_hist_.clear();
    cx_out = cx; error_out = static_cast<double>(cx - half); line_type_out = line_type;
    return cmd;
  }

  if (line_lost || error_saturated) {
    cmd = search_recovery_cmd(now);
    cx_out = cx; error_out = static_cast<double>(cx - half); line_type_out = line_type;
    return cmd;
  }

  // stuck-lock watchdog
  if (is_stuck_lock(now, static_cast<double>(cx - half))) {
    cmd = search_recovery_cmd(now);
    cx_out = cx; error_out = static_cast<double>(cx - half); line_type_out = line_type;
    return cmd;
  }

  // ── solid line-follow path ──
  last_valid_t_ = now;
  search_until_ = kNaN;  // re-acquired

  // Reference control: smooth normalized-direction following.
  // Runs the reference center-pick on the same ROI, then the soft-dir control law. The
  // dashed/sign/open-loop paths above already returned, so this only governs solid following.
  if (p_.control_mode == "ref") {
    bool ref_found = false;
    double direction = detector.ref_center_line(img, ref_prev_dir_, ref_found);
    ref_prev_dir_ = ref_found ? direction : 0.0;
    double dt = std::isnan(ref_last_ctrl_t_) ? 0.05 : clampd(now - ref_last_ctrl_t_, 0.015, 0.12);
    double speed_scale = 1.0;
    if (ref_found) {
      ref_raw_dir_ = (1.0 - p_.ref_direction_alpha) * ref_raw_dir_ + p_.ref_direction_alpha * direction;
      double target = ref_raw_dir_;
      if (std::abs(target) < p_.ref_deadband) target = 0.0;
      double max_dd = p_.ref_direction_slew_rate * dt;
      ref_filtered_dir_ = clampd(target, ref_filtered_dir_ - max_dd, ref_filtered_dir_ + max_dd);
      if (std::abs(ref_filtered_dir_) < p_.ref_deadband) ref_filtered_dir_ = 0.0;
    } else {
      ref_filtered_dir_ = 0.0; ref_prev_filtered_dir_ = 0.0;
      speed_scale = p_.ref_lost_speed_scale;
    }
    ref_last_ctrl_t_ = now;
    ref_prev_filtered_dir_ = ref_filtered_dir_;
    double soft = ref_filtered_dir_ * std::pow(std::abs(ref_filtered_dir_), p_.ref_soft_dir_exp);
    double w_t = clampd(-p_.ref_kp * soft, -p_.ref_max_w, p_.ref_max_w);
    double w_sm = (1.0 - p_.ref_omega_alpha) * ref_omega_filtered_ + p_.ref_omega_alpha * w_t;
    double maxd = p_.ref_omega_slew_rate * dt;
    double w = clampd(w_sm, ref_omega_filtered_ - maxd, ref_omega_filtered_ + maxd);
    w = clampd(w, -p_.ref_max_w, p_.ref_max_w);
    ref_omega_filtered_ = w;
    double curve_s = 1.0 - p_.ref_curve_scale_k * std::min(1.0, std::abs(ref_filtered_dir_));
    double ang_s = 1.0 - p_.ref_angular_scale_k * std::min(1.0, std::abs(w) / std::max(1e-6, p_.ref_max_w));
    double v = p_.linear_speed * speed_scale * curve_s * ang_s;
    if (speed_scale_ <= 0.0) v = 0.0; else v *= speed_scale_;
    prev_az_ = w;
    last_good_error_ = direction * half;  // keep curve-lockout / dashed-suppress fed (px-ish)
    cmd.v = v; cmd.w = w;
    cx_out = (int)std::lround((direction + 1.0) * half);
    error_out = direction * half; line_type_out = line_type;
    return cmd;
  }

  double raw_error = static_cast<double>(cx - half);
  error_hist_.push_back(raw_error);
  while ((int)error_hist_.size() > p_.error_median_n) error_hist_.pop_front();
  double error = (p_.error_median_n > 1 && !error_hist_.empty()) ? median_even(error_hist_) : raw_error;
  last_good_error_ = error;

  bool same_sign = true;
  for (double e : error_hist_) if (std::abs(e) > p_.dead_band_px) same_sign &= ((e > 0) == (error > 0));
  bool sharp_turn = (std::abs(error) >= p_.slew_bypass_error_px && same_sign);

  // turn latch
  if (p_.turn_latch_enabled) {
    if (tl_active_) {
      if (std::abs(error) < p_.turn_latch_exit_px) tl_exit_c_++; else tl_exit_c_ = 0;
      bool timed = !std::isnan(tl_start_) && (now - tl_start_) >= p_.turn_latch_max_s;
      if (tl_exit_c_ >= p_.turn_latch_exit_frames || timed) { tl_active_ = false; tl_arm_ = 0; prev_az_ = tl_sign_ * p_.turn_latch_z; }
      else {
        cmd.v = p_.turn_latch_speed * std::max(speed_scale_, 0.0); cmd.w = tl_sign_ * p_.turn_latch_z; prev_az_ = cmd.w;
        cx_out = cx; error_out = error; line_type_out = line_type; return cmd;
      }
    } else {
      if (std::abs(error) >= p_.turn_latch_error_px && same_sign) tl_arm_++; else tl_arm_ = 0;
      if (tl_arm_ >= p_.turn_latch_frames) {
        tl_active_ = true; tl_sign_ = error > 0 ? -1.0 : 1.0; tl_start_ = now; tl_exit_c_ = 0;
        cmd.v = p_.turn_latch_speed * std::max(speed_scale_, 0.0); cmd.w = tl_sign_ * p_.turn_latch_z; prev_az_ = cmd.w;
        cx_out = cx; error_out = error; line_type_out = line_type; return cmd;
      }
    }
  }

  double d_error = 0.0;
  if (!std::isnan(prev_error_t_) && p_.Kd_angular != 0.0) {
    double dt = now - prev_error_t_;
    d_error = dt > 0 ? (error - prev_error_) / dt : 0.0;
  }
  prev_error_ = error; prev_error_t_ = now;

  bool sharp = std::abs(error) > p_.sharp_turn_threshold_px;
  bool in_approach;
  if (sharp) {
    if (std::isnan(turn_first_seen_t_)) turn_first_seen_t_ = now;
    in_approach = (now - turn_first_seen_t_) < p_.turn_approach_delay_s;
  } else { turn_first_seen_t_ = kNaN; in_approach = false; }
  if (!std::isnan(acq_until_) && now < acq_until_) in_approach = true;

  double angular_z;
  if (std::abs(error) <= p_.dead_band_px) { angular_z = 0.0; steer_side_ = 0.0; }
  else {
    double kp = p_.Kp_angular * (sharp_turn ? p_.curve_gain : 1.0);
    angular_z = clampd(-kp * error - p_.Kd_angular * d_error, -p_.max_angular, p_.max_angular);
    if (in_approach && !sharp_turn) {
      double cap = p_.max_angular * 0.3;
      angular_z = clampd(angular_z, -cap, cap);
    }
    double want_side = angular_z < 0 ? -1.0 : 1.0;
    if (steer_side_ != 0.0 && want_side != steer_side_ && std::abs(error) <= p_.steer_hysteresis_px)
      angular_z = 0.0;
    else steer_side_ = want_side;
  }

  if (sharp_turn) {
    // emit full PD value; downstream slew node ramps it
  } else if (p_.angular_slew_max > 0) {
    double delta = clampd(angular_z - prev_az_, -p_.angular_slew_max, p_.angular_slew_max);
    angular_z = prev_az_ + delta;
  }
  prev_az_ = angular_z;

  // speed: curve-coupled + error-driven brake + floors
  double ang_frac = p_.max_angular > 0 ? std::abs(angular_z) / p_.max_angular : 0.0;
  double ss_curve = 1.0 - p_.curve_speed_reduction * ang_frac;
  double span = std::max(1.0, p_.curve_brake_error_px - p_.dead_band_px);
  double err_frac = clampd((std::abs(error) - p_.dead_band_px) / span, 0.0, 1.0);
  double ss_err = 1.0 - p_.curve_speed_reduction * err_frac;
  double ss = std::min(ss_curve, ss_err);
  double min_frac = p_.linear_speed > 0 ? p_.min_linear_speed / p_.linear_speed : 0.0;
  double linear_x = std::min(p_.max_linear_speed, p_.linear_speed * std::max(min_frac, ss));
  if (n_vis == 2) linear_x = std::min(linear_x, p_.linear_speed * 0.6);
  else if (n_vis <= 1) linear_x = std::max(p_.sharp_turn_speed, p_.curve_min_speed);
  if (std::abs(error) > p_.sharp_turn_threshold_px) linear_x = std::max(p_.sharp_turn_speed, p_.curve_min_speed);
  if (std::abs(error) > p_.tight_turn_error_px) linear_x = p_.sharp_turn_speed;
  if (speed_scale_ <= 0.0) linear_x = 0.0; else linear_x *= speed_scale_;

  if (std::abs(error) <= p_.sharp_turn_threshold_px && n_vis >= 2 && linear_x > p_.min_linear_speed)
    approach_speed_ = 0.7 * approach_speed_ + 0.3 * linear_x;

  cmd.v = linear_x; cmd.w = angular_z;
  cx_out = cx; error_out = error; line_type_out = line_type;
  return cmd;
}

}  // namespace pzb
