// C++ port of center_line_detector.py — see header for the parity contract.
#include "pzb_line_follower_cpp/center_line_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace pzb {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double steady_now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Median of a deque. Odd N → middle order-statistic; even N → mean of the two middle
// (matches numpy.median). Caller int-casts where Python does.
double median_deque(const std::deque<double>& d) {
  std::vector<double> v(d.begin(), d.end());
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n == 0) return 0.0;
  if (n % 2 == 1) return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void push_maxlen(std::deque<double>& d, double val, size_t maxlen) {
  d.push_back(val);
  while (d.size() > maxlen) d.pop_front();
}

// Degree-1 least squares: returns slope (and intercept via out param). double accumulators.
double polyfit1(const std::vector<double>& x, const std::vector<double>& y, double& b) {
  double n = static_cast<double>(x.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
  }
  double denom = n * sxx - sx * sx;
  double slope = (denom != 0.0) ? (n * sxy - sx * sy) / denom : 0.0;
  b = (sy - slope * sx) / n;
  return slope;
}
}  // namespace

CenterLineDetector::CenterLineDetector(std::function<double()> clock_fn)
    : clock_(clock_fn ? std::move(clock_fn) : steady_now) {
  for (const auto& k : {"left", "center", "right"}) line_history_[k] = {};
}

bool CenterLineDetector::has_pos(const std::string& k) const {
  auto it = line_positions.find(k);
  return it != line_positions.end() && !std::isnan(it->second);
}

void CenterLineDetector::reset_tracker_anchors() {
  for (const auto& k : {"left", "center", "right"}) {
    line_positions[k] = kNaN;
    line_history_[k].clear();
    line_lost_frames_[k] = 0;
    line_flags[k] = false;
  }
  history_.clear();
  prev_cx = static_cast<double>(CAMERA_WIDTH / 2);
  prev_center = kNaN;
  halfwidth_hist_.clear();
  centering = false;
}

// ── persistent adaptive threshold ────────────────────────────────────────────
cv::Mat CenterLineDetector::adaptive_threshold(const cv::Mat& gray) {
  int T = T_state_;
  double area = static_cast<double>(gray.total());
  int direction = 0;
  cv::Mat binary;
  for (int i = 0; i < 10; ++i) {
    cv::threshold(gray, binary, T, 255, cv::THRESH_BINARY_INV);
    double perc = 100.0 * cv::countNonZero(binary) / area;
    if (perc > ADAPTIVE_DARK_MAX) {
      if (T <= ADAPTIVE_T_MIN || direction == 1) break;
      T = std::max(ADAPTIVE_T_MIN, T - 10);
      direction = -1;
    } else if (perc < ADAPTIVE_DARK_MIN) {
      if (T >= ADAPTIVE_T_MAX || direction == -1) break;
      T = std::min(ADAPTIVE_T_MAX, T + 10);
      direction = 1;
    } else {
      break;
    }
  }
  T_state_ = T;
  cv::threshold(gray, binary, T, 255, cv::THRESH_BINARY_INV);
  return binary;
}

// ── contour helpers ──────────────────────────────────────────────────────────
int CenterLineDetector::count_track_contours(const cv::Mat& binary, int roi_h) {
  cv::Mat k3 = cv::Mat::ones(3, 3, CV_8U);
  cv::Mat k5 = cv::Mat::ones(5, 5, CV_8U);
  cv::Mat opened, cleaned;
  cv::morphologyEx(binary, opened, cv::MORPH_OPEN, k3);
  cv::morphologyEx(opened, cleaned, cv::MORPH_CLOSE, k5);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  int cnt = 0;
  for (const auto& v : valid_contours(contours)) {
    if (v.cy >= roi_h * 0.15) ++cnt;
  }
  return cnt;
}

std::vector<ValidContour> CenterLineDetector::valid_contours(
    const std::vector<std::vector<cv::Point>>& contours) {
  std::vector<ValidContour> valid;
  for (const auto& cnt : contours) {
    double area = cv::contourArea(cnt);
    if (area < MIN_AREA || area > MAX_AREA) continue;
    if (area < SEAM_MAX_AREA) {
      cv::Rect r = cv::boundingRect(cnt);
      double extent = (r.width * r.height > 0)
                          ? area / static_cast<double>(r.width * r.height)
                          : 1.0;
      std::vector<cv::Point> hull;
      cv::convexHull(cnt, hull);
      double hull_a = cv::contourArea(hull);
      double solidity = (hull_a > 0) ? area / hull_a : 1.0;
      if (extent < SEAM_MAX_EXTENT && solidity < SEAM_MAX_SOLIDITY) continue;
    }
    cv::Moments M = cv::moments(cnt);
    if (M.m00 == 0) continue;
    valid.push_back({M.m10 / M.m00, M.m01 / M.m00, area, cnt});
  }
  std::sort(valid.begin(), valid.end(),
            [](const ValidContour& a, const ValidContour& b) { return a.cx < b.cx; });
  return valid;
}

// ── dash candidates / classification ─────────────────────────────────────────
void CenterLineDetector::dash_candidates(const std::vector<ValidContour>& valid,
                                         std::vector<ValidContour>& out, bool& has_solid) {
  out.clear();
  has_solid = false;
  for (const auto& v : valid) {
    cv::Rect r = cv::boundingRect(v.cnt);
    bool is_tall = r.height >= DASH_MAX_HEIGHT;
    bool is_stub = r.width > 0 && r.height >= r.width;
    bool is_large = v.area >= DASH_MAX_AREA;
    if (is_tall && is_large) has_solid = true;
    if (!is_tall && !is_stub) out.push_back(v);
  }
}

std::string CenterLineDetector::classify_line(const std::vector<ValidContour>& valid, int w) {
  if (static_cast<int>(valid.size()) < DASH_MIN_COUNT) return "solid";

  std::vector<ValidContour> cand;
  bool has_solid;
  dash_candidates(valid, cand, has_solid);
  if (has_solid) return "solid";
  if (static_cast<int>(cand.size()) < DASH_MIN_COUNT) return "solid";

  std::vector<double> xs, ys;
  for (const auto& v : cand) { xs.push_back(v.cx); ys.push_back(v.cy); }
  double span = *std::max_element(xs.begin(), xs.end()) -
                *std::min_element(xs.begin(), xs.end());
  if (span < DASH_MIN_SPAN * w) return "solid";

  double center_lo = w * DASH_CENTER_ZONE;
  double center_hi = w * (1.0 - DASH_CENTER_ZONE);
  bool any_center = false;
  for (double x : xs) if (x >= center_lo && x <= center_hi) { any_center = true; break; }
  if (!any_center) return "solid";

  double b;
  double slope = polyfit1(xs, ys, b);
  double sum_sq = 0, mean = 0;
  std::vector<double> resid(xs.size());
  for (size_t i = 0; i < xs.size(); ++i) { resid[i] = ys[i] - (slope * xs[i] + b); mean += resid[i]; }
  mean /= resid.size();
  for (double r : resid) sum_sq += (r - mean) * (r - mean);
  double resid_std = std::sqrt(sum_sq / resid.size());   // population std (ddof=0)
  if (resid_std > DASH_ROW_BAND_PX) return "solid";

  return "dashed";
}

// ── dashed fusion ────────────────────────────────────────────────────────────
std::pair<int, int> CenterLineDetector::fuse_dashes(const std::vector<ValidContour>& valid,
                                                    int y_start, int h) {
  if (valid.empty()) {
    dash_slope_px = 0.0;
    dash_slope_valid = false;
    int cx = !std::isnan(prev_cx) ? static_cast<int>(std::lround(prev_cx)) : CAMERA_WIDTH / 2;
    return {cx, y_start + (h - y_start) / 2};
  }
  double total = 0, fcx = 0, fcy = 0;
  for (const auto& v : valid) { total += v.area; }
  if (total == 0) total = 1.0;
  for (const auto& v : valid) { fcx += v.cx * v.area; fcy += v.cy * v.area; }
  fcx /= total; fcy /= total;

  std::vector<ValidContour> cand;
  bool has_solid;
  dash_candidates(valid, cand, has_solid);
  dash_slope_px = 0.0;
  dash_slope_valid = false;
  if (static_cast<int>(cand.size()) >= DASH_SLOPE_MIN_N) {
    std::vector<double> xs, ys;
    for (const auto& v : cand) { xs.push_back(v.cx); ys.push_back(v.cy); }
    double x_span = *std::max_element(xs.begin(), xs.end()) -
                    *std::min_element(xs.begin(), xs.end());
    if (x_span >= DASH_SLOPE_MIN_SPAN * CAMERA_WIDTH) {
      double b;
      dash_slope_px = polyfit1(xs, ys, b);
      dash_slope_valid = true;
    }
  }
  return {static_cast<int>(std::lround(fcx)), y_start + static_cast<int>(std::lround(fcy))};
}

// ── min-cost L/C/R assignment ────────────────────────────────────────────────
std::map<std::string, double> CenterLineDetector::best_assignment(
    const std::vector<double>& xs, int w) {
  const std::array<std::string, 3> names{"left", "center", "right"};
  std::map<std::string, double> defs{{"left", 0.0},
                                     {"center", w * 0.5},
                                     {"right", static_cast<double>(w - 1)}};
  std::map<std::string, double> prevs;
  for (const auto& k : names)
    prevs[k] = has_pos(k) ? line_positions[k] : defs[k];

  int n = static_cast<int>(xs.size());
  std::map<std::string, double> best{{"left", kNaN}, {"center", kNaN}, {"right", kNaN}};
  if (n == 0) return best;

  double best_cost = std::numeric_limits<double>::infinity();
  const double ms = TRACK_MIN_SEP;

  if (n >= 3) {
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j)
        for (int k = j + 1; k < n; ++k) {
          if (xs[j] - xs[i] < ms || xs[k] - xs[j] < ms) continue;
          double cost = std::abs(xs[i] - prevs["left"]) +
                        std::abs(xs[j] - prevs["center"]) +
                        std::abs(xs[k] - prevs["right"]);
          if (cost < best_cost) {
            best_cost = cost;
            best = {{"left", xs[i]}, {"center", xs[j]}, {"right", xs[k]}};
          }
        }
    if (std::isinf(best_cost)) {  // relax min-sep
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
          for (int k = j + 1; k < n; ++k) {
            double cost = std::abs(xs[i] - prevs["left"]) +
                          std::abs(xs[j] - prevs["center"]) +
                          std::abs(xs[k] - prevs["right"]);
            if (cost < best_cost) {
              best_cost = cost;
              best = {{"left", xs[i]}, {"center", xs[j]}, {"right", xs[k]}};
            }
          }
    }
  } else if (n == 2) {
    const std::array<std::pair<std::string, std::string>, 3> pairs{
        {{"left", "center"}, {"left", "right"}, {"center", "right"}}};
    for (const auto& [sa, sb] : pairs) {
      if (xs[1] - xs[0] < ms) {
        if (!(sa == "left" && sb == "right")) continue;
      }
      double cost = std::abs(xs[0] - prevs[sa]) + std::abs(xs[1] - prevs[sb]);
      if (cost < best_cost) {
        best_cost = cost;
        std::string sc;
        for (const auto& s : names) if (s != sa && s != sb) sc = s;
        best = {{"left", kNaN}, {"center", kNaN}, {"right", kNaN}};
        best[sa] = xs[0]; best[sb] = xs[1]; best[sc] = kNaN;
      }
    }
    if (std::isinf(best_cost)) {
      std::string nearest = names[0];
      double bd = std::abs(xs[0] - prevs[nearest]);
      for (const auto& s : names) {
        double d = std::abs(xs[0] - prevs[s]);
        if (d < bd) { bd = d; nearest = s; }
      }
      best = {{"left", kNaN}, {"center", kNaN}, {"right", kNaN}};
      best[nearest] = xs[0];
    }
  } else {  // n == 1
    std::string nearest = names[0];
    double bd = std::abs(xs[0] - prevs[nearest]);
    for (const auto& s : names) {
      double d = std::abs(xs[0] - prevs[s]);
      if (d < bd) { bd = d; nearest = s; }
    }
    best = {{"left", kNaN}, {"center", kNaN}, {"right", kNaN}};
    best[nearest] = xs[0];
  }
  return best;
}

// ── three-line tracker + lane-center ─────────────────────────────────────────
std::pair<int, int> CenterLineDetector::track_three_lines(
    const std::vector<ValidContour>& valid, int w, int h, int y_start, double max_jump) {
  std::vector<double> xs;
  for (const auto& v : valid) xs.push_back(v.cx);
  std::sort(xs.begin(), xs.end());

  auto detections = best_assignment(xs, w);

  // velocity gate (CLAMP, don't discard) + median filter
  for (const auto& name : {"left", "center", "right"}) {
    double new_cx = detections[name];
    if (std::isnan(new_cx)) { line_flags[name] = false; continue; }
    double prev = line_positions[name];
    if (!std::isnan(prev) && std::abs(new_cx - prev) > max_jump)
      new_cx = prev + (new_cx > prev ? max_jump : -max_jump);
    push_maxlen(line_history_[name], new_cx, MEDIAN_K);
    double smoothed;
    if (static_cast<int>(line_history_[name].size()) < MEDIAN_K)
      smoothed = std::round(new_cx);
    else
      smoothed = static_cast<double>(static_cast<long>(median_deque(line_history_[name])));
    line_positions[name] = smoothed;
    line_flags[name] = true;
  }

  // stale anchor reset
  for (const auto& name : {"left", "center", "right"}) {
    if (line_flags[name]) {
      line_lost_frames_[name] = 0;
    } else {
      line_lost_frames_[name] += 1;
      if (line_lost_frames_[name] >= STALE_THRESH) {
        line_positions[name] = kNaN;
        line_history_[name].clear();
        line_lost_frames_[name] = 0;
      }
    }
  }

  // ordering: drop only the less-trusted side of a bad pair
  const std::array<std::pair<std::string, std::string>, 2> ord{
      {{"left", "center"}, {"center", "right"}}};
  for (const auto& [a, b] : ord) {
    if (has_pos(a) && has_pos(b) && line_positions[a] >= line_positions[b] - LINE_MIN_SEP) {
      std::string drop = (line_history_[a].size() <= line_history_[b].size()) ? a : b;
      line_flags[drop] = false;
      break;
    }
  }

  // lane center = midpoint of the two boundaries; single-boundary part-way bias
  bool L_ok = line_flags["left"] && has_pos("left");
  bool R_ok = line_flags["right"] && has_pos("right");
  double L = line_positions["left"], C = line_positions["center"], R = line_positions["right"];

  double prev_c = !std::isnan(prev_center)
                      ? prev_center
                      : (!std::isnan(prev_cx) ? prev_cx : w / 2.0);

  double cx_center;
  if (L_ok && R_ok) {
    cx_center = (L + R) / 2.0;
    push_maxlen(halfwidth_hist_, (R - L) / 2.0, LANE_HALFWIDTH_MEDIAN_N);
    centering = true;
  } else if (L_ok || R_ok) {
    double X = L_ok ? L : R;
    double hw = !halfwidth_hist_.empty() ? median_deque(halfwidth_hist_) : w * 0.4;
    double side = (X < prev_c) ? 1.0 : -1.0;
    double target = X + side * hw;
    cx_center = prev_c + SINGLE_LINE_PULL * (target - prev_c);
    centering = false;
  } else if (!std::isnan(C) && line_flags["center"]) {
    cx_center = C;
    centering = false;
  } else {
    cx_center = prev_c;
    centering = false;
  }

  // light one-frame guard against a big slot-swap jump
  if (!std::isnan(prev_center) && std::abs(cx_center - prev_center) > w * 0.25)
    cx_center = 0.5 * (cx_center + prev_center);

  prev_center = cx_center;
  line_positions["center"] = cx_center;

  int cy = y_start + (h - y_start) / 2;
  int cxi = std::max(0, std::min(w - 1, static_cast<int>(std::lround(cx_center))));
  return {cxi, cy};
}

// ── main entry ───────────────────────────────────────────────────────────────
std::pair<int, int> CenterLineDetector::detect_center_line(const cv::Mat& image,
                                                           bool pre_cropped) {
  int h = image.rows, w = image.cols;
  cv::Mat roi;
  int y_start;
  if (pre_cropped) {
    roi = image;
    y_start = 0;
  } else {
    y_start = (2 * h) / 3;
    roi = image(cv::Range(y_start, h), cv::Range::all());
  }

  cv::Mat gray, blurred, binary;
  cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);
  binary = adaptive_threshold(blurred);
  int roi_h = roi.rows;

  // white-boundary column mask
  cv::Mat roi_hsv;
  cv::cvtColor(roi, roi_hsv, cv::COLOR_BGR2HSV);
  cv::Mat white_mask;
  cv::inRange(roi_hsv, cv::Scalar(0, 0, 190), cv::Scalar(179, 40, 255), white_mask);
  int rw = roi.cols;
  std::vector<int> boundary_cols;
  for (int c = 0; c < rw; ++c) {
    int cnt = cv::countNonZero(white_mask.col(c));
    if (cnt >= WHITE_BOUNDARY_THRESH) boundary_cols.push_back(c);
  }
  if (!boundary_cols.empty()) {
    int mid = rw / 2;
    int l_lim = static_cast<int>(rw * WHITE_MASK_EDGE_FRAC);
    int r_lim = static_cast<int>(rw * (1.0 - WHITE_MASK_EDGE_FRAC));
    std::vector<int> right, left;
    for (int c : boundary_cols) { if (c >= mid) right.push_back(c); else left.push_back(c); }
    cv::Mat masked = binary.clone();
    if (!right.empty()) {
      int r0 = std::max(right.front(), r_lim);
      if (r0 < rw) masked(cv::Range::all(), cv::Range(r0, rw)).setTo(0);
    }
    if (!left.empty()) {
      int l1 = std::min(left.back() + 1, l_lim);
      if (l1 > 0) masked(cv::Range::all(), cv::Range(0, l1)).setTo(0);
    }
    if (count_track_contours(masked, roi_h) > 0) binary = masked;
  }

  cv::Mat k3 = cv::Mat::ones(3, 3, CV_8U);
  cv::Mat k5 = cv::Mat::ones(5, 5, CV_8U);
  cv::Mat opened, cleaned;
  cv::morphologyEx(binary, opened, cv::MORPH_OPEN, k3);
  cv::morphologyEx(opened, cleaned, cv::MORPH_CLOSE, k5);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<ValidContour> valid;
  for (auto& v : valid_contours(contours))
    if (v.cy >= roi_h * 0.15) valid.push_back(std::move(v));

  line_type = classify_line(valid, w);

  double now = clock_();
  double dt = !std::isnan(last_detect_t_) ? (now - last_detect_t_) : (1.0 / 30.0);
  last_detect_t_ = now;
  double max_jump = LINE_MAX_JUMP_PS * dt;

  std::pair<int, int> result;
  if (line_type == "dashed") {
    result = fuse_dashes(valid, y_start, roi_h);
  } else {
    result = track_three_lines(valid, w, roi_h, y_start, max_jump);
  }

  // LANE_BLEND_THR == 9999 → brown lane-interior blend disabled (skipped, as in Python).

  push_maxlen(history_, static_cast<double>(result.first), MEDIAN_K);
  prev_cx = static_cast<double>(static_cast<long>(median_deque(history_)));

  return result;
}

}  // namespace pzb
