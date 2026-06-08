// C++ port of center_line_detector.py (CenterLineDetector).
// FAITHFUL port — constants, algorithm order, and int/float types mirror the Python so
// the offline rosbag comparison matches. See the Python source for the rationale comments.
#pragma once

#include <array>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace pzb {

// One detected/tracked line contour: centroid + area + the contour itself.
struct ValidContour {
  double cx;
  double cy;
  double area;
  std::vector<cv::Point> cnt;
};

class CenterLineDetector {
 public:
  // ── constants (mirror the Python class attributes) ────────────────────────
  static constexpr double MIN_AREA         = 50.0;
  static constexpr double MAX_AREA         = 4000.0;
  static constexpr double SIGNIFICANT_AREA = 500.0;

  static constexpr double SEAM_MAX_AREA     = 500.0;
  static constexpr double SEAM_MAX_EXTENT   = 0.32;
  static constexpr double SEAM_MAX_SOLIDITY = 0.45;
  static constexpr int    MEDIAN_K          = 3;

  static constexpr double LINE_MAX_JUMP_PS = 2100.0;
  static constexpr int    LINE_MIN_SEP     = 15;
  static constexpr int    TRACK_MIN_SEP    = 40;
  static constexpr int    STALE_THRESH     = 15;

  static constexpr int    LANE_HALFWIDTH_MEDIAN_N = 7;
  static constexpr double SINGLE_LINE_PULL        = 0.5;

  static constexpr int    DASH_MIN_COUNT   = 3;
  static constexpr double DASH_MAX_AREA    = 3500.0;
  static constexpr int    DASH_MAX_HEIGHT  = 35;
  static constexpr double DASH_MIN_SPAN    = 0.20;
  static constexpr double DASH_CENTER_ZONE = 0.25;
  static constexpr double DASH_ROW_BAND_PX = 12.0;

  static constexpr int    DASH_SLOPE_MIN_N    = 3;
  static constexpr double DASH_SLOPE_MIN_SPAN = 0.25;

  static constexpr int    LANE_MIN_PIX   = 800;
  static constexpr int    LANE_BLEND_THR = 9999;  // disabled

  static constexpr int    WHITE_BOUNDARY_THRESH = 20;
  static constexpr double WHITE_MASK_EDGE_FRAC  = 0.30;

  static constexpr int    ADAPTIVE_T_INIT   = 185;
  static constexpr int    ADAPTIVE_T_MIN    = 100;
  static constexpr int    ADAPTIVE_T_MAX    = 220;
  static constexpr double ADAPTIVE_DARK_MIN = 2.0;
  static constexpr double ADAPTIVE_DARK_MAX = 15.0;

  static constexpr int    CAMERA_WIDTH  = 320;
  static constexpr int    CAMERA_HEIGHT = 240;

  // clock_fn returns seconds (monotonic). Default = steady_clock. Replay injects bag time.
  explicit CenterLineDetector(std::function<double()> clock_fn = {});

  // Main entry. `image` is the (pre-cropped) ROI when pre_cropped=true (matches Python
  // node usage). Returns (cx, cy). Updates the public attributes below.
  std::pair<int, int> detect_center_line(const cv::Mat& image, bool pre_cropped = true);

  void reset_tracker_anchors();

  // ── public attributes the node reads (mirror Python) ──────────────────────
  std::string line_type = "solid";
  std::map<std::string, double> line_positions{{"left", NAN}, {"center", NAN}, {"right", NAN}};
  std::map<std::string, bool>   line_flags{{"left", false}, {"center", false}, {"right", false}};
  bool   has_pos(const std::string& k) const;   // true if line_positions[k] is set (not NaN)
  double dash_slope_px = 0.0;
  bool   dash_slope_valid = false;
  bool   centering = false;
  double prev_cx = NAN;          // NaN = None
  double prev_center = NAN;      // NaN = None

 private:
  std::function<double()> clock_;
  double last_detect_t_ = NAN;   // NaN = None (first frame)
  int    T_state_ = ADAPTIVE_T_INIT;

  std::deque<double> history_;                              // global cx history (maxlen 3)
  std::map<std::string, std::deque<double>> line_history_;  // per-line (maxlen 3)
  std::map<std::string, int> line_lost_frames_{{"left", 0}, {"center", 0}, {"right", 0}};
  std::deque<double> halfwidth_hist_;                       // (R-L)/2 (maxlen 7)

  // pipeline helpers (names mirror Python)
  cv::Mat adaptive_threshold(const cv::Mat& gray);
  int count_track_contours(const cv::Mat& binary, int roi_h);
  std::vector<ValidContour> valid_contours(const std::vector<std::vector<cv::Point>>& contours);
  void dash_candidates(const std::vector<ValidContour>& valid,
                       std::vector<ValidContour>& out, bool& has_solid);
  std::string classify_line(const std::vector<ValidContour>& valid, int w);
  std::pair<int, int> fuse_dashes(const std::vector<ValidContour>& valid, int y_start, int h);
  std::map<std::string, double> best_assignment(const std::vector<double>& xs, int w);
  std::pair<int, int> track_three_lines(const std::vector<ValidContour>& valid,
                                        int w, int h, int y_start, double max_jump);
};

}  // namespace pzb
