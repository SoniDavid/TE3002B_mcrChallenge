// Phase-A parity tool: run the C++ CenterLineDetector over a bag's /camera/image_small
// with INJECTED bag-timestamp dt (matching scripts/replay_follower.py) and dump per-frame
// cx/cy/line_type/n_vis as CSV. Diff against the Python detector dump to verify parity.
//
// Usage: detector_dump <bag_dir_or_db3> [--topic /camera/image_small] [--csv out.csv]
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "pzb_line_follower_cpp/bag_reader.hpp"
#include "pzb_line_follower_cpp/center_line_detector.hpp"

// Mirror line_follower_node._image_cb ROI: bottom half -> resize (img_w, img_h//3)=(320,80).
static cv::Mat crop_resize_like_node(const cv::Mat& full, int img_w = 320, int img_h = 240) {
  int rs = full.rows / 2;
  cv::Mat roi = full(cv::Range(rs, full.rows), cv::Range::all());
  int th = img_h / 3, tw = img_w;
  if (roi.rows != th || roi.cols != tw) {
    cv::Mat out;
    cv::resize(roi, out, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
    return out;
  }
  return roi.clone();
}

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: detector_dump <bag> [--topic T] [--csv F]\n"; return 1; }
  std::string bag = argv[1];
  std::string topic = "/camera/image_small";
  std::string csv;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--topic") && i + 1 < argc) topic = argv[++i];
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
  }

  std::string db3 = pzb::resolve_db3(bag);
  auto imgs = pzb::read_images(db3, topic);
  if (imgs.empty()) { std::cerr << "no images on " << topic << "\n"; return 1; }

  // Injected clock: returns the current frame's bag time (seconds since first frame).
  double cur_t = 0.0;
  pzb::CenterLineDetector det([&cur_t]() { return cur_t; });

  std::ostream* out = &std::cout;
  std::ofstream f;
  if (!csv.empty()) { f.open(csv); out = &f; }
  *out << "t,cx,cy,line_type,n_vis,prev_cx\n";

  int64_t t0 = imgs.front().timestamp_ns;
  for (const auto& m : imgs) {
    cur_t = (m.timestamp_ns - t0) / 1e9;
    cv::Mat roi = crop_resize_like_node(m.bgr);
    auto [cx, cy] = det.detect_center_line(roi, /*pre_cropped=*/true);
    int nvis = (det.line_flags["left"] ? 1 : 0) + (det.line_flags["center"] ? 1 : 0) +
               (det.line_flags["right"] ? 1 : 0);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%.6f,%d,%d,%s,%d,%.1f\n", cur_t, cx, cy,
                  det.line_type.c_str(), nvis, det.prev_cx);
    *out << buf;
  }
  std::cerr << "frames: " << imgs.size() << "\n";
  return 0;
}
