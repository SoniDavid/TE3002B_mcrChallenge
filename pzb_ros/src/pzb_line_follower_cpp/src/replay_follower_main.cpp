// Phase-B validation: drive the C++ FollowerCore over a bag's /camera/image_small with
// injected bag-timestamp dt, feeding /traffic_speed_scale and /yolo/sign from the same bag,
// and compare per-frame cx/error/line_type to the RECORDED /line_follower/* (the Python
// node's on-robot output). Prints a per-frame CSV and a summary (mean/p90/max |dcx|,
// line_type mismatch). Join recorded values to each image frame by nearest-preceding stamp.
//
// Usage: replay_follower <bag> [--csv out.csv]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "pzb_line_follower_cpp/bag_reader.hpp"
#include "pzb_line_follower_cpp/follower_core.hpp"

namespace {
// For a sorted-by-ts vector, find the value at the latest entry with ts <= t (or first).
template <typename T>
const T* latest_at(const std::vector<T>& v, int64_t t, size_t& cursor) {
  while (cursor + 1 < v.size() && v[cursor + 1].timestamp_ns <= t) ++cursor;
  if (v.empty()) return nullptr;
  return &v[cursor];
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: replay_follower <bag> [--csv F]\n"; return 1; }
  std::string bag = argv[1], csv;
  for (int i = 2; i < argc; ++i)
    if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];

  std::string db3 = pzb::resolve_db3(bag);
  auto imgs = pzb::read_images(db3, "/camera/image_small");
  if (imgs.empty()) { std::cerr << "no /camera/image_small\n"; return 1; }
  auto rec_cx   = pzb::read_int32(db3, "/line_follower/cx");
  auto rec_err  = pzb::read_float32(db3, "/line_follower/error");
  auto rec_type = pzb::read_string(db3, "/line_follower/line_type");
  auto scale    = pzb::read_float32(db3, "/traffic_speed_scale");
  auto sign     = pzb::read_string(db3, "/yolo/sign");

  double cur_t = 0.0;
  pzb::FollowerParams p;  // defaults match the YAML-tuned values
  pzb::FollowerCore core(p, [&cur_t]() { return cur_t; });

  std::ostream* out = &std::cout;
  std::ofstream f;
  if (!csv.empty()) { f.open(csv); out = &f; }
  *out << "t,cx_cpp,cx_rec,dcx,err_cpp,err_rec,ltype_cpp,ltype_rec,match\n";

  int64_t t0 = imgs.front().timestamp_ns;
  size_t c_cx = 0, c_err = 0, c_type = 0, c_scale = 0, c_sign = 0;
  std::vector<double> dcx_all, derr_all;
  int type_mismatch = 0, compared = 0;

  for (const auto& m : imgs) {
    cur_t = (m.timestamp_ns - t0) / 1e9;
    // feed external inputs as of this frame's time
    if (const auto* s = latest_at(scale, m.timestamp_ns, c_scale)) core.set_speed_scale(s->value);
    if (const auto* sg = latest_at(sign, m.timestamp_ns, c_sign)) core.set_yolo_sign(sg->value, cur_t);

    int cx; double err; std::string lt;
    core.process_frame(m.bgr, cur_t, cx, err, lt);

    const auto* rc = latest_at(rec_cx, m.timestamp_ns, c_cx);
    const auto* re = latest_at(rec_err, m.timestamp_ns, c_err);
    const auto* rt = latest_at(rec_type, m.timestamp_ns, c_type);
    int rcx = rc ? (int)rc->value : -1;
    double rerr = re ? re->value : NAN;
    std::string rtype = rt ? rt->value : "";

    int dcx = (rc ? std::abs(cx - rcx) : -1);
    bool tmatch = (rt ? lt == rtype : true);
    if (rc) { dcx_all.push_back(dcx); compared++; if (!tmatch) type_mismatch++; }
    if (re) derr_all.push_back(std::abs(err - rerr));

    char buf[160];
    std::snprintf(buf, sizeof(buf), "%.6f,%d,%d,%d,%.1f,%.1f,%s,%s,%d\n",
                  cur_t, cx, rcx, dcx, err, rerr, lt.c_str(), rtype.c_str(), tmatch ? 1 : 0);
    *out << buf;
  }

  auto stats = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double mean = v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    auto pct = [&](double q) { return v.empty() ? 0.0 : v[std::min(v.size()-1, (size_t)(q*v.size()))]; };
    return std::tuple<double,double,double>(mean, pct(0.90), v.empty()?0:v.back());
  };
  auto [mcx, p90cx, mxcx] = stats(dcx_all);
  auto [merr, p90err, mxerr] = stats(derr_all);
  std::cerr << "\n=== " << bag << " : C++ FollowerCore vs recorded /line_follower/* ===\n";
  std::cerr << "frames=" << imgs.size() << " compared=" << compared << "\n";
  std::fprintf(stderr, "|dcx|:   mean=%.2f  p90=%.0f  max=%.0f\n", mcx, p90cx, mxcx);
  std::fprintf(stderr, "|derr|:  mean=%.2f  p90=%.0f  max=%.0f\n", merr, p90err, mxerr);
  std::fprintf(stderr, "line_type mismatch: %d (%.2f%%)\n", type_mismatch,
               compared ? 100.0 * type_mismatch / compared : 0.0);
  return 0;
}
