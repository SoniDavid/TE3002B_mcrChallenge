// Minimal ROS-free rosbag2 (sqlite3) reader + CDR sensor_msgs/Image / Int32 / Float32 /
// String parsing — mirrors scripts/bag_to_video.py:parse_image so the offline C++ tools
// read the same frames as the Python harness.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace pzb {

struct ImageMsg {
  int64_t timestamp_ns;  // bag receive timestamp
  cv::Mat bgr;           // decoded bgr8 frame
};

struct ScalarMsg {
  int64_t timestamp_ns;
  double  value;         // Int32/Float32 value
};

struct StringMsg {
  int64_t timestamp_ns;
  std::string value;
};

// Resolve the *_0.db3 (or first .db3) inside a bag directory (or pass a .db3 path through).
std::string resolve_db3(const std::string& path);

// Read all messages of a topic. Returns timestamp-ordered vectors. Empty if topic absent.
std::vector<ImageMsg>  read_images(const std::string& db3, const std::string& topic);
std::vector<ScalarMsg> read_int32(const std::string& db3, const std::string& topic);
std::vector<ScalarMsg> read_float32(const std::string& db3, const std::string& topic);
std::vector<StringMsg> read_string(const std::string& db3, const std::string& topic);

// Parse a CDR sensor_msgs/Image blob into a bgr8 cv::Mat. Throws std::runtime_error on
// unsupported encoding. (rgb8 is converted to bgr.)
cv::Mat parse_image(const std::vector<uint8_t>& data);

}  // namespace pzb
