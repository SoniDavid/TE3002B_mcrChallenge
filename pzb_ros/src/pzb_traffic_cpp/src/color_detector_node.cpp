// C++ port of color_detector_node.py — HSV traffic-light detector. Drop-in: same node
// name (color_detector_node), same params (reuses traffic_params.yaml), same topics
// (/camera/image_raw -> /traffic_light_color). FAITHFUL: crop top 2/3 -> resize (320,160)
// -> GaussianBlur -> BGR2HSV -> CLAHE on V -> per-color inRange OR-masks -> MORPH_OPEN x2
// -> largest round/bright/aspect-valid blob -> winner above min_area -> N-frame confirm.
// Every-3rd-frame skip preserved.
#include <map>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

class ColorDetectorNode : public rclcpp::Node {
 public:
  ColorDetectorNode() : rclcpp::Node("color_detector_node") {
    auto gi = [&](const std::string& n, int d) { return this->declare_parameter(n, d); };
    red_h_low1_ = gi("red_h_low1", 0);   red_h_high1_ = gi("red_h_high1", 10);
    red_h_low2_ = gi("red_h_low2", 165); red_h_high2_ = gi("red_h_high2", 180);
    red_s_min_  = gi("red_s_min", 100);  red_v_min_   = gi("red_v_min", 80);
    green_h_low_ = gi("green_h_low", 45); green_h_high_ = gi("green_h_high", 85);
    green_s_min_ = gi("green_s_min", 80); green_v_min_  = gi("green_v_min", 80);
    yellow_h_low_ = gi("yellow_h_low", 18); yellow_h_high_ = gi("yellow_h_high", 35);
    yellow_s_min_ = gi("yellow_s_min", 120); yellow_v_min_ = gi("yellow_v_min", 100);
    min_area_  = this->declare_parameter("min_blob_area", 800.0);
    confirm_n_ = this->declare_parameter("confirm_frames", 3);
    std::string in_topic = this->declare_parameter("input_topic", std::string("/camera/image_raw"));

    morph_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));

    auto best_effort = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    sub_ = create_subscription<sensor_msgs::msg::Image>(
        in_topic, best_effort, [this](sensor_msgs::msg::Image::SharedPtr m) { on_image(m); });
    pub_ = create_publisher<std_msgs::msg::String>("/traffic_light_color", 10);
    RCLCPP_INFO(get_logger(), "C++ color_detector_node ready (min_area=%.0f, confirm=%d)",
                min_area_, confirm_n_);
  }

 private:
  // OR all (lower,upper) HSV ranges then MORPH_OPEN x2 (matches _build_mask).
  cv::Mat build_mask(const cv::Mat& hsv,
                     const std::vector<std::pair<cv::Scalar, cv::Scalar>>& ranges) {
    cv::Mat combined;
    for (auto& [lo, hi] : ranges) {
      cv::Mat m; cv::inRange(hsv, lo, hi, m);
      if (combined.empty()) combined = m; else cv::bitwise_or(combined, m, combined);
    }
    cv::Mat opened;
    cv::morphologyEx(combined, opened, cv::MORPH_OPEN, morph_kernel_, cv::Point(-1, -1), 2);
    return opened;
  }

  // Largest round/bright-aspect blob area above min_area (matches _largest_blob_area:
  // area>=min, circularity>=0.35, aspect in (0.5,2.0)).
  double largest_blob_area(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double best = 0.0;
    for (auto& c : contours) {
      double area = cv::contourArea(c);
      if (area < min_area_) continue;
      double perim = cv::arcLength(c, true);
      if (perim == 0) continue;
      double circ = 4.0 * M_PI * area / (perim * perim);
      if (circ < 0.35) continue;
      cv::Rect r = cv::boundingRect(c);
      double aspect = r.height > 0 ? static_cast<double>(r.width) / r.height : 0.0;
      if (!(aspect > 0.5 && aspect < 2.0)) continue;
      if (area > best) best = area;
    }
    return best;
  }

  void on_image(sensor_msgs::msg::Image::SharedPtr msg) {
    frame_skip_ = (frame_skip_ + 1) % 3;
    if (frame_skip_ != 0) return;

    cv::Mat full;
    try { full = cv_bridge::toCvShare(msg, "bgr8")->image; }
    catch (const std::exception& e) { RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what()); return; }

    cv::Mat top = full(cv::Range(0, full.rows * 2 / 3), cv::Range::all());
    cv::Mat frame; cv::resize(top, frame, cv::Size(320, 160), 0, 0, cv::INTER_AREA);
    cv::Mat blurred; cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
    cv::Mat hsv; cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch; cv::split(hsv, ch);
    clahe_->apply(ch[2], ch[2]);
    cv::merge(ch, hsv);

    std::map<std::string, std::vector<std::pair<cv::Scalar, cv::Scalar>>> ranges{
      {"red", {{cv::Scalar(red_h_low1_, red_s_min_, red_v_min_), cv::Scalar(red_h_high1_, 255, 255)},
               {cv::Scalar(red_h_low2_, red_s_min_, red_v_min_), cv::Scalar(red_h_high2_, 255, 255)}}},
      {"green", {{cv::Scalar(green_h_low_, green_s_min_, green_v_min_), cv::Scalar(green_h_high_, 255, 255)}}},
      {"yellow", {{cv::Scalar(yellow_h_low_, yellow_s_min_, yellow_v_min_), cv::Scalar(yellow_h_high_, 255, 255)}}},
    };

    std::map<std::string, double> areas;
    for (auto& [name, r] : ranges) areas[name] = largest_blob_area(build_mask(hsv, r));

    std::string best_color = "none";
    double best_area = min_area_;
    // iterate in a fixed order matching Python dict insertion (red, green, yellow)
    for (const std::string& name : {"red", "green", "yellow"})
      if (areas[name] > best_area) { best_area = areas[name]; best_color = name; }

    if (best_color == candidate_) candidate_count_++;
    else { candidate_ = best_color; candidate_count_ = 1; }
    if (candidate_count_ >= confirm_n_) confirmed_color_ = candidate_;

    std_msgs::msg::String m; m.data = confirmed_color_; pub_->publish(m);
  }

  int red_h_low1_, red_h_high1_, red_h_low2_, red_h_high2_, red_s_min_, red_v_min_;
  int green_h_low_, green_h_high_, green_s_min_, green_v_min_;
  int yellow_h_low_, yellow_h_high_, yellow_s_min_, yellow_v_min_;
  double min_area_; int confirm_n_;
  int frame_skip_ = 0, candidate_count_ = 0;
  std::string candidate_ = "none", confirmed_color_ = "none";
  cv::Mat morph_kernel_;
  cv::Ptr<cv::CLAHE> clahe_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ColorDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
