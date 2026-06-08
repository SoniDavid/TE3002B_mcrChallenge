// rclcpp wrapper around FollowerCore. Drop-in for the Python line_follower_node:
// same node name, topics, params. Uses the image header stamp as the clock so dt matches
// a recorded run when replayed.
#include <chrono>
#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

#include "pzb_line_follower_cpp/follower_core.hpp"

using std::placeholders::_1;

class LineFollowerNode : public rclcpp::Node {
 public:
  LineFollowerNode() : rclcpp::Node("line_follower_node") {
    pzb::FollowerParams p;
    auto gd = [&](const std::string& n, auto def) { return this->declare_parameter(n, def); };
    p.image_width = gd("image_width", p.image_width);
    p.image_height = gd("image_height", p.image_height);
    p.Kp_angular = gd("Kp_angular", p.Kp_angular);
    p.Kd_angular = gd("Kd_angular", p.Kd_angular);
    p.dead_band_px = gd("dead_band_px", p.dead_band_px);
    p.linear_speed = gd("linear_speed", p.linear_speed);
    p.max_linear_speed = gd("max_linear_speed", p.max_linear_speed);
    p.max_angular = gd("max_angular", p.max_angular);
    p.curve_speed_reduction = gd("curve_speed_reduction", p.curve_speed_reduction);
    p.min_linear_speed = gd("min_linear_speed", p.min_linear_speed);
    p.sharp_turn_threshold_px = gd("sharp_turn_threshold_px", p.sharp_turn_threshold_px);
    p.sharp_turn_speed = gd("sharp_turn_speed", p.sharp_turn_speed);
    p.lost_timeout_s = gd("lost_timeout_s", p.lost_timeout_s);
    p.lost_speed_scale = gd("lost_speed_scale", p.lost_speed_scale);
    p.dashed_confirm_frames = gd("dashed_confirm_frames", p.dashed_confirm_frames);
    p.dashed_coast_s = gd("dashed_coast_s", p.dashed_coast_s);
    p.openloop_speed_mps = gd("openloop_speed_mps", p.openloop_speed_mps);
    p.openloop_dist_m = gd("openloop_dist_m", p.openloop_dist_m);
    p.recovery_angular_z = gd("recovery_angular_z", p.recovery_angular_z);
    p.turn_approach_delay_s = gd("turn_approach_delay_s", p.turn_approach_delay_s);
    p.acquire_guard_s = gd("acquire_guard_s", p.acquire_guard_s);
    p.stuck_lock_s = gd("stuck_lock_s", p.stuck_lock_s);
    p.stuck_lock_band_px = gd("stuck_lock_band_px", p.stuck_lock_band_px);
    p.stuck_lock_var_px = gd("stuck_lock_var_px", p.stuck_lock_var_px);
    p.dashed_recovery_enabled = gd("dashed_recovery_enabled", p.dashed_recovery_enabled);
    p.max_error_jump_px = gd("max_error_jump_px", p.max_error_jump_px);
    p.error_median_n = gd("error_median_n", p.error_median_n);
    p.search_timeout_s = gd("search_timeout_s", p.search_timeout_s);
    p.search_speed_mps = gd("search_speed_mps", p.search_speed_mps);
    p.search_angular_z = gd("search_angular_z", p.search_angular_z);
    p.search_rotate_z = gd("search_rotate_z", p.search_rotate_z);
    p.search_sweep_s = gd("search_sweep_s", p.search_sweep_s);
    p.frame_blind_s = gd("frame_blind_s", p.frame_blind_s);
    p.curve_min_speed = gd("curve_min_speed", p.curve_min_speed);
    p.angular_slew_max = gd("angular_slew_max", p.angular_slew_max);
    p.slew_bypass_error_px = gd("slew_bypass_error_px", p.slew_bypass_error_px);
    p.angular_slew_max_sharp = gd("angular_slew_max_sharp", p.angular_slew_max_sharp);
    p.curve_gain = gd("curve_gain", p.curve_gain);
    p.turn_latch_enabled = gd("turn_latch_enabled", p.turn_latch_enabled);
    p.turn_latch_error_px = gd("turn_latch_error_px", p.turn_latch_error_px);
    p.turn_latch_frames = gd("turn_latch_frames", p.turn_latch_frames);
    p.turn_latch_z = gd("turn_latch_z", p.turn_latch_z);
    p.turn_latch_exit_px = gd("turn_latch_exit_px", p.turn_latch_exit_px);
    p.turn_latch_exit_frames = gd("turn_latch_exit_frames", p.turn_latch_exit_frames);
    p.turn_latch_max_s = gd("turn_latch_max_s", p.turn_latch_max_s);
    p.turn_latch_speed = gd("turn_latch_speed", p.turn_latch_speed);
    p.dashed_suppress_error_px = gd("dashed_suppress_error_px", p.dashed_suppress_error_px);
    p.tight_turn_error_px = gd("tight_turn_error_px", p.tight_turn_error_px);
    p.curve_brake_error_px = gd("curve_brake_error_px", p.curve_brake_error_px);
    p.frame_stale_s = gd("frame_stale_s", p.frame_stale_s);
    p.stale_speed_scale = gd("stale_speed_scale", p.stale_speed_scale);
    p.steer_hysteresis_px = gd("steer_hysteresis_px", p.steer_hysteresis_px);
    p.error_sat_px = gd("error_sat_px", p.error_sat_px);
    p.error_sat_frames = gd("error_sat_frames", p.error_sat_frames);
    p.dashed_align_enabled = gd("dashed_align_enabled", p.dashed_align_enabled);
    p.align_deadband_deg = gd("align_deadband_deg", p.align_deadband_deg);
    p.k_align_z = gd("k_align_z", p.k_align_z);
    p.align_max_z = gd("align_max_z", p.align_max_z);
    p.align_sign = gd("align_sign", p.align_sign);
    p.align_slope_median_n = gd("align_slope_median_n", p.align_slope_median_n);
    p.align_window_s = gd("align_window_s", p.align_window_s);
    p.align_max_tilt_deg = gd("align_max_tilt_deg", p.align_max_tilt_deg);
    p.turn_sign_left_class = gd("turn_sign_left_class", p.turn_sign_left_class);
    p.turn_sign_right_class = gd("turn_sign_right_class", p.turn_sign_right_class);
    p.turn_sign_straight_class = gd("turn_sign_straight_class", p.turn_sign_straight_class);
    p.turn_sign_stale_s = gd("turn_sign_stale_s", p.turn_sign_stale_s);
    p.cross_turn_z = gd("cross_turn_z", p.cross_turn_z);
    p.cross_turn_s = gd("cross_turn_s", p.cross_turn_s);
    p.cross_turn_speed = gd("cross_turn_speed", p.cross_turn_speed);
    p.crossing_coast_speed = gd("crossing_coast_speed", p.crossing_coast_speed);
    p.crossing_exit_frames = gd("crossing_exit_frames", p.crossing_exit_frames);
    std::string topic_in = gd("topic_image_in", std::string("/camera/image_small"));
    std::string topic_cmd = gd("topic_cmd_vel", std::string("/cmd_vel_desired_raw"));

    // Clock: use the image header stamp for dt parity; fall back to steady_clock.
    core_ = std::make_unique<pzb::FollowerCore>(p, [this]() { return cur_t_; });

    auto best_effort = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    auto reliable = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    pub_cx_ = create_publisher<std_msgs::msg::Int32>("/line_follower/cx", reliable);
    pub_err_ = create_publisher<std_msgs::msg::Float32>("/line_follower/error", reliable);
    pub_type_ = create_publisher<std_msgs::msg::String>("/line_follower/line_type", reliable);
    pub_cmd_ = create_publisher<geometry_msgs::msg::Twist>(topic_cmd, reliable);

    sub_img_ = create_subscription<sensor_msgs::msg::Image>(
        topic_in, best_effort, std::bind(&LineFollowerNode::on_image, this, _1));
    sub_scale_ = create_subscription<std_msgs::msg::Float32>(
        "/traffic_speed_scale", reliable,
        [this](std_msgs::msg::Float32::SharedPtr m) { core_->set_speed_scale(m->data); });
    sub_sign_ = create_subscription<std_msgs::msg::String>(
        "/yolo/sign", rclcpp::QoS(10),
        [this](std_msgs::msg::String::SharedPtr m) { core_->set_yolo_sign(m->data, now_s()); });

    timer_ = create_wall_timer(std::chrono::milliseconds(50),
                               std::bind(&LineFollowerNode::on_timer, this));
    p_ = p;
    RCLCPP_INFO(get_logger(), "C++ line_follower_node ready (img %dx%d, Kp=%.4f, curve_gain=%.2f)",
                p.image_width, p.image_height, p.Kp_angular, p.curve_gain);
  }

 private:
  double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void on_image(sensor_msgs::msg::Image::SharedPtr msg) {
    last_frame_t_ = now_s();
    // header-stamp clock for dt parity
    cur_t_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    cv::Mat full;
    try {
      full = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
      return;
    }
    int cx; double err; std::string lt;
    pzb::Twist2 cmd = core_->process_frame(full, cur_t_, cx, err, lt);
    latest_v_ = cmd.v; latest_w_ = cmd.w;

    std_msgs::msg::Int32 cm; cm.data = cx; pub_cx_->publish(cm);
    std_msgs::msg::Float32 em; em.data = static_cast<float>(err); pub_err_->publish(em);
    std_msgs::msg::String tm; tm.data = lt; pub_type_->publish(tm);
  }

  void on_timer() {
    geometry_msgs::msg::Twist cmd;
    double blind_for = (last_frame_t_ > 0) ? (now_s() - last_frame_t_) : 0.0;
    if (p_.frame_blind_s > 0 && last_frame_t_ > 0 && blind_for > p_.frame_blind_s) {
      // multi-second freeze → full stop
    } else if (p_.frame_stale_s > 0 && last_frame_t_ > 0 && blind_for > p_.frame_stale_s) {
      cmd.linear.x = latest_v_ * p_.stale_speed_scale;
      cmd.angular.z = latest_w_;
    } else {
      cmd.linear.x = latest_v_;
      cmd.angular.z = latest_w_;
    }
    pub_cmd_->publish(cmd);
  }

  pzb::FollowerParams p_;
  std::unique_ptr<pzb::FollowerCore> core_;
  double cur_t_ = 0.0, last_frame_t_ = 0.0, latest_v_ = 0.0, latest_w_ = 0.0;

  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_cx_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_err_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_type_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_scale_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_sign_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LineFollowerNode>());
  rclcpp::shutdown();
  return 0;
}
