#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class TwistSlewLimiter(Node):
    """Limit linear/angular slew rate for Twist commands.

    This node is controller-agnostic: any upstream command source publishes to
    input_topic, and downstream controllers consume output_topic.
    """

    def __init__(self):
        super().__init__('twist_slew_limiter')

        self.declare_parameter('input_topic', '/cmd_vel_desired_raw')
        self.declare_parameter('output_topic', '/cmd_vel_desired')
        self.declare_parameter('loop_hz', 50.0)
        self.declare_parameter('max_linear_accel', 0.25)
        self.declare_parameter('max_angular_accel', 1.20)
        self.declare_parameter('max_linear_speed', 0.30)
        self.declare_parameter('max_angular_speed', 1.50)
        self.declare_parameter('cmd_timeout_s', 0.50)
        self.declare_parameter('initial_linear_speed', 0.0)
        self.declare_parameter('initial_angular_speed', 0.0)

        self._input_topic = str(self.get_parameter('input_topic').value)
        self._output_topic = str(self.get_parameter('output_topic').value)
        self._loop_hz = float(self.get_parameter('loop_hz').value)
        self._max_a_v = float(self.get_parameter('max_linear_accel').value)
        self._max_a_w = float(self.get_parameter('max_angular_accel').value)
        self._max_v = float(self.get_parameter('max_linear_speed').value)
        self._max_w = float(self.get_parameter('max_angular_speed').value)
        self._cmd_timeout = float(self.get_parameter('cmd_timeout_s').value)
        self._v_out = float(self.get_parameter('initial_linear_speed').value)
        self._w_out = float(self.get_parameter('initial_angular_speed').value)

        if self._loop_hz <= 0.0:
            raise ValueError('loop_hz must be > 0')
        if self._max_a_v <= 0.0 or self._max_a_w <= 0.0:
            raise ValueError('max_linear_accel and max_angular_accel must be > 0')

        self._v_target = _clamp(self._v_out, -self._max_v, self._max_v)
        self._w_target = _clamp(self._w_out, -self._max_w, self._max_w)
        self._last_target_time = None
        self._last_time = None

        self.create_subscription(Twist, self._input_topic, self._cb_cmd, 10)
        self._pub = self.create_publisher(Twist, self._output_topic, 10)
        self.create_timer(1.0 / self._loop_hz, self._tick)

        self.get_logger().info(
            f'TwistSlewLimiter ready: {self._input_topic} -> {self._output_topic}, '
            f'max_a_v={self._max_a_v}, max_a_w={self._max_a_w}, '
            f'v0={self._v_out}, w0={self._w_out}'
        )

    def _cb_cmd(self, msg: Twist):
        self._v_target = _clamp(msg.linear.x, -self._max_v, self._max_v)
        self._w_target = _clamp(msg.angular.z, -self._max_w, self._max_w)
        self._last_target_time = self.get_clock().now()

    def _tick(self):
        now = self.get_clock().now()
        if self._last_time is None:
            self._last_time = now
            self._publish()
            return

        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now
        if dt <= 0.0 or dt > 0.5:
            return

        if self._last_target_time is not None:
            age = (now - self._last_target_time).nanoseconds * 1e-9
            if age > self._cmd_timeout:
                self._v_target = 0.0
                self._w_target = 0.0

        dv_max = self._max_a_v * dt
        dw_max = self._max_a_w * dt

        self._v_out += _clamp(self._v_target - self._v_out, -dv_max, dv_max)
        self._w_out += _clamp(self._w_target - self._w_out, -dw_max, dw_max)

        self._v_out = _clamp(self._v_out, -self._max_v, self._max_v)
        self._w_out = _clamp(self._w_out, -self._max_w, self._max_w)

        self._publish()

    def _publish(self):
        msg = Twist()
        msg.linear.x = self._v_out
        msg.angular.z = self._w_out
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TwistSlewLimiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
