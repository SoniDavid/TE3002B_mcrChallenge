#!/usr/bin/env python3
"""
Velocity controller — inner loop of the cascade control system.

Subscribes to /cmd_vel_desired (desired body velocity) and /robot_vel
(actual body velocity from MCU), runs a body-level PI, and publishes the
corrected command to /cmd_vel.

Only /cmd_vel is ever written. 
The MCU owns all kinematic parameters.
"""
import math
import signal
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, TwistStamped

_MCU_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _deadzone_comp(value: float, dz: float) -> float:
    """If |value| is non-zero but below the dead zone, snap it to ±dz."""
    if dz <= 0.0 or value == 0.0:
        return value
    if abs(value) < dz:
        return math.copysign(dz, value)
    return value


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, integral_clamp: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_clamp = integral_clamp
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        self._integral = _clamp(
            self._integral + error * dt,
            -self.integral_clamp,
            self.integral_clamp,
        )
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')

        self.declare_parameter('kp_v', 0.5)
        self.declare_parameter('ki_v', 0.1)
        self.declare_parameter('kd_v', 0.0)
        self.declare_parameter('kp_w', 0.8)
        self.declare_parameter('ki_w', 0.1)
        self.declare_parameter('kd_w', 0.0)
        self.declare_parameter('integral_clamp', 0.2)
        self.declare_parameter('loop_hz', 50.0)
        self.declare_parameter('max_linear_speed', 0.30)
        self.declare_parameter('max_angular_speed', 1.50)
        self.declare_parameter('cmd_timeout_s', 0.50)
        self.declare_parameter('deadzone_comp_v', 0.0)
        self.declare_parameter('deadzone_comp_w', 0.0)

        self._max_v = self.get_parameter('max_linear_speed').value
        self._max_w = self.get_parameter('max_angular_speed').value
        self._cmd_timeout = self.get_parameter('cmd_timeout_s').value
        self._dz_v = self.get_parameter('deadzone_comp_v').value
        self._dz_w = self.get_parameter('deadzone_comp_w').value

        ic = self.get_parameter('integral_clamp').value
        self._pi_v = PIDController(
            self.get_parameter('kp_v').value,
            self.get_parameter('ki_v').value,
            self.get_parameter('kd_v').value,
            ic,
        )
        self._pi_w = PIDController(
            self.get_parameter('kp_w').value,
            self.get_parameter('ki_w').value,
            self.get_parameter('kd_w').value,
            ic,
        )

        self._v_des = 0.0
        self._w_des = 0.0
        self._v_actual = 0.0
        self._w_actual = 0.0
        self._last_desired_time = None
        self._last_time = None

        self.create_subscription(Twist, '/cmd_vel_desired', self._cb_desired, 10)
        # /robot_vel is the MCU's body velocity — read-only feedback
        self.create_subscription(TwistStamped, '/robot_vel', self._cb_robot_vel, _MCU_QOS)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        hz = self.get_parameter('loop_hz').value
        self.create_timer(1.0 / hz, self._control_loop)

        self.get_logger().info(
            f'VelocityController ready — '
            f'kp_v={self._pi_v.kp}, ki_v={self._pi_v.ki}, kd_v={self._pi_v.kd}, '
            f'kp_w={self._pi_w.kp}, ki_w={self._pi_w.ki}, kd_w={self._pi_w.kd}'
        )

    def _cb_desired(self, msg: Twist):
        self._v_des = msg.linear.x
        self._w_des = msg.angular.z
        self._last_desired_time = self.get_clock().now()

    def _cb_robot_vel(self, msg: TwistStamped):
        self._v_actual = msg.twist.linear.x
        self._w_actual = msg.twist.angular.z

    def _control_loop(self):
        now = self.get_clock().now()
        if self._last_time is None:
            self._last_time = now
            return

        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now

        if dt <= 0.0 or dt > 0.5:
            return

        # Safety: stale /cmd_vel_desired → stop and reset integrators
        if self._last_desired_time is not None:
            age = (now - self._last_desired_time).nanoseconds * 1e-9
            if age > self._cmd_timeout:
                self._v_des = 0.0
                self._w_des = 0.0
                self._pi_v.reset()
                self._pi_w.reset()

        # When desired velocity is zero: skip PI, reset integrators, publish exact zero.
        # This prevents integral windup from causing post-stop creep.
        if self._v_des == 0.0 and self._w_des == 0.0:
            self._pi_v.reset()
            self._pi_w.reset()
            self._cmd_pub.publish(Twist())
            return

        v_cmd = _clamp(
            self._v_des + self._pi_v.update(self._v_des - self._v_actual, dt),
            -self._max_v, self._max_v,
        )
        w_cmd = _clamp(
            self._w_des + self._pi_w.update(self._w_des - self._w_actual, dt),
            -self._max_w, self._max_w,
        )

        v_cmd = _deadzone_comp(v_cmd, self._dz_v)
        w_cmd = _deadzone_comp(w_cmd, self._dz_w)

        msg = Twist()
        msg.linear.x = v_cmd
        msg.angular.z = w_cmd
        self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityController()

    def _shutdown(signum, frame):
        node._cmd_pub.publish(Twist())
        rclpy.try_shutdown()

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._cmd_pub.publish(Twist())
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
