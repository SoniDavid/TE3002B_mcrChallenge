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
import time
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
        self.declare_parameter('robot_vel_stale_timeout_s', 0.10)
        self.declare_parameter('min_dt_s', 0.005)
        self.declare_parameter('hold_zero_until_feedback', True)
        self.declare_parameter('diag_period_s', 5.0)

        self._max_v = self.get_parameter('max_linear_speed').value
        self._max_w = self.get_parameter('max_angular_speed').value
        self._cmd_timeout = self.get_parameter('cmd_timeout_s').value
        self._dz_v = self.get_parameter('deadzone_comp_v').value
        self._dz_w = self.get_parameter('deadzone_comp_w').value
        self._robot_vel_stale_timeout = self.get_parameter('robot_vel_stale_timeout_s').value
        self._min_dt = self.get_parameter('min_dt_s').value
        self._hold_zero_until_feedback = self.get_parameter('hold_zero_until_feedback').value
        self._diag_period = self.get_parameter('diag_period_s').value

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
        self._last_robot_vel_time = None
        self._last_time = None
        self._last_diag_warn_time = 0.0
        self._last_stale_feedback_warn_time = 0.0
        self._last_startup_hold_warn_time = 0.0
        self._last_dt_warn_time = 0.0
        self._count_startup_hold = 0
        self._count_stale_feedback = 0
        self._count_dt_rejected = 0
        self._count_cmd_timeout = 0

        self.create_subscription(Twist, '/cmd_vel_desired', self._cb_desired, 10)
        # /robot_vel is the MCU's body velocity — read-only feedback
        self.create_subscription(TwistStamped, '/robot_vel', self._cb_robot_vel, _MCU_QOS)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        hz = self.get_parameter('loop_hz').value
        self.create_timer(1.0 / hz, self._control_loop)
        self.create_timer(self._diag_period, self._publish_diag_summary)

        self.get_logger().info(
            f'VelocityController ready — '
            f'kp_v={self._pi_v.kp}, ki_v={self._pi_v.ki}, kd_v={self._pi_v.kd}, '
            f'kp_w={self._pi_w.kp}, ki_w={self._pi_w.ki}, kd_w={self._pi_w.kd}, '
            f'min_dt={self._min_dt:.3f}s, robot_vel_stale={self._robot_vel_stale_timeout:.3f}s, '
            f'diag_period={self._diag_period:.1f}s'
        )

    def _cb_desired(self, msg: Twist):
        self._v_des = msg.linear.x
        self._w_des = msg.angular.z
        self._last_desired_time = self.get_clock().now()

    def _cb_robot_vel(self, msg: TwistStamped):
        self._v_actual = msg.twist.linear.x
        self._w_actual = msg.twist.angular.z
        self._last_robot_vel_time = self.get_clock().now()

    def _control_loop(self):
        self._warn_if_robot_interface_missing()

        now = self.get_clock().now()
        if self._last_time is None:
            self._last_time = now
            return

        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now

        if dt <= 0.0 or dt > 0.5 or dt < self._min_dt:
            if dt < self._min_dt:
                self._count_dt_rejected += 1
                self._warn_dt_rejected(dt)
            return

        if self._hold_zero_until_feedback and self._last_robot_vel_time is None:
            self._count_startup_hold += 1
            self._warn_startup_hold()
            self._pi_v.reset()
            self._pi_w.reset()
            self._cmd_pub.publish(Twist())
            return

        if self._last_robot_vel_time is not None:
            robot_vel_age = (now - self._last_robot_vel_time).nanoseconds * 1e-9
            if robot_vel_age > self._robot_vel_stale_timeout:
                self._count_stale_feedback += 1
                self._warn_stale_feedback(robot_vel_age)
                self._pi_v.reset()
                self._pi_w.reset()
                self._cmd_pub.publish(Twist())
                return

        # Safety: stale /cmd_vel_desired → stop and reset integrators
        if self._last_desired_time is not None:
            age = (now - self._last_desired_time).nanoseconds * 1e-9
            if age > self._cmd_timeout:
                self._count_cmd_timeout += 1
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

    def _warn_startup_hold(self):
        now_s = time.monotonic()
        if now_s - self._last_startup_hold_warn_time < 2.0:
            return
        self.get_logger().warn(
            'Holding /cmd_vel at zero: waiting for first /robot_vel feedback sample.'
        )
        self._last_startup_hold_warn_time = now_s

    def _warn_stale_feedback(self, age_s: float):
        now_s = time.monotonic()
        if now_s - self._last_stale_feedback_warn_time < 2.0:
            return
        self.get_logger().warn(
            'Stale /robot_vel feedback '
            f'(age={age_s:.3f}s > {self._robot_vel_stale_timeout:.3f}s). '
            'Publishing zero /cmd_vel for safety.'
        )
        self._last_stale_feedback_warn_time = now_s

    def _warn_dt_rejected(self, dt_s: float):
        now_s = time.monotonic()
        if now_s - self._last_dt_warn_time < 2.0:
            return
        self.get_logger().warn(
            f'Control step rejected due to small dt={dt_s:.6f}s '
            f'(min_dt_s={self._min_dt:.6f}s).'
        )
        self._last_dt_warn_time = now_s

    def _warn_if_robot_interface_missing(self):
        # Emit this warning periodically so missing bring-up is obvious at runtime.
        now_s = time.monotonic()
        if now_s - self._last_diag_warn_time < 2.0:
            return

        cmd_vel_subs = self.count_subscribers('/cmd_vel')
        robot_vel_pubs = self.count_publishers('/robot_vel')

        if cmd_vel_subs == 0 or robot_vel_pubs == 0:
            self.get_logger().warn(
                'Robot interface missing: '
                f'/cmd_vel subscribers={cmd_vel_subs}, '
                f'/robot_vel publishers={robot_vel_pubs}. '
                'Start the base/MCU bridge node so the robot can move.'
            )
            self._last_diag_warn_time = now_s

    def _publish_diag_summary(self):
        # Periodic health snapshot helps correlate safety gates with runtime behavior.
        self.get_logger().info(
            'Safety summary: '
            f'startup_hold={self._count_startup_hold}, '
            f'stale_feedback={self._count_stale_feedback}, '
            f'dt_rejected={self._count_dt_rejected}, '
            f'cmd_timeout={self._count_cmd_timeout}'
        )


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
