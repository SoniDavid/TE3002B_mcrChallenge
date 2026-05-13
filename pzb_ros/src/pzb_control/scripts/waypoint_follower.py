#!/usr/bin/env python3
"""
Waypoint follower — outer loop of the cascade control system.

Uses a smooth unicycle controller: v and ω are commanded simultaneously so
the robot curves toward each waypoint instead of stopping to turn.

Control law (go-to-goal):
    d     = distance to goal
    alpha = heading error to goal (wrap to [-pi, pi])

    v = Kp_dist * d * max(0, cos(alpha))   # slows when misaligned, stops if > 90 deg off
    ω = Kp_yaw  * alpha                    # steers toward goal at all times

The cos(alpha) factor makes trajectories naturally curved: full speed when
aligned, graceful deceleration when steering.  No discrete stop-turn-move
phases.

Final yaw alignment (optional per waypoint via 'align_yaw' parameter):
    After reaching position, smoothly rotate to the target heading.
"""
import enum
import math
import signal

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class State(enum.Enum):
    IDLE = 0
    MOVING = 1       # smooth curved approach to position
    ALIGNING = 2     # final yaw rotation at waypoint
    DONE = 3


class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        self.declare_parameter('Kp_dist', 1.0)
        self.declare_parameter('Kp_yaw', 2.0)
        self.declare_parameter('Ki_yaw', 0.3)
        self.declare_parameter('yaw_integral_clamp_deg', 30.0)
        self.declare_parameter('dist_tol_m', 0.05)
        self.declare_parameter('yaw_tol_deg', 3.0)
        self.declare_parameter('max_linear_speed', 0.20)
        self.declare_parameter('max_angular_speed', 1.20)
        self.declare_parameter('min_linear_speed', 0.05)  # dead zone: below this → 0
        self.declare_parameter('align_yaw', True)
        # When true, stop and align yaw at every waypoint, not just the last.
        # Use this for precise paths (e.g. square) where corner drift matters.
        self.declare_parameter('align_at_all_waypoints', False)
        # Intermediate waypoints are advanced this far before reaching them,
        # so the robot never slows to zero between waypoints.
        # Set to 0.0 to disable early advance (requires align_at_all_waypoints for clean corners).
        self.declare_parameter('lookahead_dist', 0.15)
        self.declare_parameter('decel_zone_deg', 30.0)
        self.declare_parameter('blend_radius_m', 0.20)
        self.declare_parameter('omega_deadband_rad_s', 0.08)
        self.declare_parameter('loop_hz', 20.0)
        self.declare_parameter('waypoints_xyyaw', [0.5, 0.0, 0.0])

        self._kp_dist = self.get_parameter('Kp_dist').value
        self._kp_yaw = self.get_parameter('Kp_yaw').value
        self._ki_yaw = self.get_parameter('Ki_yaw').value
        self._yaw_integral_clamp = math.radians(self.get_parameter('yaw_integral_clamp_deg').value)
        self._dist_tol = self.get_parameter('dist_tol_m').value
        self._yaw_tol = math.radians(self.get_parameter('yaw_tol_deg').value)
        self._max_v = self.get_parameter('max_linear_speed').value
        self._max_w = self.get_parameter('max_angular_speed').value
        self._min_v = self.get_parameter('min_linear_speed').value
        self._align_yaw = self.get_parameter('align_yaw').value
        self._align_all = self.get_parameter('align_at_all_waypoints').value
        self._lookahead = self.get_parameter('lookahead_dist').value
        self._decel_zone = math.radians(self.get_parameter('decel_zone_deg').value)
        self._blend_radius = self.get_parameter('blend_radius_m').value
        self._omega_deadband = float(self.get_parameter('omega_deadband_rad_s').value)

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._yaw_integral = 0.0

        self._state = State.IDLE
        self._waypoints: list[tuple[float, float, float]] = []
        self._wp_idx = 0

        raw = self.get_parameter('waypoints_xyyaw').value
        self._load_from_list(raw)

        self._speed_scale = 1.0

        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self.create_subscription(PoseArray, '/waypoints', self._cb_waypoints, 10)
        self.create_subscription(Float32, '/traffic_speed_scale', self._cb_speed_scale, 10)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_desired', 10)

        hz = self.get_parameter('loop_hz').value
        self._dt = 1.0 / hz
        self.create_timer(self._dt, self._control_loop)

        self.get_logger().info(
            f'WaypointFollower ready — {len(self._waypoints)} waypoint(s), '
            f'align_yaw={self._align_yaw}, Kp_yaw={self._kp_yaw}, Ki_yaw={self._ki_yaw}'
        )
        if self._waypoints:
            self._state = State.MOVING
            self.get_logger().info(f'Moving to WP 0: {self._waypoints[0]}')

    #  Subscribers

    def _cb_speed_scale(self, msg: Float32):
        self._speed_scale = float(msg.data)

    def _cb_odom(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._theta = _yaw_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

    def _cb_waypoints(self, msg: PoseArray):
        wps = [
            (p.position.x, p.position.y,
             _yaw_from_quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))
            for p in msg.poses
        ]
        if wps:
            self._waypoints = wps
            self._wp_idx = 0
            self._state = State.MOVING
            self.get_logger().info(
                f'Received {len(wps)} waypoint(s) via topic — restarting mission.'
            )

    #  Control loop 

    def _control_loop(self):
        if self._state in (State.IDLE, State.DONE):
            self._publish(0.0, 0.0)
            return

        xg, yg, yaw_goal = self._waypoints[self._wp_idx]

        if self._state == State.MOVING:
            dx = xg - self._x
            dy = yg - self._y
            d = math.hypot(dx, dy)
            alpha_path = _wrap_to_pi(math.atan2(dy, dx) - self._theta)

            is_last = (self._wp_idx == len(self._waypoints) - 1)
            should_align = self._align_yaw and (is_last or self._align_all)

            # Heading blend: within blend_radius, smoothly rotate toward the
            # goal heading rather than purely steering at the waypoint.
            # v is always computed from alpha_path so forward progress is unaffected.
            if should_align and self._blend_radius > 0.0 and d < self._blend_radius:
                blend = _clamp(1.0 - d / self._blend_radius, 0.0, 1.0)
                alpha_goal = _wrap_to_pi(yaw_goal - self._theta)
                # Interpolate along the shortest angular arc
                alpha_diff = _wrap_to_pi(alpha_goal - alpha_path)
                alpha_for_omega = _wrap_to_pi(alpha_path + blend * alpha_diff)
            else:
                alpha_for_omega = alpha_path

            v = _clamp(self._kp_dist * d * max(0.0, math.cos(alpha_path)),
                       0.0, self._max_v)
            if v < self._min_v:
                v = 0.0
            omega = _clamp(self._kp_yaw * alpha_for_omega, -self._max_w, self._max_w)

            # Traffic light gate: scale=0.0 means full stop (red latch)
            if self._speed_scale <= 0.0:
                self._publish(0.0, 0.0)
            else:
                self._publish(v * self._speed_scale, omega)

            # Last waypoint: use dist_tol for precise arrival.
            # Intermediate: use lookahead for early advance, but never below dist_tol.
            # Freeze advancement while stopped — prevents the slew limiter's
            # coast-down from pushing the robot past the lookahead threshold
            # and skipping to the wrong waypoint during a traffic-light stop.
            if self._speed_scale > 0.0:
                threshold = self._dist_tol if is_last else max(self._lookahead, self._dist_tol)
                if d < threshold:
                    if should_align:
                        e_yaw = _wrap_to_pi(yaw_goal - self._theta)
                        if abs(e_yaw) > self._yaw_tol:
                            # Blending didn't fully converge — fine-correct with ALIGNING
                            self._state = State.ALIGNING
                            self._yaw_integral = 0.0
                            self.get_logger().info(
                                f'[WP {self._wp_idx}] Position reached — '
                                f'aligning yaw ({math.degrees(e_yaw):.1f}° remaining)'
                            )
                        else:
                            self._advance()
                    else:
                        self._advance()

        elif self._state == State.ALIGNING:
            e_yaw = _wrap_to_pi(yaw_goal - self._theta)
            self._yaw_integral = _clamp(
                self._yaw_integral + e_yaw * self._dt,
                -self._yaw_integral_clamp,
                self._yaw_integral_clamp,
            )
            # Ramp down max ω proportionally inside the decel zone so the robot
            # decelerates into the target heading instead of snapping to zero.
            w_cap = _clamp(abs(e_yaw) / max(self._decel_zone, 1e-6) * self._max_w,
                           0.0, self._max_w)
            omega = _clamp(
                self._kp_yaw * e_yaw + self._ki_yaw * self._yaw_integral,
                -w_cap, w_cap,
            )
            # Below the motor deadzone the wheel alternates start/stop → whine.
            # Zero the command instead so the motor stops cleanly.
            if abs(omega) < self._omega_deadband:
                omega = 0.0
            self._publish(0.0, omega)
            if abs(e_yaw) < self._yaw_tol:
                self.get_logger().info(
                    f'[WP {self._wp_idx}] Done '
                    f'(x={self._x:.3f} y={self._y:.3f} θ={math.degrees(self._theta):.1f}°)'
                )
                self._advance()

    def _advance(self):
        self._publish(0.0, 0.0)
        self._yaw_integral = 0.0
        self._wp_idx += 1
        if self._wp_idx >= len(self._waypoints):
            self._state = State.DONE
            self.get_logger().info('All waypoints reached.')
        else:
            self._state = State.MOVING
            self.get_logger().info(
                f'Moving to WP {self._wp_idx}: {self._waypoints[self._wp_idx]}'
            )

    def _publish(self, v: float, omega: float):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = omega
        self._cmd_pub.publish(msg)

    def _load_from_list(self, raw: list):
        if len(raw) % 3 != 0 or len(raw) == 0:
            self.get_logger().warn(
                f'waypoints_xyyaw has {len(raw)} elements — must be multiple of 3.'
            )
            return
        self._waypoints = [
            (raw[i], raw[i + 1], math.radians(raw[i + 2]))
            for i in range(0, len(raw), 3)
        ]


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()

    def _shutdown(signum, frame):
        node._publish(0.0, 0.0)
        rclpy.try_shutdown()

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._publish(0.0, 0.0)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
