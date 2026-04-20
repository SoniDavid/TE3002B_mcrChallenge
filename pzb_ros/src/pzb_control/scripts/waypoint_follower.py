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


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _yaw_from_quaternion(qz: float, qw: float) -> float:
    return 2.0 * math.atan2(qz, qw)


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
        self.declare_parameter('dist_tol_m', 0.05)
        self.declare_parameter('yaw_tol_rad', 0.05)
        self.declare_parameter('max_linear_speed', 0.20)
        self.declare_parameter('max_angular_speed', 1.20)
        self.declare_parameter('min_linear_speed', 0.05)  # dead zone: below this → 0
        self.declare_parameter('align_yaw', True)
        # Intermediate waypoints are advanced this far before reaching them,
        # so the robot never slows to zero between waypoints.
        self.declare_parameter('lookahead_dist', 0.15)
        self.declare_parameter('loop_hz', 20.0)
        self.declare_parameter('waypoints_xyyaw', [0.5, 0.0, 0.0])

        self._kp_dist = self.get_parameter('Kp_dist').value
        self._kp_yaw = self.get_parameter('Kp_yaw').value
        self._dist_tol = self.get_parameter('dist_tol_m').value
        self._yaw_tol = self.get_parameter('yaw_tol_rad').value
        self._max_v = self.get_parameter('max_linear_speed').value
        self._max_w = self.get_parameter('max_angular_speed').value
        self._min_v = self.get_parameter('min_linear_speed').value
        self._align_yaw = self.get_parameter('align_yaw').value
        self._lookahead = self.get_parameter('lookahead_dist').value

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0

        self._state = State.IDLE
        self._waypoints: list[tuple[float, float, float]] = []
        self._wp_idx = 0

        raw = self.get_parameter('waypoints_xyyaw').value
        self._load_from_list(raw)

        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self.create_subscription(PoseArray, '/waypoints', self._cb_waypoints, 10)
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_desired', 10)

        hz = self.get_parameter('loop_hz').value
        self.create_timer(1.0 / hz, self._control_loop)

        self.get_logger().info(
            f'WaypointFollower ready — {len(self._waypoints)} waypoint(s), '
            f'align_yaw={self._align_yaw}'
        )
        if self._waypoints:
            self._state = State.MOVING
            self.get_logger().info(f'Moving to WP 0: {self._waypoints[0]}')

    #  Subscribers

    def _cb_odom(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._theta = _yaw_from_quaternion(
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

    def _cb_waypoints(self, msg: PoseArray):
        wps = [
            (p.position.x, p.position.y,
             _yaw_from_quaternion(p.orientation.z, p.orientation.w))
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
            alpha = _wrap_to_pi(math.atan2(dy, dx) - self._theta)

            v = _clamp(self._kp_dist * d * max(0.0, math.cos(alpha)),
                       0.0, self._max_v)
            # Dead zone: ignore tiny computed speeds — prevents post-goal creep
            if v < self._min_v:
                v = 0.0
            omega = _clamp(self._kp_yaw * alpha, -self._max_w, self._max_w)
            self._publish(v, omega)

            is_last = (self._wp_idx == len(self._waypoints) - 1)
            # Intermediate waypoints: advance early (lookahead) to stay continuous.
            # Last waypoint: wait for precise arrival.
            threshold = self._dist_tol if is_last else self._lookahead
            if d < threshold:
                if is_last and self._align_yaw:
                    self._state = State.ALIGNING
                    self.get_logger().info(
                        f'[WP {self._wp_idx}] Position reached — aligning yaw'
                    )
                else:
                    self._advance()

        elif self._state == State.ALIGNING:
            e_yaw = _wrap_to_pi(yaw_goal - self._theta)
            self._publish(0.0, _clamp(self._kp_yaw * e_yaw, -self._max_w, self._max_w))
            if abs(e_yaw) < self._yaw_tol:
                self.get_logger().info(
                    f'[WP {self._wp_idx}] Done '
                    f'(x={self._x:.3f} y={self._y:.3f} θ={math.degrees(self._theta):.1f}°)'
                )
                self._advance()

    def _advance(self):
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
        self._waypoints = [(raw[i], raw[i+1], raw[i+2]) for i in range(0, len(raw), 3)]


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
