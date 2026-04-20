#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MiniChallengeCtrl(Node):
    def __init__(self):
        super().__init__('open_loop_controller')

        self.declare_parameter('path_mode', 'square')
        self.declare_parameter('square_side_m', 0.35)
        self.declare_parameter('waypoints_xy', [])
        self.declare_parameter('close_path', True)
        self.declare_parameter('segment_types', ['move', 'turn', 'move', 'turn', 'move', 'turn', 'move'])
        self.declare_parameter('segment_values', [0.35, math.pi / 2.0, 0.35, math.pi / 2.0, 0.35, math.pi / 2.0, 0.35])

        self.declare_parameter('plan_mode', 'speed')
        self.declare_parameter('target_linear_speed', 0.10)
        self.declare_parameter('target_angular_speed', 0.60)
        self.declare_parameter('total_time_s', 120.0)

        self.declare_parameter('max_linear_speed', 0.50)
        self.declare_parameter('max_angular_speed', 1.50)
        self.declare_parameter('min_linear_speed_cmd', 0.08)
        self.declare_parameter('min_angular_speed_cmd', 0.20)
        self.declare_parameter('max_linear_accel', 0.25)
        self.declare_parameter('max_angular_accel', 1.0)

        self.declare_parameter('loop_hz', 20.0)
        self.declare_parameter('max_run_time_s', 240.0)
        self.declare_parameter('stop_burst_cycles', 30)

        self.path_mode = str(self.get_parameter('path_mode').value)
        self.square_side_m = float(self.get_parameter('square_side_m').value)
        self.close_path = bool(self.get_parameter('close_path').value)
        self.segment_types = [str(x) for x in list(self.get_parameter('segment_types').value)]
        self.segment_values = [float(x) for x in list(self.get_parameter('segment_values').value)]

        self.plan_mode = str(self.get_parameter('plan_mode').value)
        self.target_linear_speed = float(self.get_parameter('target_linear_speed').value)
        self.target_angular_speed = float(self.get_parameter('target_angular_speed').value)
        self.total_time_s = float(self.get_parameter('total_time_s').value)

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.min_linear_speed_cmd = float(self.get_parameter('min_linear_speed_cmd').value)
        self.min_angular_speed_cmd = float(self.get_parameter('min_angular_speed_cmd').value)
        self.max_linear_accel = float(self.get_parameter('max_linear_accel').value)
        self.max_angular_accel = float(self.get_parameter('max_angular_accel').value)

        self.loop_hz = float(self.get_parameter('loop_hz').value)
        self.max_run_time_s = float(self.get_parameter('max_run_time_s').value)
        self.stop_burst_cycles = int(self.get_parameter('stop_burst_cycles').value)

        if self.max_linear_speed <= 0.0 or self.max_angular_speed <= 0.0:
            raise ValueError('max_linear_speed and max_angular_speed must be > 0')
        if self.loop_hz <= 0.0 or self.max_run_time_s <= 0.0 or self.stop_burst_cycles <= 0:
            raise ValueError('loop_hz, max_run_time_s and stop_burst_cycles must be > 0')

        self.segments, total_distance, total_turn = self._build_segments_from_inputs()
        if not self.segments:
            raise ValueError('No valid segments found. Check segment_types/segment_values or waypoint settings.')

        linear_speed, angular_speed, estimated_time = self._solve_plan(total_distance, total_turn)
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.estimated_time_s = estimated_time

        self._apply_segment_durations()

        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', 10)
        self.wait_for_ros_time_if_needed()

        self.current_idx = 0
        self.state_start_time = self.get_clock().now()
        self.sequence_start_time = self.state_start_time
        self.current_v = 0.0
        self.current_w = 0.0
        self.stop_counter = 0
        self.in_emergency_stop = False

        self.timer = self.create_timer(1.0 / self.loop_hz, self.control_loop)

        self.get_logger().info(
            f'Plan ready: segments={len(self.segments)}, mode={self.plan_mode}, '
            f'v={self.linear_speed:.3f} m/s, w={self.angular_speed:.3f} rad/s, '
            f'estimated_time={self.estimated_time_s:.2f}s'
        )

    def wait_for_ros_time_if_needed(self):
        use_sim_time = bool(self.get_parameter('use_sim_time').value)
        if not use_sim_time:
            return
        self.get_logger().info('Waiting for ROS sim time to become active...')
        while rclpy.ok():
            now = self.get_clock().now()
            if now.nanoseconds > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('ROS sim time is active.')

    def _build_segments_from_inputs(self):
        # Preferred mode: explicit sequence in YAML via segment_types/segment_values.
        if self.segment_types:
            return self._build_segments_from_sequence(self.segment_types, self.segment_values)

        # Backward-compatible fallback for existing configs.
        waypoints = self._build_waypoints()
        return self._build_segments_from_waypoints(waypoints)

    def _build_segments_from_sequence(self, types: List[str], values: List[float]):
        if len(types) != len(values):
            raise ValueError('segment_types and segment_values must have the same length')

        segments = []
        total_distance = 0.0
        total_turn = 0.0

        for idx, seg_type in enumerate(types):
            kind = seg_type.strip().lower()
            val = float(values[idx])
            if kind not in ('move', 'turn'):
                raise ValueError(f"Invalid segment_types[{idx}]='{seg_type}'. Allowed values: move, turn")

            if kind == 'move':
                if val <= 0.0:
                    raise ValueError(f'segment_values[{idx}] for move must be > 0 (meters)')
                total_distance += val
                segments.append({'type': 'move', 'value': val, 'duration': 0.0})
            else:
                if abs(val) <= 1e-9:
                    raise ValueError(f'segment_values[{idx}] for turn must be non-zero (radians)')
                total_turn += abs(val)
                segments.append({'type': 'turn', 'value': val, 'duration': 0.0})

        return segments, total_distance, total_turn

    def _build_waypoints(self) -> List[Tuple[float, float]]:
        if self.path_mode == 'square':
            l = self.square_side_m
            if l <= 0.0:
                raise ValueError('square_side_m must be > 0')
            return [(0.0, 0.0), (l, 0.0), (l, l), (0.0, l), (0.0, 0.0)]

        if self.path_mode != 'waypoints':
            raise ValueError("path_mode must be 'square' or 'waypoints'")

        raw = list(self.get_parameter('waypoints_xy').value)
        if len(raw) < 4 or len(raw) % 2 != 0:
            raise ValueError('waypoints_xy must contain an even number of values (x1,y1,x2,y2,...)')

        points = []
        for i in range(0, len(raw), 2):
            points.append((float(raw[i]), float(raw[i + 1])))

        if self.close_path and points[0] != points[-1]:
            points.append(points[0])

        return points

    def _build_segments_from_waypoints(self, points: List[Tuple[float, float]]):
        segments = []
        headings = []
        total_distance = 0.0
        total_turn = 0.0

        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            distance = math.hypot(dx, dy)
            if distance < 1e-6:
                continue

            heading = math.atan2(dy, dx)
            headings.append(heading)
            segments.append({'type': 'move', 'value': distance, 'duration': 0.0})
            total_distance += distance

        for i in range(len(headings) - 1):
            dtheta = wrap_to_pi(headings[i + 1] - headings[i])
            if abs(dtheta) < 1e-6:
                continue
            segments.insert((2 * i) + 1, {'type': 'turn', 'value': dtheta, 'duration': 0.0})
            total_turn += abs(dtheta)

        return segments, total_distance, total_turn

    def _solve_plan(self, total_distance: float, total_turn: float):
        w = min(abs(self.target_angular_speed), self.max_angular_speed)
        if w <= 0.0:
            raise ValueError('target_angular_speed must be > 0')

        if self.plan_mode == 'speed':
            v = min(abs(self.target_linear_speed), self.max_linear_speed)
            if v <= 0.0:
                raise ValueError('target_linear_speed must be > 0')
            time_s = (total_distance / v) + (total_turn / w)
            return v, w, time_s

        if self.plan_mode == 'time':
            if self.total_time_s <= 0.0:
                raise ValueError('total_time_s must be > 0')
            turn_time = total_turn / w
            linear_time = self.total_time_s - turn_time
            if linear_time <= 0.0:
                raise ValueError('Unreachable path/time: total_time_s is too short for required turns')
            v = total_distance / linear_time
            if v > self.max_linear_speed:
                raise ValueError(
                    f'Unreachable path/time: required linear speed {v:.3f} > max_linear_speed {self.max_linear_speed:.3f}'
                )
            return v, w, self.total_time_s

        raise ValueError("plan_mode must be 'speed' or 'time'")

    def _apply_segment_durations(self):
        for seg in self.segments:
            if seg['type'] == 'move':
                seg['duration'] = self._duration_with_accel(seg['value'], self.linear_speed, self.max_linear_accel)
            else:
                seg['duration'] = self._duration_with_accel(abs(seg['value']), self.angular_speed, self.max_angular_accel)

    def _duration_with_accel(self, amount: float, cruise_speed: float, accel_limit: float) -> float:
        if amount <= 0.0:
            return 0.0
        if cruise_speed <= 0.0:
            raise ValueError('cruise_speed must be > 0')
        if accel_limit <= 1e-9:
            return amount / cruise_speed

        ramp_time = cruise_speed / accel_limit
        ramp_amount = 0.5 * accel_limit * ramp_time * ramp_time

        if amount <= ramp_amount:
            return math.sqrt(2.0 * amount / accel_limit)

        cruise_amount = amount - ramp_amount
        return ramp_time + (cruise_amount / cruise_speed)

    def _ramp(self, current: float, target: float, accel_limit: float, dt: float) -> float:
        max_step = accel_limit * dt
        delta = target - current
        if delta > max_step:
            return current + max_step
        if delta < -max_step:
            return current - max_step
        return target

    def publish_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub_cmd.publish(msg)

    def control_loop(self):
        now = self.get_clock().now()
        elapsed_total = (now - self.sequence_start_time).nanoseconds * 1e-9
        if elapsed_total >= self.max_run_time_s and self.current_idx < len(self.segments):
            self.get_logger().error('Failsafe timeout reached. Triggering emergency stop.')
            self.in_emergency_stop = True
            self.current_idx = len(self.segments)
            self.stop_counter = 0

        if self.current_idx >= len(self.segments):
            self.current_v = 0.0
            self.current_w = 0.0
            self.publish_cmd(0.0, 0.0)
            self.stop_counter += 1
            if self.stop_counter >= self.stop_burst_cycles:
                self.timer.cancel()
                if self.in_emergency_stop:
                    self.get_logger().warn('Emergency stop complete.')
                else:
                    self.get_logger().info('Path complete. Robot stopped.')
            return

        seg = self.segments[self.current_idx]
        elapsed_seg = (now - self.state_start_time).nanoseconds * 1e-9

        desired_v = 0.0
        desired_w = 0.0
        if seg['type'] == 'move':
            desired_v = self.linear_speed
        else:
            desired_w = math.copysign(self.angular_speed, seg['value'])

        dt = 1.0 / self.loop_hz
        # Open-loop axis isolation: moves are pure translation, turns are pure rotation.
        if seg['type'] == 'move':
            self.current_w = 0.0
            self.current_v = self._ramp(self.current_v, desired_v, self.max_linear_accel, dt)
        else:
            self.current_v = 0.0
            self.current_w = self._ramp(self.current_w, desired_w, self.max_angular_accel, dt)

        # Enforce minimum command only when that axis is actively commanded.
        if abs(desired_v) > 0.0 and abs(self.current_v) > 0.0 and abs(self.current_v) < self.min_linear_speed_cmd:
            self.current_v = math.copysign(self.min_linear_speed_cmd, self.current_v)
        if abs(desired_w) > 0.0 and abs(self.current_w) > 0.0 and abs(self.current_w) < self.min_angular_speed_cmd:
            self.current_w = math.copysign(self.min_angular_speed_cmd, self.current_w)

        self.publish_cmd(self.current_v, self.current_w)

        if elapsed_seg >= seg['duration']:
            self.current_idx += 1
            self.state_start_time = now
            self.get_logger().info(f'Segment completed: {self.current_idx}/{len(self.segments)}')

    def emergency_stop(self, reason: str):
        self.get_logger().warn(f'Emergency stop requested: {reason}')
        self.in_emergency_stop = True
        self.current_idx = len(self.segments)
        self.stop_counter = 0
        for _ in range(self.stop_burst_cycles):
            self.publish_cmd(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.01)


def main(args=None):
    rclpy.init(args=args)
    node = MiniChallengeCtrl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.emergency_stop('keyboard interrupt')
    finally:
        node.emergency_stop('node shutdown')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()