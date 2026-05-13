#!/usr/bin/env python3
"""
Traffic light FSM node.

Consumes detected traffic light color and applies behavioral rules:
  - Green  → full speed (scale = 1.0)
  - Yellow → fixed reduced speed (yellow_speed_scale, e.g. 0.4) while yellow visible
  - Red    → immediate stop (scale = 0.0)
  - None   → treat as green, resume full speed

Subscribes:
  /traffic_light_color  (std_msgs/String)  "red" | "green" | "yellow" | "none"

Publishes:
  /traffic_speed_scale  (std_msgs/Float32)  multiplier applied to waypoint follower v
"""

import signal
import enum
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32


class TrafficState(enum.Enum):
    RUNNING = 'running'   # scale = 1.0
    SLOW    = 'slow'      # scale = yellow_speed_scale (fixed)
    STOPPED = 'stopped'   # scale ramps 1.0 → 0.0 over red_ramp_duration_s


class TrafficLightFSMNode(Node):

    def __init__(self):
        super().__init__('traffic_light_fsm_node')

        self.declare_parameter('yellow_speed_scale', 0.4)
        self.declare_parameter('color_lost_timeout', 1.5)
        self.declare_parameter('loop_hz',            20.0)

        self._yellow_scale = float(self.get_parameter('yellow_speed_scale').value)
        self._lost_timeout = float(self.get_parameter('color_lost_timeout').value)

        self._state              = TrafficState.RUNNING
        self._last_color         = 'none'
        self._last_color_time    = None
        self._waiting_for_green  = False   # latched True on red; only green clears it

        self.create_subscription(String, '/traffic_light_color', self._color_callback, 10)
        self._scale_pub = self.create_publisher(Float32, '/traffic_speed_scale', 10)

        hz = self.get_parameter('loop_hz').value
        self.create_timer(1.0 / hz, self._publish_loop)

        self.get_logger().info(
            f'TrafficLightFSMNode ready — '
            f'yellow_scale={self._yellow_scale}, lost_timeout={self._lost_timeout}s'
        )
        self.get_logger().info(f'Initial state: {self._state.value}')

    # ------------------------------------------------------------------ FSM

    def _color_callback(self, msg: String):
        color = msg.data.lower().strip()
        self._last_color_time = time.monotonic()

        if color == self._last_color:
            return   # no change

        prev = self._last_color
        self._last_color = color
        self._transition(color)
        if self._state.value != self._state_for_color(prev):
            self.get_logger().info(
                f'Traffic: {prev} → {color}  |  state → {self._state.value}  '
                f'(scale={self._current_scale():.2f})'
            )

    def _transition(self, color: str):
        if color == 'green':
            self._waiting_for_green = False
            self._set_state(TrafficState.RUNNING)

        elif color == 'none':
            if not self._waiting_for_green:
                self._set_state(TrafficState.RUNNING)

        elif color == 'red':
            self._waiting_for_green = True
            self._set_state(TrafficState.STOPPED)

        elif color == 'yellow':
            if self._state == TrafficState.RUNNING:
                self._set_state(TrafficState.SLOW)

    def _set_state(self, new_state: TrafficState):
        if new_state != self._state:
            self.get_logger().info(
                f'FSM: {self._state.value} → {new_state.value}'
            )
            self._state = new_state

    def _current_scale(self) -> float:
        if self._state == TrafficState.RUNNING:
            return 1.0
        if self._state == TrafficState.SLOW:
            return self._yellow_scale   # fixed reduced speed while yellow visible
        return 0.0   # STOPPED — immediate

    @staticmethod
    def _state_for_color(color: str) -> str:
        return {'green': 'running', 'yellow': 'slow', 'red': 'stopped'}.get(color, 'running')

    # ------------------------------------------------------------------ publish loop

    def _publish_loop(self):
        now = time.monotonic()

        # If SLOW and detection lost for too long, resume
        if (self._state == TrafficState.SLOW
                and self._last_color_time is not None
                and now - self._last_color_time > self._lost_timeout):
            self.get_logger().info(
                f'Yellow timed out after {self._lost_timeout}s without signal — resuming.'
            )
            self._last_color = 'none'
            self._set_state(TrafficState.RUNNING)

        scale_msg = Float32()
        scale_msg.data = self._current_scale()
        self._scale_pub.publish(scale_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightFSMNode()

    def _shutdown(signum, frame):
        msg = Float32()
        msg.data = 0.0
        node._scale_pub.publish(msg)
        rclpy.try_shutdown()

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
