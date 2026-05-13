#!/usr/bin/env python3
"""
Traffic light FSM node.

Consumes detected traffic light color and applies behavioral rules:
  - Green  → full speed (scale = 1.0)
  - Yellow → ramp from yellow_speed_scale down to 0.0 over yellow_ramp_duration_s;
             resumes instantly on green; if yellow lost, times out back to RUNNING
  - Red    → full stop, LATCHED — stays stopped even if red disappears;
             only resumes when green is explicitly seen
  - None   → drive normally (scale = 1.0), unless latched on red

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
    SLOW    = 'slow'      # scale = yellow_speed_scale
    STOPPED = 'stopped'   # scale = 0.0, latched


class TrafficLightFSMNode(Node):

    def __init__(self):
        super().__init__('traffic_light_fsm_node')

        self.declare_parameter('yellow_speed_scale', 0.4)
        self.declare_parameter('yellow_ramp_duration_s', 3.0)
        self.declare_parameter('color_lost_timeout', 1.5)
        self.declare_parameter('loop_hz', 20.0)

        self._yellow_scale    = float(self.get_parameter('yellow_speed_scale').value)
        self._ramp_duration   = float(self.get_parameter('yellow_ramp_duration_s').value)
        self._lost_timeout    = float(self.get_parameter('color_lost_timeout').value)

        self._state           = TrafficState.RUNNING
        self._last_color      = 'none'
        self._last_color_time = None   # wall time of last /traffic_light_color message
        self._yellow_start    = None   # wall time when SLOW state was entered

        self.create_subscription(String, '/traffic_light_color', self._color_callback, 10)
        self._scale_pub = self.create_publisher(Float32, '/traffic_speed_scale', 10)

        hz = self.get_parameter('loop_hz').value
        self.create_timer(1.0 / hz, self._publish_loop)

        self.get_logger().info(
            f'TrafficLightFSMNode ready — '
            f'yellow_scale={self._yellow_scale}, ramp={self._ramp_duration}s, '
            f'lost_timeout={self._lost_timeout}s'
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
            # Green always resumes, regardless of current state
            self._set_state(TrafficState.RUNNING)

        elif color == 'red':
            # Red always stops, regardless of current state
            self._set_state(TrafficState.STOPPED)

        elif color == 'yellow':
            if self._state == TrafficState.RUNNING:
                self._set_state(TrafficState.SLOW)
            # If STOPPED (latched on red), yellow does NOT resume

        elif color == 'none':
            # Lost detection:
            # - STOPPED: latch holds — stay stopped until green
            # - SLOW: will time out back to RUNNING via _publish_loop
            # - RUNNING: no change
            pass

    def _set_state(self, new_state: TrafficState):
        if new_state != self._state:
            self.get_logger().info(
                f'FSM: {self._state.value} → {new_state.value}'
            )
            self._state = new_state
            if new_state == TrafficState.SLOW:
                self._yellow_start = time.monotonic()

    def _current_scale(self) -> float:
        if self._state == TrafficState.RUNNING:
            return 1.0
        if self._state == TrafficState.SLOW:
            elapsed = time.monotonic() - self._yellow_start
            frac = min(elapsed / self._ramp_duration, 1.0)
            return self._yellow_scale * (1.0 - frac)
        return 0.0   # STOPPED

    @staticmethod
    def _state_for_color(color: str) -> str:
        return {'green': 'running', 'yellow': 'slow', 'red': 'stopped'}.get(color, 'running')

    # ------------------------------------------------------------------ publish loop

    def _publish_loop(self):
        now = time.monotonic()

        # If SLOW and we haven't seen a color for lost_timeout, return to RUNNING.
        # STOPPED does NOT time out — it requires explicit green to resume.
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
        # Publish zero scale on shutdown for safety
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
