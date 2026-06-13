#!/usr/bin/env python3
"""
Traffic light FSM node.

Consumes the traffic-light color AND YOLO signs, and outputs a single speed scale.

Traffic light:
  - Green / None → full speed (1.0)
  - Yellow       → reduced speed (yellow_speed_scale) while visible
  - Red          → full stop (0.0)

YOLO signs (non-turn; turn signs handled by the line follower at a dashed crossing):
  - construccion → reduce speed (yellow_speed_scale) while seen
  - stopSign     → full stop while seen; resume after it leaves view + a confirm delay
  - GIVE WAY     → one brief stop (give_way_pause_s) then continue; re-arms after it
                   leaves view (once per sighting)
  stopSign / GIVE WAY only fire outside a curve (|/line_follower/error| <= curve gate);
  if seen mid-curve the action is deferred until the robot is straight again.

Subscribes:
  /traffic_light_color  (std_msgs/String)
  /yolo/sign            (std_msgs/String)   detected sign class, or 'none'
  /line_follower/error  (std_msgs/Float32)  steering error px (the 'in curve' gate)

Publishes:
  /traffic_speed_scale  (std_msgs/Float32)  multiplier applied to the follower's v
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
        # Traffic-light obedience. Default false: this course has no real light, and the HSV
        # detector false-fires 'red' under changed lighting, latching STOPPED forever. When
        # false, /traffic_light_color is ignored — the YOLO sign behaviors keep working. Set
        # true only with a real light.
        self.declare_parameter('traffic_light_enabled', False)
        # YOLO sign behaviors (non-turn signs; turn signs are handled by the line follower at
        # a dashed crossing):
        #   construccion → slow (like yellow) while seen.
        #   stopSign     → full stop while seen; resume after it leaves view + a confirm
        #                  delay (stop_resume_confirm_s).
        #   GIVE WAY     → one brief stop (give_way_pause_s) then continue; re-arms after the
        #                  sign leaves view (once per sighting).
        # stopSign / GIVE WAY only fire outside a curve (|steering error| <= curve gate);
        # if seen mid-curve the action is deferred until straight again.
        self.declare_parameter('sign_topic',          '/yolo/sign')
        self.declare_parameter('slow_sign_classes',   ['construccion'])
        self.declare_parameter('stop_sign_classes',   ['stopSign'])
        self.declare_parameter('give_way_classes',    ['GIVE WAY'])
        self.declare_parameter('sign_lost_timeout',   1.0)   # construccion slow hold
        self.declare_parameter('give_way_pause_s',    1.5)   # brief stop duration
        self.declare_parameter('stop_resume_confirm_s', 0.5) # delay after stopSign gone
        # 'Outside a curve' gate: act only when |steering error| <= this (px). The robot
        # publishes /line_follower/error; large |error| = a curve, so stops are deferred.
        self.declare_parameter('error_topic',         '/line_follower/error')
        self.declare_parameter('curve_gate_error_px', 35.0)
        self.declare_parameter('sign_intent_stale_s', 3.0)   # forget a deferred intent older than this
        # Peak-area trigger: construccion / GIVE WAY act at the sign's closest point — when
        # the bbox area (parsed off /yolo/sign) has peaked then dropped below peak_drop_frac ×
        # running-max, past peak_min_area. stopSign is exempt (it acts while seen — set
        # sign_peak_enabled false to revert all to seen-based).
        self.declare_parameter('sign_peak_enabled',   True)
        self.declare_parameter('sign_peak_min_area',  0.05)  # area must reach this before acting
        self.declare_parameter('sign_peak_drop_frac', 0.7)   # act when area < this × running-max

        self._yellow_scale = float(self.get_parameter('yellow_speed_scale').value)
        self._lost_timeout = float(self.get_parameter('color_lost_timeout').value)
        self._traffic_light_enabled = bool(self.get_parameter('traffic_light_enabled').value)
        self._slow_signs   = set(self.get_parameter('slow_sign_classes').value)
        self._stop_signs   = set(self.get_parameter('stop_sign_classes').value)
        self._give_way_signs = set(self.get_parameter('give_way_classes').value)
        self._sign_lost_timeout = float(self.get_parameter('sign_lost_timeout').value)
        self._give_way_pause_s  = float(self.get_parameter('give_way_pause_s').value)
        self._stop_resume_confirm_s = float(self.get_parameter('stop_resume_confirm_s').value)
        self._curve_gate_px = float(self.get_parameter('curve_gate_error_px').value)
        self._sign_intent_stale_s = float(self.get_parameter('sign_intent_stale_s').value)
        self._sign_peak_enabled = bool(self.get_parameter('sign_peak_enabled').value)
        self._sign_peak_min_area = float(self.get_parameter('sign_peak_min_area').value)
        self._sign_peak_drop_frac = float(self.get_parameter('sign_peak_drop_frac').value)
        # Per-class running-max area + last-seen, for the peak trigger.
        self._peak_max  = {}   # class -> running max area while in view
        self._peak_seen = {}   # class -> last time seen

        self._state              = TrafficState.RUNNING
        self._last_color         = 'none'
        self._last_color_time    = None
        self._waiting_for_green  = False   # latched True on red; only green clears it
        # construccion slow input — INDEPENDENT of the light; combined in the scale.
        self._sign_slow_until    = None    # monotonic time until which the slow sign holds
        # 'in curve' gate from /line_follower/error
        self._abs_error          = 0.0
        # stopSign state: latched intent (seen, pending) + last-seen time + a resume timer
        self._stop_seen_t        = None    # last time stopSign was seen
        self._stop_active        = False   # currently holding a full stop for stopSign
        self._stop_resume_at     = None    # monotonic time we may resume (sign gone + confirm)
        self._stop_intent_t      = None    # latched deferred stop intent timestamp
        # GIVE WAY state: pending intent + pause window + once-per-sighting arm
        self._gw_intent_t        = None    # latched deferred give-way intent timestamp
        self._gw_pause_until     = None    # monotonic end of the brief pause
        self._gw_armed           = True    # False after a pause until the sign leaves view
        self._gw_seen_t          = None    # last time GIVE WAY was seen

        self.create_subscription(String, '/traffic_light_color', self._color_callback, 10)
        self.create_subscription(
            String, self.get_parameter('sign_topic').value, self._sign_callback, 10)
        self.create_subscription(
            Float32, self.get_parameter('error_topic').value, self._error_callback, 10)
        self._scale_pub = self.create_publisher(Float32, '/traffic_speed_scale', 10)

        hz = self.get_parameter('loop_hz').value
        self.create_timer(1.0 / hz, self._publish_loop)

        self.get_logger().info(
            f'TrafficLightFSMNode ready — yellow_scale={self._yellow_scale}, '
            f'slow={sorted(self._slow_signs)}, stop={sorted(self._stop_signs)}, '
            f'give_way={sorted(self._give_way_signs)} (pause={self._give_way_pause_s}s), '
            f'curve_gate=|err|<= {self._curve_gate_px}px'
        )
        self.get_logger().info(f'Initial state: {self._state.value}')

    # ------------------------------------------------------------------ FSM

    def _color_callback(self, msg: String):
        # Traffic-light obedience is off by default (no real light on this course; the HSV
        # detector false-fires red under changed lighting). When disabled, ignore the light
        # entirely — the FSM stays RUNNING and only the YOLO sign behaviors affect the scale.
        if not self._traffic_light_enabled:
            return
        color = msg.data.lower().strip()
        now = time.monotonic()
        self._last_color_time = now

        if color == self._last_color:
            return   # no change

        prev = self._last_color
        self._last_color = color
        self._transition(color)
        if self._state.value != self._state_for_color(prev):
            self.get_logger().info(
                f'Traffic: {prev} → {color}  |  state → {self._state.value}  '
                f'(scale={self._current_scale(now):.2f})'
            )

    def _transition(self, color: str):
        if color == 'green':
            self._waiting_for_green = False
            self._set_state(TrafficState.RUNNING)

        elif color == 'none':
            # Resume when the light clears, even if we were waiting for green. A red that
            # disappears (light passed, or a false red under bad lighting) must not strand the
            # robot waiting for a green that never comes. Only a sustained red holds the stop
            # (re-asserted each red frame below).
            self._waiting_for_green = False
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

    def _error_callback(self, msg: Float32):
        self._abs_error = abs(float(msg.data))

    def _in_curve(self) -> bool:
        return self._abs_error > self._curve_gate_px

    def _sign_callback(self, msg: String):
        """Route a YOLO sign to slow / stop / give-way intent.

        construccion / GIVE WAY act at the sign's closest point: their area is tracked as a
        per-class running-max and the intent is latched only when the area has peaked then
        dropped below sign_peak_drop_frac × max (past closest), past sign_peak_min_area.
        stopSign is exempt (acts while seen). With sign_peak_enabled=false all revert to the
        prior seen-based latch.
        """
        # /yolo/sign carries "<class>:<area_frac>". Parse both.
        raw = msg.data.strip()
        c, _, area_s = raw.partition(':')
        try:
            area = float(area_s) if area_s else 0.0
        except ValueError:
            area = 0.0
        now = time.monotonic()

        def _past_peak(cls):
            """True once cls's area peaked (>= min) and dropped below drop_frac × max."""
            if not self._sign_peak_enabled:
                return True   # peak disabled → act on sight (legacy)
            # reset running-max if cls had left view (new approach)
            seen = self._peak_seen.get(cls)
            if seen is None or (now - seen) > self._sign_lost_timeout:
                self._peak_max[cls] = 0.0
            self._peak_seen[cls] = now
            self._peak_max[cls] = max(self._peak_max.get(cls, 0.0), area)
            mx = self._peak_max[cls]
            return mx >= self._sign_peak_min_area and area <= self._sign_peak_drop_frac * mx

        if c in self._slow_signs:
            if _past_peak(c):
                self._sign_slow_until = now + self._sign_lost_timeout
        if c in self._stop_signs:
            # stopSign EXEMPT from peak — act while seen.
            self._stop_seen_t = now
            if self._stop_intent_t is None:
                self._stop_intent_t = now      # latch a pending stop (curve-deferred)
        if c in self._give_way_signs:
            self._gw_seen_t = now
            if self._gw_armed and self._gw_intent_t is None and _past_peak(c):
                self._gw_intent_t = now        # latch a pending give-way (curve-deferred)

    def _sign_slow_active(self, now) -> bool:
        return (self._sign_slow_until is not None
                and now < self._sign_slow_until)

    def _stop_sign_seen_recent(self, now) -> bool:
        return (self._stop_seen_t is not None
                and (now - self._stop_seen_t) <= self._lost_timeout)

    def _give_way_seen_recent(self, now) -> bool:
        return (self._gw_seen_t is not None
                and (now - self._gw_seen_t) <= self._lost_timeout)

    def _update_sign_behaviors(self, now):
        """Advance the stopSign / GIVE WAY state machines (called each publish loop)."""
        # ── GIVE WAY: re-arm once the sign has left view ──────────────────────
        if not self._give_way_seen_recent(now):
            self._gw_armed = True
            self._gw_intent_t = None           # drop a stale, never-fired intent
        # Drop a deferred give-way intent that went stale (never got a straight window).
        if (self._gw_intent_t is not None
                and (now - self._gw_intent_t) > self._sign_intent_stale_s):
            self._gw_intent_t = None
        # Fire the brief pause when out of a curve and not already pausing.
        if (self._gw_intent_t is not None and self._gw_pause_until is None
                and self._gw_armed and not self._in_curve()):
            self._gw_pause_until = now + self._give_way_pause_s
            self._gw_armed = False             # once per sighting
            self._gw_intent_t = None
            self.get_logger().info('GIVE WAY: brief stop, then continue.')
        if self._gw_pause_until is not None and now >= self._gw_pause_until:
            self._gw_pause_until = None         # pause over → resume

        # ── stopSign: hold a full stop while seen; resume after gone + confirm ──
        if (self._stop_intent_t is not None
                and (now - self._stop_intent_t) > self._sign_intent_stale_s
                and not self._stop_sign_seen_recent(now)):
            self._stop_intent_t = None          # stale, never-fired intent
        if self._stop_sign_seen_recent(now):
            self._stop_resume_at = None         # still seeing it → no resume yet
            # engage the hold once out of a curve (deferred until straight)
            if not self._stop_active and not self._in_curve():
                self._stop_active = True
                self._stop_intent_t = None
                self.get_logger().info('stopSign: full stop until the sign leaves view.')
        elif self._stop_active:
            # sign no longer seen → start/await the confirm delay, then release
            if self._stop_resume_at is None:
                self._stop_resume_at = now + self._stop_resume_confirm_s
            elif now >= self._stop_resume_at:
                self._stop_active = False
                self._stop_resume_at = None
                self.get_logger().info('stopSign gone — resuming.')

    def _give_way_pausing(self, now) -> bool:
        return self._gw_pause_until is not None and now < self._gw_pause_until

    def _current_scale(self, now) -> float:
        # Highest-priority FULL STOPs first.
        if self._state == TrafficState.STOPPED:   # red light
            return 0.0
        if self._stop_active:                      # stopSign hold
            return 0.0
        if self._give_way_pausing(now):            # GIVE WAY brief pause
            return 0.0
        # Then SLOW: yellow light OR construccion sign.
        if self._state == TrafficState.SLOW or self._sign_slow_active(now):
            return self._yellow_scale
        return 1.0

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

        # Advance the stopSign / GIVE WAY state machines (curve-gated, timed).
        self._update_sign_behaviors(now)

        scale_msg = Float32()
        scale_msg.data = self._current_scale(now)
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
