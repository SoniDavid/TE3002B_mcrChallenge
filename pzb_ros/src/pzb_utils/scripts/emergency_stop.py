#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32


class EmergencyStop(Node):
    def __init__(self):
        super().__init__('emergency_stop')
        self.declare_parameter('duration_s', 3.0)
        self.declare_parameter('rate_hz', 30.0)

        self.duration_s = float(self.get_parameter('duration_s').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        if self.duration_s <= 0.0 or self.rate_hz <= 0.0:
            raise ValueError('duration_s and rate_hz must be > 0')

        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_vel_r = self.create_publisher(Float32, 'VelocitySetR', 10)
        self.pub_vel_l = self.create_publisher(Float32, 'VelocitySetL', 10)

        self.deadline = self.get_clock().now().nanoseconds * 1e-9 + self.duration_s
        self.timer = self.create_timer(1.0 / self.rate_hz, self.stop_loop)
        self.get_logger().warn('Emergency stop active: publishing zero commands.')

    def stop_loop(self):
        tw = Twist()
        tw.linear.x = 0.0
        tw.angular.z = 0.0
        self.pub_cmd_vel.publish(tw)

        right = Float32()
        left = Float32()
        right.data = 0.0
        left.data = 0.0
        self.pub_vel_r.publish(right)
        self.pub_vel_l.publish(left)

        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s >= self.deadline:
            self.get_logger().warn('Emergency stop complete.')
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = EmergencyStop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()