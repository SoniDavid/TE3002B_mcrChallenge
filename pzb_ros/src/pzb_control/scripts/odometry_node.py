#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

# MCU publishes with BEST_EFFORT
_MCU_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')

        self.declare_parameter('publish_tf', True)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_pose_covariance', 0.01)
        self.declare_parameter('odom_twist_covariance', 0.01)

        self._publish_tf = self.get_parameter('publish_tf').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._pose_cov = self.get_parameter('odom_pose_covariance').value
        self._twist_cov = self.get_parameter('odom_twist_covariance').value

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._last_time = None
        self._v = 0.0
        self._omega = 0.0

        # /robot_vel is the MCU's own body-velocity estimate (already in m/s, rad/s)
        self.create_subscription(TwistStamped, '/robot_vel', self._cb_robot_vel, _MCU_QOS)

        self._odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f'OdometryNode ready — integrating /robot_vel, tf={self._publish_tf}'
        )

    def _cb_robot_vel(self, msg: TwistStamped):
        now = self.get_clock().now()

        self._v = msg.twist.linear.x
        self._omega = msg.twist.angular.z

        if self._last_time is None:
            self._last_time = now
            return

        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now

        if dt <= 0.0 or dt > 0.5:
            return

        self._x += self._v * math.cos(self._theta) * dt
        self._y += self._v * math.sin(self._theta) * dt
        self._theta = _wrap_to_pi(self._theta + self._omega * dt)

        self._publish_odom(now)
        if self._publish_tf:
            self._broadcast_tf(now)

    def _publish_odom(self, now):
        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame

        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        msg.pose.pose.orientation.z = math.sin(self._theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(self._theta / 2.0)

        msg.twist.twist.linear.x = self._v
        msg.twist.twist.angular.z = self._omega

        cov_p = [0.0] * 36
        cov_p[0] = self._pose_cov
        cov_p[7] = self._pose_cov
        cov_p[35] = self._pose_cov
        msg.pose.covariance = cov_p

        cov_t = [0.0] * 36
        cov_t[0] = self._twist_cov
        cov_t[35] = self._twist_cov
        msg.twist.covariance = cov_t

        self._odom_pub.publish(msg)

    def _broadcast_tf(self, now):
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self._odom_frame
        t.child_frame_id = self._base_frame
        t.transform.translation.x = self._x
        t.transform.translation.y = self._y
        t.transform.rotation.z = math.sin(self._theta / 2.0)
        t.transform.rotation.w = math.cos(self._theta / 2.0)
        self._tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
