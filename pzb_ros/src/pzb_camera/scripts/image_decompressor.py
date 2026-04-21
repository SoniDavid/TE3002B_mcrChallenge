#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class ImageDecompressor(Node):
    def __init__(self):
        super().__init__('image_decompressor')

        self.declare_parameter('input_topic', '/camera/image_compressed')
        self.declare_parameter('output_topic', '/camera/image_raw_decompressed')
        self.declare_parameter('frame_id_override', '')

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self._frame_id_override = str(self.get_parameter('frame_id_override').value)

        self._bridge = CvBridge()

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub = self.create_subscription(
            CompressedImage,
            input_topic,
            self._cb,
            sub_qos,
        )
        self._pub = self.create_publisher(Image, output_topic, pub_qos)

        self.get_logger().info(
            f'ImageDecompressor ready: {input_topic} -> {output_topic}'
        )

    def _cb(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('Failed to decode compressed frame')
            return

        out = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._frame_id_override or msg.header.frame_id
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImageDecompressor()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
