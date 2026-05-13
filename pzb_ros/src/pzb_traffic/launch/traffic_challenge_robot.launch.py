#!/usr/bin/env python3
"""
Traffic challenge — ROBOT (Jetson) side.

Run this on the Jetson. Run traffic_challenge_pc.launch.py on the PC.
NEVER run this alongside traffic_challenge.launch.py (all-in-one) — doing so
puts two camera publishers and two velocity controllers on the same topics.

Launches:
  micro_ros_agent      — MCU serial bridge (/dev/ttyUSB0)
  usb_camera_publisher — USB camera → /camera/image_compressed (JPEG, no raw)

Usage:
    ros2 launch pzb_traffic traffic_challenge_robot.launch.py
    ros2 launch pzb_traffic traffic_challenge_robot.launch.py device_index:=2
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    device_arg = DeclareLaunchArgument(
        'device_index', default_value='0',
        description='V4L2 device index — 0 for /dev/video0')
    width_arg = DeclareLaunchArgument(
        'width', default_value='640',
        description='Capture width in pixels')
    height_arg = DeclareLaunchArgument(
        'height', default_value='480',
        description='Capture height in pixels')
    fps_arg = DeclareLaunchArgument(
        'framerate', default_value='30.0',
        description='Target framerate')
    quality_arg = DeclareLaunchArgument(
        'jpeg_quality', default_value='80',
        description='JPEG compression quality (1–100)')

    micro_ros = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=['serial', '-D', '/dev/ttyUSB0'],
        output='screen',
    )

    camera = Node(
        package='pzb_camera',
        executable='usb_camera_publisher',
        name='usb_camera_publisher',
        output='screen',
        parameters=[{
            'device_index':       LaunchConfiguration('device_index'),
            'width':              LaunchConfiguration('width'),
            'height':             LaunchConfiguration('height'),
            'framerate':          LaunchConfiguration('framerate'),
            'jpeg_quality':       LaunchConfiguration('jpeg_quality'),
            'publish_compressed': True,
            'publish_raw':        False,   # raw is ~30 MB/s over WiFi; PC only needs JPEG
        }],
    )

    return LaunchDescription([
        device_arg,
        width_arg,
        height_arg,
        fps_arg,
        quality_arg,
        micro_ros,
        camera,
    ])
