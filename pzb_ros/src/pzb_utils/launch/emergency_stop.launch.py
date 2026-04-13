#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('duration_s', default_value='3.0'),
        DeclareLaunchArgument('rate_hz', default_value='30.0'),
        Node(
            package='pzb_utils',
            executable='emergency_stop',
            name='emergency_stop',
            output='screen',
            parameters=[{
                'duration_s': LaunchConfiguration('duration_s'),
                'rate_hz': LaunchConfiguration('rate_hz'),
            }],
        ),
    ])
