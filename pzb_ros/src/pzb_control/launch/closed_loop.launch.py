#!/usr/bin/env python3
"""
Launch odometry_node + velocity_controller.

Use this when doing closed-loop teleoperation:
  - Run teleop_twist_keyboard and remap /cmd_vel to /cmd_vel_desired, OR
  - Publish directly to /cmd_vel_desired from any node.

The velocity controller tracks the desired body velocity and drives
/VelocitySetR and /VelocitySetL on the MCU.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution([
        FindPackageShare('pzb_control'),
        'config',
        'closed_loop_params.yaml',
    ])

    params_arg = DeclareLaunchArgument('params_file', default_value=default_params)
    sim_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    params = LaunchConfiguration('params_file')
    sim = LaunchConfiguration('use_sim_time')

    odom_node = Node(
        package='pzb_control',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[params, {'use_sim_time': sim}],
    )

    vel_ctrl_node = Node(
        package='pzb_control',
        executable='velocity_controller',
        name='velocity_controller',
        output='screen',
        parameters=[params, {'use_sim_time': sim}],
    )

    return LaunchDescription([params_arg, sim_arg, odom_node, vel_ctrl_node])
