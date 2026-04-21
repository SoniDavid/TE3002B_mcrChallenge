#!/usr/bin/env python3
"""
Full MPC-IBVS stack launch file.

Brings up:
  1. camera_publisher        (pzb_camera)
  2. odometry_node           (pzb_control)
  3. velocity_controller     (pzb_control)
  4. visual_detector_node    (pzb_ibvs)
  5. mpc_ibvs_node           (pzb_ibvs)

The MPC outputs /cmd_vel_desired which feeds directly into the existing
velocity_controller inner loop.

Usage:
  ros2 launch pzb_ibvs mpc_ibvs.launch.py
  ros2 launch pzb_ibvs mpc_ibvs.launch.py detector_type:=aruco
  ros2 launch pzb_ibvs mpc_ibvs.launch.py params_file:=/path/to/custom.yaml
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ibvs_params = PathJoinSubstitution([
        FindPackageShare('pzb_ibvs'), 'config', 'mpc_ibvs_params.yaml',
    ])
    ctrl_params = PathJoinSubstitution([
        FindPackageShare('pzb_control'), 'config', 'pid_vel_params.yaml',
    ])
    cam_params = PathJoinSubstitution([
        FindPackageShare('pzb_camera'), 'config', 'camera_params.yaml',
    ])

    ibvs_params_arg = DeclareLaunchArgument('params_file', default_value=ibvs_params)
    ctrl_params_arg = DeclareLaunchArgument('ctrl_params_file', default_value=ctrl_params)
    sim_arg = DeclareLaunchArgument('use_sim_time', default_value='false')
    detector_arg = DeclareLaunchArgument('detector_type', default_value='color_blob')

    params = LaunchConfiguration('params_file')
    ctrl_params_cfg = LaunchConfiguration('ctrl_params_file')
    sim = LaunchConfiguration('use_sim_time')
    detector_type = LaunchConfiguration('detector_type')

    camera_node = Node(
        package='pzb_camera',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[cam_params, {'use_sim_time': sim}],
    )

    odom_node = Node(
        package='pzb_control',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[ctrl_params_cfg, {'use_sim_time': sim}],
    )

    vel_ctrl_node = Node(
        package='pzb_control',
        executable='velocity_controller',
        name='velocity_controller',
        output='screen',
        parameters=[ctrl_params_cfg, {'use_sim_time': sim}],
    )

    detector_node = Node(
        package='pzb_ibvs',
        executable='visual_detector_node',
        name='visual_detector_node',
        output='screen',
        parameters=[params, {'use_sim_time': sim, 'detector_type': detector_type}],
    )

    mpc_node = Node(
        package='pzb_ibvs',
        executable='mpc_ibvs_node',
        name='mpc_ibvs_node',
        output='screen',
        parameters=[params, {'use_sim_time': sim}],
    )

    return LaunchDescription([
        ibvs_params_arg,
        ctrl_params_arg,
        sim_arg,
        detector_arg,
        camera_node,
        odom_node,
        vel_ctrl_node,
        detector_node,
        mpc_node,
    ])
