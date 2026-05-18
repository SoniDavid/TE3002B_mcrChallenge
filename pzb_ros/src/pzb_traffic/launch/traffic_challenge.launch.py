#!/usr/bin/env python3
"""
DEPRECATED — use the split pair instead:

  Jetson:  ros2 launch pzb_traffic traffic_challenge_robot.launch.py
  PC:      ros2 launch pzb_traffic traffic_challenge_pc.launch.py

Running this all-in-one file alongside either of the above will put duplicate
nodes on the same topics and cause MCU brown-outs (see memory for root cause).

This file is kept only as a fallback for single-machine testing.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    control_pkg = get_package_share_directory('pzb_control')
    traffic_pkg  = get_package_share_directory('pzb_traffic')

    default_control_params = os.path.join(control_pkg, 'config', 'pid_vel_params.yaml')
    default_traffic_params = os.path.join(traffic_pkg,  'config', 'traffic_params.yaml')

    control_params_arg = DeclareLaunchArgument(
        'control_params', default_value=default_control_params)
    traffic_params_arg = DeclareLaunchArgument(
        'traffic_params', default_value=default_traffic_params)
    sim_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    control_params = LaunchConfiguration('control_params')
    traffic_params = LaunchConfiguration('traffic_params')
    sim            = LaunchConfiguration('use_sim_time')

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
            'device_index':       0,
            'width':              640,
            'height':             480,
            'framerate':          30.0,
            'publish_compressed': True,
            'publish_raw':        False,
        }],
    )

    odom_node = Node(
        package='pzb_control',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[control_params, {'use_sim_time': sim}],
    )

    vel_ctrl_node = Node(
        package='pzb_control',
        executable='velocity_controller',
        name='velocity_controller',
        output='screen',
        parameters=[control_params, {'use_sim_time': sim}],
    )

    wp_node = Node(
        package='pzb_control',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[control_params, {'use_sim_time': sim}],
    )

    color_detector = Node(
        package='pzb_traffic',
        executable='color_detector_node',
        name='color_detector_node',
        output='screen',
        parameters=[traffic_params, {'use_sim_time': sim}],
    )

    traffic_fsm = Node(
        package='pzb_traffic',
        executable='traffic_light_fsm_node',
        name='traffic_light_fsm_node',
        output='screen',
        parameters=[traffic_params, {'use_sim_time': sim}],
    )

    return LaunchDescription([
        control_params_arg,
        traffic_params_arg,
        sim_arg,
        micro_ros,
        camera,
        odom_node,
        vel_ctrl_node,
        wp_node,
        color_detector,
        traffic_fsm,
    ])
