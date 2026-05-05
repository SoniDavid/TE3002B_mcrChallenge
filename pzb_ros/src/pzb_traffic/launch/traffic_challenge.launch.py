#!/usr/bin/env python3
"""
Full traffic-light challenge stack.

Launches:
  micro_ros_agent     — MCU serial bridge
  camera_publisher    — IMX219 CSI camera
  odometry_node       — integrates /robot_vel → /odom
  velocity_controller — inner-loop PI
  waypoint_follower   — outer-loop position controller
  color_detector_node — HSV traffic light color detection
  traffic_light_fsm_node — behavioral FSM, publishes /traffic_speed_scale

Usage:
    ros2 launch pzb_traffic traffic_challenge.launch.py
    ros2 launch pzb_traffic traffic_challenge.launch.py \
        control_params:=/path/to/my_params.yaml \
        traffic_params:=/path/to/my_traffic.yaml
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
    camera_pkg   = get_package_share_directory('pzb_camera')

    default_control_params = os.path.join(control_pkg, 'config', 'pid_vel_params.yaml')
    default_traffic_params = os.path.join(traffic_pkg,  'config', 'traffic_params.yaml')
    default_camera_params  = os.path.join(camera_pkg,   'config', 'camera_params.yaml')

    control_params_arg = DeclareLaunchArgument(
        'control_params', default_value=default_control_params,
        description='Path to control stack YAML config')
    traffic_params_arg = DeclareLaunchArgument(
        'traffic_params', default_value=default_traffic_params,
        description='Path to traffic light YAML config')
    camera_params_arg = DeclareLaunchArgument(
        'camera_params', default_value=default_camera_params,
        description='Path to camera YAML config')
    sim_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    control_params = LaunchConfiguration('control_params')
    traffic_params = LaunchConfiguration('traffic_params')
    camera_params  = LaunchConfiguration('camera_params')
    sim            = LaunchConfiguration('use_sim_time')

    micro_ros = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=['serial', '-D', '/dev/ttyUSB0'],
        output='screen',
    )

    camera_node = Node(
        package='pzb_camera',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[camera_params, {'use_sim_time': sim}],
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
        camera_params_arg,
        sim_arg,
        micro_ros,
        camera_node,
        odom_node,
        vel_ctrl_node,
        wp_node,
        color_detector,
        traffic_fsm,
    ])
