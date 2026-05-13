#!/usr/bin/env python3
"""
Traffic challenge — PC side.

Run this on the PC. Run traffic_challenge_robot.launch.py on the Jetson first.
NEVER run this alongside traffic_challenge.launch.py (all-in-one).

Requires from Jetson (same ROS_DOMAIN_ID):
  /camera/image_compressed  — JPEG frames from usb_camera_publisher
  /cmd_vel                  — sent to MCU via micro_ros_agent
  /robot_vel, /VelocityEncR, /VelocityEncL — from MCU via micro_ros_agent

Launches:
  odometry_node          — integrates /robot_vel → /odom
  velocity_controller    — inner-loop PI velocity servo
  waypoint_follower      — outer-loop position controller
  twist_slew_limiter     — acceleration limiter (optional, default on)
  color_detector_node    — HSV traffic light detection → /traffic_light_color
  traffic_light_fsm_node — FSM → /traffic_speed_scale

Usage:
    ros2 launch pzb_traffic traffic_challenge_pc.launch.py
    ros2 launch pzb_traffic traffic_challenge_pc.launch.py \
        params_file:=/path/to/my_mission.yaml \
        traffic_params:=/path/to/my_traffic.yaml
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution([
        FindPackageShare('pzb_control'), 'config', 'pid_vel_params.yaml'])
    default_traffic = PathJoinSubstitution([
        FindPackageShare('pzb_traffic'), 'config', 'traffic_params.yaml'])

    params_arg    = DeclareLaunchArgument('params_file',       default_value=default_params)
    traffic_arg   = DeclareLaunchArgument('traffic_params',    default_value=default_traffic)
    sim_arg       = DeclareLaunchArgument('use_sim_time',      default_value='false')
    use_slew_arg  = DeclareLaunchArgument('use_slew_limiter',  default_value='true')
    accel_v_arg   = DeclareLaunchArgument('max_linear_accel',  default_value='0.20')
    accel_w_arg   = DeclareLaunchArgument('max_angular_accel', default_value='1.20')

    params         = LaunchConfiguration('params_file')
    traffic_params = LaunchConfiguration('traffic_params')
    sim            = LaunchConfiguration('use_sim_time')
    use_slew       = LaunchConfiguration('use_slew_limiter')
    accel_v        = LaunchConfiguration('max_linear_accel')
    accel_w        = LaunchConfiguration('max_angular_accel')

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

    wp_node_slew = Node(
        package='pzb_control',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        condition=IfCondition(use_slew),
        parameters=[params, {'use_sim_time': sim}],
        remappings=[('/cmd_vel_desired', '/cmd_vel_desired_raw')],
    )

    wp_node_direct = Node(
        package='pzb_control',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        condition=UnlessCondition(use_slew),
        parameters=[params, {'use_sim_time': sim}],
    )

    slew_node = Node(
        package='pzb_utils',
        executable='twist_slew_limiter',
        name='twist_slew_limiter',
        output='screen',
        condition=IfCondition(use_slew),
        parameters=[{
            'input_topic':          '/cmd_vel_desired_raw',
            'output_topic':         '/cmd_vel_desired',
            'loop_hz':              50.0,
            'max_linear_accel':     accel_v,
            'max_angular_accel':    accel_w,
            'max_linear_speed':     0.40,
            'max_angular_speed':    1.50,
            'cmd_timeout_s':        0.50,
            'initial_linear_speed': 0.0,
            'initial_angular_speed': 0.0,
            'use_sim_time':         sim,
        }],
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
        params_arg,
        traffic_arg,
        sim_arg,
        use_slew_arg,
        accel_v_arg,
        accel_w_arg,
        odom_node,
        vel_ctrl_node,
        wp_node_slew,
        wp_node_direct,
        slew_node,
        color_detector,
        traffic_fsm,
    ])
