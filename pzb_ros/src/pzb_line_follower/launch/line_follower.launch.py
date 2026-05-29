#!/usr/bin/env python3
"""
Full line-follower stack for Puzzlebot.

Launches:
  micro_ros_agent      — MCU serial bridge
  camera_publisher     — CSI camera with color correction (pzb_camera)
  odometry_node        — wheel odometry integrator (pzb_control)
  velocity_controller  — inner PI feedback loop (pzb_control)
  twist_slew_limiter   — acceleration rate-limiter, prevents MCU brown-out (pzb_utils)
  line_follower_node   — image-based line detection + PD steering (this pkg)
  color_detector_node  — traffic light HSV detector (pzb_traffic, if use_traffic=true)
  traffic_light_fsm_node — speed scale FSM (pzb_traffic, if use_traffic=true)

Topic chain:
  line_follower_node → /cmd_vel_desired_raw
    → twist_slew_limiter → /cmd_vel_desired
    → velocity_controller → /cmd_vel → MCU

Usage:
    ros2 launch pzb_line_follower line_follower.launch.py
    ros2 launch pzb_line_follower line_follower.launch.py linear_speed:=0.12 Kp_angular:=0.004
    ros2 launch pzb_line_follower line_follower.launch.py use_traffic:=false
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    lf_share      = get_package_share_directory('pzb_line_follower')
    cam_share     = get_package_share_directory('pzb_camera')
    ctrl_share    = get_package_share_directory('pzb_control')
    traffic_share = get_package_share_directory('pzb_traffic')

    lf_params      = os.path.join(lf_share,      'config', 'line_follower_params.yaml')
    cam_params     = os.path.join(cam_share,     'config', 'camera_params.yaml')
    cam_info       = os.path.join(cam_share,     'config', 'camera_info_8x5_3cm.yaml')
    ctrl_params    = os.path.join(ctrl_share,    'config', 'pid_vel_params.yaml')
    traffic_params = os.path.join(traffic_share, 'config', 'traffic_params.yaml')

    return LaunchDescription([
        # ── Tunable launch arguments ──────────────────────────────────────────
        DeclareLaunchArgument('linear_speed',  default_value='0.08',
                              description='Forward speed in m/s'),
        DeclareLaunchArgument('Kp_angular',    default_value='0.005',
                              description='Proportional steering gain (rad/s per px)'),
        DeclareLaunchArgument('Kd_angular',    default_value='0.001',
                              description='Derivative steering gain (rad/s per px/frame); 0 disables'),
        DeclareLaunchArgument('dead_band_px',  default_value='8',
                              description='Pixel dead band for steering'),
        DeclareLaunchArgument('stop_on_dashed', default_value='false',
                              description='Stop linear motion at intersections'),
        DeclareLaunchArgument('publish_debug', default_value='false',
                              description='Publish debug image topic'),
        DeclareLaunchArgument('use_traffic',       default_value='true',
                              description='Launch traffic light detector and FSM'),
        DeclareLaunchArgument('curve_speed_reduction', default_value='0.75',
                              description='Speed reduction on turns: 0=off, 1=stop at max angular'),
        DeclareLaunchArgument('min_linear_speed',  default_value='0.05',
                              description='Floor linear speed in m/s (keep above motor deadband)'),
        DeclareLaunchArgument('max_linear_accel',  default_value='0.5',
                              description='Max linear acceleration for slew limiter (m/s²)'),
        DeclareLaunchArgument('max_angular_accel', default_value='1.20',
                              description='Max angular acceleration for slew limiter (rad/s²)'),
        DeclareLaunchArgument('sharp_turn_threshold_px', default_value='80',
                              description='|error| in px above which sharp-turn slow mode activates'),
        DeclareLaunchArgument('sharp_turn_speed',  default_value='0.03',
                              description='Linear speed (m/s) during sharp turns — robot slows to spin'),

        # # ── MCU bridge ────────────────────────────────────────────────────────
        # Node(
        #     package='micro_ros_agent',
        #     executable='micro_ros_agent',
        #     name='micro_ros_agent',
        #     arguments=['serial', '-D', '/dev/ttyUSB0'],
        #     output='screen',
        # ),

        # ── Camera ───────────────────────────────────────────────────────────
        Node(
            package='pzb_camera',
            executable='camera_raw_publisher',
            name='camera_raw_publisher',
            output='screen',
            parameters=[
                cam_params,
                {
                    'camera_info_file':    cam_info,
                    'publish_camera_info': True,
                },
            ],
        ),

        # ── Control stack ─────────────────────────────────────────────────────
        Node(
            package='pzb_control',
            executable='odometry_node',
            name='odometry_node',
            output='screen',
            parameters=[ctrl_params],
        ),
        Node(
            package='pzb_control',
            executable='velocity_controller',
            name='velocity_controller',
            output='screen',
            parameters=[ctrl_params],
        ),

        # ── Line follower ─────────────────────────────────────────────────────
        # Publishes to /cmd_vel_desired_raw; twist_slew_limiter forwards to /cmd_vel_desired
        Node(
            package='pzb_line_follower',
            executable='line_follower_node',
            name='line_follower_node',
            output='screen',
            parameters=[
                lf_params,
                {
                    'linear_speed':          LaunchConfiguration('linear_speed'),
                    'Kp_angular':            LaunchConfiguration('Kp_angular'),
                    'Kd_angular':            LaunchConfiguration('Kd_angular'),
                    'dead_band_px':          LaunchConfiguration('dead_band_px'),
                    'stop_on_dashed':        LaunchConfiguration('stop_on_dashed'),
                    'publish_debug':         LaunchConfiguration('publish_debug'),
                    'curve_speed_reduction':   LaunchConfiguration('curve_speed_reduction'),
                    'min_linear_speed':        LaunchConfiguration('min_linear_speed'),
                    'sharp_turn_threshold_px': LaunchConfiguration('sharp_turn_threshold_px'),
                    'sharp_turn_speed':        LaunchConfiguration('sharp_turn_speed'),
                    'topic_cmd_vel':           '/cmd_vel_desired_raw',
                },
            ],
        ),

        # ── Slew limiter (acceleration rate-limiting — prevents MCU brown-out) ──
        # Sits between /cmd_vel_desired_raw and /cmd_vel_desired
        Node(
            package='pzb_utils',
            executable='twist_slew_limiter',
            name='twist_slew_limiter',
            output='screen',
            parameters=[{
                'input_topic':        '/cmd_vel_desired_raw',
                'output_topic':       '/cmd_vel_desired',
                'max_linear_accel':   LaunchConfiguration('max_linear_accel'),
                'max_angular_accel':  LaunchConfiguration('max_angular_accel'),
                'max_linear_speed':   0.20,
                'max_angular_speed':  0.80,
            }],
        ),

        # ── Traffic light detection + FSM ─────────────────────────────────────
        Node(
            package='pzb_traffic',
            executable='color_detector_node',
            name='color_detector_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_traffic')),
            parameters=[traffic_params],
        ),
        Node(
            package='pzb_traffic',
            executable='traffic_light_fsm_node',
            name='traffic_light_fsm_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_traffic')),
            parameters=[traffic_params],
        ),
    ])
