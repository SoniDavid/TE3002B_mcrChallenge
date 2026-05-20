#!/usr/bin/env python3
"""
Full line-follower stack for Puzzlebot.

Launches:
  micro_ros_agent   — MCU serial bridge
  camera_publisher  — CSI camera with color correction (pzb_camera)
  odometry_node     — wheel odometry integrator (pzb_control)
  velocity_controller — inner PID loop (pzb_control)
  line_follower_node  — image-based line detection + steering (this pkg)

Usage:
    ros2 launch pzb_line_follower line_follower.launch.py
    ros2 launch pzb_line_follower line_follower.launch.py linear_speed:=0.12 Kp_angular:=0.004
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    lf_share     = get_package_share_directory('pzb_line_follower')
    cam_share    = get_package_share_directory('pzb_camera')
    ctrl_share   = get_package_share_directory('pzb_control')

    lf_params    = os.path.join(lf_share,  'config', 'line_follower_params.yaml')
    cam_params   = os.path.join(cam_share, 'config', 'camera_params.yaml')
    cam_info     = os.path.join(cam_share, 'config', 'camera_info_8x5_3cm.yaml')
    ctrl_params  = os.path.join(ctrl_share, 'config', 'pid_vel_params.yaml')

    return LaunchDescription([
        # ── Tunable launch arguments ──────────────────────────────────────────
        DeclareLaunchArgument('linear_speed',  default_value='0.10',
                              description='Forward speed in m/s'),
        DeclareLaunchArgument('Kp_angular',    default_value='0.003',
                              description='Proportional steering gain (rad/s per px)'),
        DeclareLaunchArgument('dead_band_px',  default_value='8',
                              description='Pixel dead band for steering'),
        DeclareLaunchArgument('stop_on_dashed', default_value='false',
                              description='Stop linear motion at intersections'),
        DeclareLaunchArgument('publish_debug', default_value='true',
                              description='Publish debug image topic'),

        # ── MCU bridge ────────────────────────────────────────────────────────
        Node(
            package='micro_ros_agent',
            executable='micro_ros_agent',
            name='micro_ros_agent',
            arguments=['serial', '-D', '/dev/ttyUSB0'],
            output='screen',
        ),

        # ── Camera ───────────────────────────────────────────────────────────
        Node(
            package='pzb_camera',
            executable='camera_publisher',
            name='camera_publisher',
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
        Node(
            package='pzb_line_follower',
            executable='line_follower_node',
            name='line_follower_node',
            output='screen',
            parameters=[
                lf_params,
                {
                    'linear_speed':   LaunchConfiguration('linear_speed'),
                    'Kp_angular':     LaunchConfiguration('Kp_angular'),
                    'dead_band_px':   LaunchConfiguration('dead_band_px'),
                    'stop_on_dashed': LaunchConfiguration('stop_on_dashed'),
                    'publish_debug':  LaunchConfiguration('publish_debug'),
                },
            ],
        ),
    ])
