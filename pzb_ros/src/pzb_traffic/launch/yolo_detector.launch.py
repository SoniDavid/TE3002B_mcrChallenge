#!/usr/bin/env python3
"""
Off-board PERCEPTION — run on the laptop (GPU). Started by scripts/run_yolo.sh.

Both nodes here consume the Jetson's /camera/image_compressed (JPEG) over the shared ROS
domain — that's why they run on the PC, not the Jetson:
  yolo_detector_node   — YOLO signs → /yolo/sign, /yolo/turn_sign
  color_detector_node  — HSV traffic-light color → /traffic_light_color  (moved off the
      Jetson: it now reads /camera/image_compressed + a tight top-center ROI). The traffic
      FSM (/traffic_light_color + /yolo/sign → /traffic_speed_scale) stays on the JETSON
      (it's light) — see line_follower.launch.py.

Usage:
    ros2 launch pzb_traffic yolo_detector.launch.py model_path:=/abs/path/to/model.pt

Optional overrides:
    conf_threshold:=0.45
    device:=cuda          # or 'cpu'
    use_color:=false      # disable the HSV traffic-light color detector
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share      = get_package_share_directory('pzb_traffic')
    yolo_params    = os.path.join(pkg_share, 'config', 'yolo_params.yaml')
    traffic_params = os.path.join(pkg_share, 'config', 'traffic_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/soni/Documents/classes/IRS_6to/manchesterRobotics/TE3002B_mcrChallenge/best.pt',
            description='Absolute path to the YOLOv8 .pt weights file'),
        DeclareLaunchArgument(
            'conf_threshold', default_value='0.55',
            description='Minimum detection confidence (0–1)'),
        DeclareLaunchArgument(
            'device', default_value='cuda',
            description='Inference device: "cuda" or "cpu"'),
        DeclareLaunchArgument(
            'use_color', default_value='true',
            description='Launch the HSV traffic-light color detector (→ /traffic_light_color)'),

        Node(
            package='pzb_traffic',
            executable='yolo_detector_node',
            name='yolo_detector_node',
            output='screen',
            parameters=[
                yolo_params,
                {
                    'model_path':     LaunchConfiguration('model_path'),
                    'conf_threshold': LaunchConfiguration('conf_threshold'),
                    'device':         LaunchConfiguration('device'),
                },
            ],
        ),

        # HSV traffic-light color detector — PC side (reads /camera/image_compressed).
        Node(
            package='pzb_traffic',
            executable='color_detector_node',
            name='color_detector_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_color')),
            parameters=[traffic_params],
        ),
    ])
