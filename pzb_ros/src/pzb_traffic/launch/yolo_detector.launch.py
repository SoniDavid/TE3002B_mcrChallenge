#!/usr/bin/env python3
"""
YOLO detector — run on the laptop (GPU).

Usage:
    ros2 launch pzb_traffic yolo_detector.launch.py model_path:=/abs/path/to/model.pt

Optional overrides:
    conf_threshold:=0.45
    device:=cuda          # or 'cpu'
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share   = get_package_share_directory('pzb_traffic')
    yolo_params = os.path.join(pkg_share, 'config', 'yolo_params.yaml')

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
    ])
