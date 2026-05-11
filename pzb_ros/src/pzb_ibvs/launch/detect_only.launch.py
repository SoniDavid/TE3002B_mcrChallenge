#!/usr/bin/env python3
"""
Detection-only launch: camera + visual_detector_node.
No MPC, no velocity controller — robot will NOT move.

Usage:
  ros2 launch pzb_ibvs detect_only.launch.py
  ros2 launch pzb_ibvs detect_only.launch.py detector_type:=aruco
  ros2 launch pzb_ibvs detect_only.launch.py camera_type:=usb device_index:=2

Verify detection:
  ros2 topic echo /visual_features                                      # [eu, ev, ea, confidence]
  ros2 run rqt_image_view rqt_image_view /visual_detector/debug_image  # annotated camera feed
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ibvs_params = PathJoinSubstitution([
        FindPackageShare('pzb_ibvs'), 'config', 'mpc_ibvs_params.yaml',
    ])
    cam_params = PathJoinSubstitution([
        FindPackageShare('pzb_camera'), 'config', 'camera_params.yaml',
    ])

    params_arg       = DeclareLaunchArgument('params_file',   default_value=ibvs_params)
    detector_arg     = DeclareLaunchArgument('detector_type', default_value='color_blob',
                                             description="'color_blob' or 'aruco'")
    camera_type_arg  = DeclareLaunchArgument('camera_type',  default_value='csi',
                                             description="'csi' (Jetson native) or 'usb'")
    device_index_arg = DeclareLaunchArgument('device_index', default_value='2',
                                             description='USB /dev/videoN index')

    params        = LaunchConfiguration('params_file')
    detector_type = LaunchConfiguration('detector_type')
    camera_type   = LaunchConfiguration('camera_type')
    device_index  = LaunchConfiguration('device_index')

    is_usb = PythonExpression(["'", camera_type, "' == 'usb'"])

    csi_camera_node = Node(
        package='pzb_camera',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[cam_params],
        condition=UnlessCondition(is_usb),
    )

    usb_camera_node = Node(
        package='pzb_camera',
        executable='usb_camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[{
            'device_index':       device_index,
            'width':              1280,
            'height':             720,
            'framerate':          30.0,
            'jpeg_quality':       80,
            'publish_compressed': True,
            'publish_raw':        False,
        }],
        condition=IfCondition(is_usb),
    )

    detector_node = Node(
        package='pzb_ibvs',
        executable='visual_detector_node',
        name='visual_detector_node',
        output='screen',
        parameters=[params, {'detector_type': detector_type}],
    )

    return LaunchDescription([
        params_arg,
        detector_arg,
        camera_type_arg,
        device_index_arg,
        csi_camera_node,
        usb_camera_node,
        detector_node,
    ])
