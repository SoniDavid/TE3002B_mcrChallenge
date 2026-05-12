#!/usr/bin/env python3
"""
ROBOT-SIDE launch file (run on the Jetson).

Brings up hardware-dependent nodes that must live on the robot:
  1. micro_ros_agent   — serial bridge to the ESP/MCU (/dev/ttyUSB0)
  2. camera_publisher  — CSI or USB camera → /camera/image_compressed
  3. odometry_node     — wheel encoder odometry from MCU topics
  4. velocity_controller — inner-loop PI: /cmd_vel_desired → /cmd_vel → MCU

The PC-side launch (mpc_ibvs_pc.launch.py) runs the vision detector and MPC
node. Both machines must be on the same ROS2 network (ROS_DOMAIN_ID must match).

Usage:
  ros2 launch pzb_ibvs mpc_ibvs_robot.launch.py
  ros2 launch pzb_ibvs mpc_ibvs_robot.launch.py camera_type:=usb device_index:=1
  ros2 launch pzb_ibvs mpc_ibvs_robot.launch.py ctrl_params_file:=/path/to/custom.yaml
""" 
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ctrl_params = PathJoinSubstitution([
        FindPackageShare('pzb_control'), 'config', 'pid_vel_params.yaml',
    ])
    cam_params = PathJoinSubstitution([
        FindPackageShare('pzb_camera'), 'config', 'camera_params.yaml',
    ])

    ctrl_params_arg  = DeclareLaunchArgument('ctrl_params_file', default_value=ctrl_params)
    sim_arg          = DeclareLaunchArgument('use_sim_time',      default_value='false')
    camera_type_arg  = DeclareLaunchArgument('camera_type',       default_value='csi',
                                             description="'csi' (Jetson native) or 'usb'")
    device_index_arg = DeclareLaunchArgument('device_index',      default_value='1',
                                             description='USB /dev/videoN index')

    ctrl_params_cfg = LaunchConfiguration('ctrl_params_file')
    sim             = LaunchConfiguration('use_sim_time')
    camera_type     = LaunchConfiguration('camera_type')
    device_index    = LaunchConfiguration('device_index')

    # ── 1. micro-ROS bridge ────────────────────────────────────────────────────
    micro_ros_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        output='screen',
        arguments=['serial', '--dev', '/dev/ttyUSB0'],
    )

    # ── 2. Camera ─────────────────────────────────────────────────────────────
    is_usb = PythonExpression(["'", camera_type, "' == 'usb'"])

    csi_camera_node = Node(
        package='pzb_camera',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[cam_params, {'use_sim_time': sim}],
        condition=UnlessCondition(is_usb),
    )

    usb_camera_node = Node(
        package='pzb_camera',
        executable='usb_camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[{
            'device_index':       device_index,
            'width':              640,
            'height':             480,
            'framerate':          30.0,
            'jpeg_quality':       80,
            'publish_compressed': True,
            'publish_raw':        False,
        }],
        condition=IfCondition(is_usb),
    )

    # 3 AND 4 RUNNING ON PC FOR EASIER SETUP 
    # ── 3. Odometry ───────────────────────────────────────────────────────────
    odom_node = Node(
        package='pzb_control',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[ctrl_params_cfg, {'use_sim_time': sim}],
    )

    # ── 4. Velocity controller (inner PI loop) ────────────────────────────────
    vel_ctrl_node = Node(
        package='pzb_control',
        executable='velocity_controller',
        name='velocity_controller',
        output='screen',
        parameters=[ctrl_params_cfg, {'use_sim_time': sim}],
    )


    return LaunchDescription([
        ctrl_params_arg,
        sim_arg,
        camera_type_arg,
        device_index_arg,
        micro_ros_node,
        csi_camera_node,
        usb_camera_node,
        # odom_node, # RUNNING ON PC FOR EASIER SETUP
        # vel_ctrl_node,

    ])
