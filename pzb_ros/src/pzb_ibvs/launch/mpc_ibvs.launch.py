#!/usr/bin/env python3
"""
SINGLE-MACHINE MPC-IBVS stack launch file (everything on one box).

Brings up the full chain:
  1. micro_ros_agent       — bridges /dev/ttyUSB0 → ROS2 topics
  2. camera_publisher      — CSI (Jetson) or USB, 640x480
  3. visual_detector_node  — /camera/image_compressed → /visual_features
  4. mpc_ibvs_node         — Linear MPC; publishes /cmd_vel_desired_raw
  5. twist_slew_limiter    — rate-limits /cmd_vel_desired_raw → /cmd_vel_desired
  6. odometry_node         — wheel odometry from /robot_vel
  7. velocity_controller   — inner PI: /cmd_vel_desired → /cmd_vel → MCU

WARNING: MUTUALLY EXCLUSIVE with the pc/robot split launches. Run EITHER this
    file alone, OR `mpc_ibvs_robot.launch.py` (Jetson) + `mpc_ibvs_pc.launch.py`
    (PC) -- NEVER both. Two mpc_ibvs_node instances publishing to
    /cmd_vel_desired make the motor command thrash and brown out the MCU.

Usage:
  ros2 launch pzb_ibvs mpc_ibvs.launch.py
  ros2 launch pzb_ibvs mpc_ibvs.launch.py detector_type:=aruco
  ros2 launch pzb_ibvs mpc_ibvs.launch.py camera_type:=usb device_index:=2
  ros2 launch pzb_ibvs mpc_ibvs.launch.py params_file:=/path/to/custom.yaml
"""
import os

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
    ctrl_params = PathJoinSubstitution([
        FindPackageShare('pzb_control'), 'config', 'pid_vel_params.yaml',
    ])
    cam_params = PathJoinSubstitution([
        FindPackageShare('pzb_camera'), 'config', 'camera_params.yaml',
    ])

    ibvs_params_arg  = DeclareLaunchArgument('params_file',      default_value=ibvs_params)
    ctrl_params_arg  = DeclareLaunchArgument('ctrl_params_file',  default_value=ctrl_params)
    sim_arg          = DeclareLaunchArgument('use_sim_time',      default_value='false')
    detector_arg     = DeclareLaunchArgument('detector_type',     default_value='color_blob',
                                             description="'color_blob' or 'aruco'")
    camera_type_arg  = DeclareLaunchArgument('camera_type',       default_value='csi',
                                             description="'csi' (Jetson native) or 'usb'")
    device_index_arg = DeclareLaunchArgument('device_index',      default_value='2',
                                             description='USB /dev/videoN index')

    params          = LaunchConfiguration('params_file')
    ctrl_params_cfg = LaunchConfiguration('ctrl_params_file')
    sim             = LaunchConfiguration('use_sim_time')
    detector_type   = LaunchConfiguration('detector_type')
    camera_type     = LaunchConfiguration('camera_type')
    device_index    = LaunchConfiguration('device_index')

    micro_ros_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        output='screen',
        arguments=['serial', '--dev', '/dev/ttyUSB0'],
    )

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
        remappings=[('/cmd_vel_desired', '/cmd_vel_desired_raw')],
    )

    # Slew limiter — sits between the MPC and the velocity controller, caps the
    # rate of change of the command reaching the MCU. The MPC publishes
    # /cmd_vel_desired_raw; this re-publishes the rate-limited /cmd_vel_desired.
    slew_node = Node(
        package='pzb_utils',
        executable='twist_slew_limiter',
        name='twist_slew_limiter',
        output='screen',
        parameters=[{
            'input_topic':       '/cmd_vel_desired_raw',
            'output_topic':      '/cmd_vel_desired',
            'loop_hz':           50.0,
            'max_linear_accel':  0.20,   # m/s²
            'max_angular_accel': 0.50,   # rad/s²
            'max_linear_speed':  0.25,
            'max_angular_speed': 0.45,
            'cmd_timeout_s':     0.60,
        }],
    )

    return LaunchDescription([
        ibvs_params_arg,
        ctrl_params_arg,
        sim_arg,
        detector_arg,
        camera_type_arg,
        device_index_arg,
        micro_ros_node,
        csi_camera_node,
        usb_camera_node,
        detector_node,
        mpc_node,
        slew_node,
        odom_node,
        vel_ctrl_node,
    ])
