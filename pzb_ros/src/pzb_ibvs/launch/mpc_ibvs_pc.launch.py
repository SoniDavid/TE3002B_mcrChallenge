#!/usr/bin/env python3
"""
PC-SIDE launch file (run on your laptop / desktop).

Brings up the computation-heavy nodes that do not need robot hardware:
  1. visual_detector_node  — CV feature detector; subscribes /camera/image_compressed
  2. mpc_ibvs_node         — Linear MPC controller; publishes /cmd_vel_desired

Usage:
  ros2 launch pzb_ibvs mpc_ibvs_pc.launch.py
  ros2 launch pzb_ibvs mpc_ibvs_pc.launch.py detector_type:=aruco
  ros2 launch pzb_ibvs mpc_ibvs_pc.launch.py params_file:=/path/to/custom.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ibvs_params = PathJoinSubstitution([
        FindPackageShare('pzb_ibvs'), 'config', 'mpc_ibvs_params.yaml',
    ])

    ctrl_params_default = PathJoinSubstitution([
        FindPackageShare('pzb_control'), 'config', 'pid_vel_params.yaml',
    ])

    ibvs_params_arg  = DeclareLaunchArgument('params_file',      default_value=ibvs_params)
    ctrl_params_arg  = DeclareLaunchArgument('ctrl_params_file', default_value=ctrl_params_default)
    sim_arg          = DeclareLaunchArgument('use_sim_time',     default_value='false')
    detector_arg     = DeclareLaunchArgument('detector_type',    default_value='color_blob',
                                             description="'color_blob' or 'aruco'")

    ctrl_params_cfg = LaunchConfiguration('ctrl_params_file')
    params        = LaunchConfiguration('params_file')
    sim           = LaunchConfiguration('use_sim_time')
    detector_type = LaunchConfiguration('detector_type')

    # ── 1. Visual feature detector ────────────────────────────────────────────
    # Receives /camera/image_compressed from the Jetson over the network.
    # Publishes /visual_features [eu, ev, ea, confidence] at camera frame rate.
    detector_node = Node(
        package='pzb_ibvs',
        executable='visual_detector_node',
        name='visual_detector_node',
        output='screen',
        parameters=[params, {'use_sim_time': sim, 'detector_type': detector_type}],
    )

    # ── 2. MPC IBVS controller ────────────────────────────────────────────────
    # Publishes /cmd_vel_desired_raw — TwistSlewLimiter re-publishes as /cmd_vel_desired.
    mpc_node = Node(
        package='pzb_ibvs',
        executable='mpc_ibvs_node',
        name='mpc_ibvs_node',
        output='screen',
        parameters=[params, {'use_sim_time': sim}],
        remappings=[('/cmd_vel_desired', '/cmd_vel_desired_raw')],
    )

    # ── 3. Slew limiter — protects MCU from current spikes ───────────────────
    # Sits between MPC output and the velocity controller. Enforces a hard
    # acceleration cap on the command that reaches the MCU so that sudden
    # detection losses / re-acquisitions cannot cause large current transients.
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

    # ── 4. Odometry ───────────────────────────────────────────────────────────
    odom_node = Node(
        package='pzb_control',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[ctrl_params_cfg, {'use_sim_time': sim}],
    )

    # ── 5. Velocity controller (inner PI loop) ────────────────────────────────
    vel_ctrl_node = Node(
        package='pzb_control',
        executable='velocity_controller',
        name='velocity_controller',
        output='screen',
        parameters=[ctrl_params_cfg, {'use_sim_time': sim}],
    )

    return LaunchDescription([
        ibvs_params_arg,
        ctrl_params_arg,
        sim_arg,
        detector_arg,
        detector_node,
        mpc_node,
        slew_node,
        odom_node,
        vel_ctrl_node,
    ])
