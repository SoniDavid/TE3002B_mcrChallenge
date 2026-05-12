#!/usr/bin/env python3
"""
PC-SIDE launch file (run on your laptop / desktop).

Brings up the computation-heavy nodes that do not need robot hardware:
  1. visual_detector_node  — CV feature detector; subscribes /camera/image_compressed
  2. mpc_ibvs_node         — Linear MPC controller; publishes /cmd_vel_desired

The robot-side launch (mpc_ibvs_robot.launch.py) must already be running on
the Jetson. Both machines must share the same ROS2 network:
  - Same ROS_DOMAIN_ID (default 0; set with: export ROS_DOMAIN_ID=<n>)
  - Reachable via UDP multicast OR unicast with ROS_DISCOVERY_SERVER / FastDDS XML.
  - Tip: verify with `ros2 topic list` on the PC — you should see /camera/image_compressed.

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

    ibvs_params_arg = DeclareLaunchArgument('params_file',    default_value=ibvs_params)
    sim_arg         = DeclareLaunchArgument('use_sim_time',   default_value='false')
    detector_arg    = DeclareLaunchArgument('detector_type',  default_value='color_blob',
                                            description="'color_blob' or 'aruco'")

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
    # Subscribes /visual_features + /robot_vel (both available over the network).
    # Publishes /cmd_vel_desired → picked up by velocity_controller on the Jetson.
    mpc_node = Node(
        package='pzb_ibvs',
        executable='mpc_ibvs_node',
        name='mpc_ibvs_node',
        output='screen',
        parameters=[params, {'use_sim_time': sim}],
    )

    return LaunchDescription([
        ibvs_params_arg,
        sim_arg,
        detector_arg,
        detector_node,
        mpc_node,
    ])
