#!/usr/bin/env python3
"""
Minimal launch for PID tuning sessions: odometry_node + velocity_controller only.

Publish step commands to /cmd_vel_desired to characterize the PID
velocity response:

  ros2 topic pub /cmd_vel_desired geometry_msgs/msg/Twist \
    "{linear: {x: 0.15}, angular: {z: 0.0}}" --rate 20

Monitor with:
  rqt_plot /VelocityEncR/data /VelocityEncL/data /VelocitySetR/data /VelocitySetL/data
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution([
        FindPackageShare('pzb_control'),
        'config',
        'pid_vel_params.yaml',
    ])

    params_arg = DeclareLaunchArgument('params_file', default_value=default_params)
    sim_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    params = LaunchConfiguration('params_file')
    sim = LaunchConfiguration('use_sim_time')

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

    return LaunchDescription([params_arg, sim_arg, odom_node, vel_ctrl_node])
