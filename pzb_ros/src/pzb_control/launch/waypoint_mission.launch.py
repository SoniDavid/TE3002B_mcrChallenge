#!/usr/bin/env python3
"""
Full closed-loop stack: odometry_node + velocity_controller + waypoint_follower.

Configure waypoints in closed_loop_params.yaml (waypoints_xyyaw) or pass a
custom params_file argument:

  ros2 launch pzb_control waypoint_mission.launch.py \
    params_file:=/path/to/my_mission.yaml
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
        'closed_loop_params.yaml',
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

    wp_node = Node(
        package='pzb_control',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[params, {'use_sim_time': sim}],
    )

    return LaunchDescription([params_arg, sim_arg, odom_node, vel_ctrl_node, wp_node])
