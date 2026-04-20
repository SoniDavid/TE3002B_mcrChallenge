#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution([
        FindPackageShare('pzb_control'),
        'config',
        'open_loop_sequence.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('path_mode', default_value='square'),
        DeclareLaunchArgument('square_side_m', default_value='0.35'),
        DeclareLaunchArgument('plan_mode', default_value='speed'),
        DeclareLaunchArgument('target_linear_speed', default_value='0.10'),
        DeclareLaunchArgument('target_angular_speed', default_value='0.60'),
        DeclareLaunchArgument('total_time_s', default_value='120.0'),
        DeclareLaunchArgument('max_linear_speed', default_value='0.50'),
        DeclareLaunchArgument('max_angular_speed', default_value='1.50'),
        DeclareLaunchArgument('loop_hz', default_value='20.0'),
        DeclareLaunchArgument('max_run_time_s', default_value='240.0'),
        DeclareLaunchArgument('stop_burst_cycles', default_value='30'),
        Node(
            package='pzb_control',
            executable='open_loop_controller',
            name='open_loop_controller',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'path_mode': LaunchConfiguration('path_mode'),
                'square_side_m': LaunchConfiguration('square_side_m'),
                'plan_mode': LaunchConfiguration('plan_mode'),
                'target_linear_speed': LaunchConfiguration('target_linear_speed'),
                'target_angular_speed': LaunchConfiguration('target_angular_speed'),
                'total_time_s': LaunchConfiguration('total_time_s'),
                'max_linear_speed': LaunchConfiguration('max_linear_speed'),
                'max_angular_speed': LaunchConfiguration('max_angular_speed'),
                'loop_hz': LaunchConfiguration('loop_hz'),
                'max_run_time_s': LaunchConfiguration('max_run_time_s'),
                'stop_burst_cycles': LaunchConfiguration('stop_burst_cycles'),
            }],
        ),
    ])
