#!/usr/bin/env python3
"""
PID teleop with smooth command ramping.

Stack:
  teleop_twist_keyboard (/cmd_vel) -> twist_slew_limiter -> /cmd_vel_desired
  /cmd_vel_desired -> velocity_controller -> /cmd_vel (to base/MCU bridge)

This launch includes pzb_control's odometry + velocity controller bring-up.
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
    lin_arg = DeclareLaunchArgument('initial_linear_speed', default_value='0.20')
    ang_arg = DeclareLaunchArgument('initial_angular_speed', default_value='1.00')
    accel_v_arg = DeclareLaunchArgument('max_linear_accel', default_value='0.25')
    accel_w_arg = DeclareLaunchArgument('max_angular_accel', default_value='1.20')
    input_timeout_arg = DeclareLaunchArgument('input_cmd_timeout_s', default_value='1.50')

    params = LaunchConfiguration('params_file')
    sim = LaunchConfiguration('use_sim_time')
    init_v = LaunchConfiguration('initial_linear_speed')
    init_w = LaunchConfiguration('initial_angular_speed')
    accel_v = LaunchConfiguration('max_linear_accel')
    accel_w = LaunchConfiguration('max_angular_accel')
    input_timeout = LaunchConfiguration('input_cmd_timeout_s')

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

    limiter = Node(
        package='pzb_utils',
        executable='twist_slew_limiter',
        name='twist_slew_limiter',
        output='screen',
        parameters=[{
            'input_topic': '/cmd_vel_desired_raw',
            'output_topic': '/cmd_vel_desired',
            'loop_hz': 50.0,
            'max_linear_accel': accel_v,
            'max_angular_accel': accel_w,
            'max_linear_speed': 0.30,
            'max_angular_speed': 1.50,
            'cmd_timeout_s': input_timeout,
            'initial_linear_speed': 0.0,
            'initial_angular_speed': 0.0,
        }],
    )

    teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e',
        remappings=[('/cmd_vel', '/cmd_vel_desired_raw')],
        parameters=[{
            'speed': init_v,
            'turn': init_w,
            'use_sim_time': sim,
        }],
    )

    return LaunchDescription([
        params_arg,
        sim_arg,
        lin_arg,
        ang_arg,
        accel_v_arg,
        accel_w_arg,
        input_timeout_arg,
        odom_node,
        vel_ctrl_node,
        limiter,
        teleop,
    ])
