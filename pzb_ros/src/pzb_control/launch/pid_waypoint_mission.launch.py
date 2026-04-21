#!/usr/bin/env python3
"""
Full PID stack: odometry_node + velocity_controller + waypoint_follower.

Configure waypoints in pid_vel_params.yaml (waypoints_xyyaw) or pass a
custom params_file argument:

    ros2 launch pzb_control pid_waypoint_mission.launch.py \
    params_file:=/path/to/my_mission.yaml
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
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
    use_slew_arg = DeclareLaunchArgument('use_slew_limiter', default_value='true')
    accel_v_arg = DeclareLaunchArgument('max_linear_accel', default_value='0.20')
    accel_w_arg = DeclareLaunchArgument('max_angular_accel', default_value='1.20')

    params = LaunchConfiguration('params_file')
    sim = LaunchConfiguration('use_sim_time')
    use_slew = LaunchConfiguration('use_slew_limiter')
    accel_v = LaunchConfiguration('max_linear_accel')
    accel_w = LaunchConfiguration('max_angular_accel')

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

    wp_node_slew = Node(
        package='pzb_control',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        condition=IfCondition(use_slew),
        parameters=[params, {'use_sim_time': sim}],
        remappings=[('/cmd_vel_desired', '/cmd_vel_desired_raw')],
    )

    wp_node_direct = Node(
        package='pzb_control',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        condition=UnlessCondition(use_slew),
        parameters=[params, {'use_sim_time': sim}],
    )

    slew_node = Node(
        package='pzb_utils',
        executable='twist_slew_limiter',
        name='twist_slew_limiter',
        output='screen',
        condition=IfCondition(use_slew),
        parameters=[{
            'input_topic': '/cmd_vel_desired_raw',
            'output_topic': '/cmd_vel_desired',
            'loop_hz': 50.0,
            'max_linear_accel': accel_v,
            'max_angular_accel': accel_w,
            'max_linear_speed': 0.30,
            'max_angular_speed': 1.50,
            'cmd_timeout_s': 0.50,
            'initial_linear_speed': 0.0,
            'initial_angular_speed': 0.0,
            'use_sim_time': sim,
        }],
    )

    return LaunchDescription([
        params_arg,
        sim_arg,
        use_slew_arg,
        accel_v_arg,
        accel_w_arg,
        odom_node,
        vel_ctrl_node,
        wp_node_slew,
        wp_node_direct,
        slew_node,
    ])
