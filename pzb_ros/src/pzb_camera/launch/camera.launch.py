import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('pzb_camera')
    default_params = os.path.join(pkg_share, 'config', 'camera_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to camera parameter YAML file',
        ),
        DeclareLaunchArgument(
            'sensor_id',
            default_value='0',
            description='CSI camera sensor index',
        ),
        DeclareLaunchArgument(
            'width',
            default_value='1280',
            description='Capture width in pixels',
        ),
        DeclareLaunchArgument(
            'height',
            default_value='720',
            description='Capture height in pixels',
        ),
        DeclareLaunchArgument(
            'framerate',
            default_value='30',
            description='Target framerate',
        ),
        DeclareLaunchArgument(
            'jpeg_quality',
            default_value='80',
            description='JPEG compression quality (1-100)',
        ),

        Node(
            package='pzb_camera',
            executable='camera_publisher',
            name='camera_publisher',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'sensor_id':    LaunchConfiguration('sensor_id'),
                    'width':        LaunchConfiguration('width'),
                    'height':       LaunchConfiguration('height'),
                    'framerate':    LaunchConfiguration('framerate'),
                    'jpeg_quality': LaunchConfiguration('jpeg_quality'),
                },
            ],
        ),
    ])
