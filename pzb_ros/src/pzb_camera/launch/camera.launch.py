import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('pzb_camera')
    default_params = os.path.join(pkg_share, 'config', 'camera_params.yaml')
    default_camera_info = os.path.join(pkg_share, 'config', 'camera_info_8x5_3cm.yaml')

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
            default_value='75',
            description='JPEG compression quality (1-100)',
        ),
        DeclareLaunchArgument(
            'flip_method',
            default_value='0',
            description='nvvidconv flip-method (0:none, 2:rotate-180, 4:horizontal, 6:vertical)',
        ),
        DeclareLaunchArgument(
            'color_cal_file',
            default_value='/home/puzzlebot/color_cal/colorCalibration.npz',
            description='Path to .npz color calibration file (arr_0 gains). Empty to disable.',
        ),
        DeclareLaunchArgument(
            'publish_camera_info',
            default_value='true',
            description='Publish /camera/camera_info using the calibration YAML.',
        ),
        DeclareLaunchArgument(
            'camera_info_file',
            default_value=default_camera_info,
            description='Path to ROS camera_info YAML for intrinsics.',
        ),
        DeclareLaunchArgument(
            'topic_camera_info',
            default_value='/camera/camera_info',
            description='Topic name for CameraInfo messages.',
        ),

        Node(
            package='pzb_camera',
            executable='camera_compressed_publisher',
            name='camera_compressed_publisher',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'sensor_id':           LaunchConfiguration('sensor_id'),
                    'width':               LaunchConfiguration('width'),
                    'height':              LaunchConfiguration('height'),
                    'framerate':           LaunchConfiguration('framerate'),
                    'jpeg_quality':        LaunchConfiguration('jpeg_quality'),
                    'flip_method':         LaunchConfiguration('flip_method'),
                    'color_cal_file':      LaunchConfiguration('color_cal_file'),
                    'publish_camera_info': LaunchConfiguration('publish_camera_info'),
                    'camera_info_file':    LaunchConfiguration('camera_info_file'),
                    'topic_camera_info':   LaunchConfiguration('topic_camera_info'),
                },
            ],
        ),
    ])
