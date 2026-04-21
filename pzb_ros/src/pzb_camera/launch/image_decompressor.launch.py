from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'input_topic',
            default_value='/camera/image_compressed',
            description='Compressed image input topic.',
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/camera/image_raw_decompressed',
            description='Raw decompressed image output topic.',
        ),
        DeclareLaunchArgument(
            'frame_id_override',
            default_value='',
            description='Optional frame_id override (empty keeps original).',
        ),
        Node(
            package='pzb_camera',
            executable='image_decompressor',
            name='image_decompressor',
            output='screen',
            parameters=[{
                'input_topic':        LaunchConfiguration('input_topic'),
                'output_topic':       LaunchConfiguration('output_topic'),
                'frame_id_override':  LaunchConfiguration('frame_id_override'),
            }],
        ),
    ])
