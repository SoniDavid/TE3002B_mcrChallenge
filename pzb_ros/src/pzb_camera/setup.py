from glob import glob
import os
from setuptools import setup

package_name = 'pzb_camera'

setup(
    name=package_name,
    version='0.1.0',
    packages=['pzb_camera_scripts'],
    package_dir={'pzb_camera_scripts': 'scripts'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.[yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@example.com',
    description='Puzzlebot IMX219 CSI camera publisher for Jetson Nano.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = pzb_camera_scripts.camera_publisher:main',
            'camera_raw_publisher = pzb_camera_scripts.camera_raw_publisher:main',
            'camera_compressed_publisher = pzb_camera_scripts.camera_compressed_publisher:main',
            'usb_camera_publisher = pzb_camera_scripts.usb_camera_publisher:main',
            'image_decompressor = pzb_camera_scripts.image_decompressor:main',
            'calibrate_from_rosbag = pzb_camera_scripts.calibrate_from_rosbag:main',
        ],
    },
)
