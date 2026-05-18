from glob import glob
import os
from setuptools import setup

package_name = 'pzb_traffic'

setup(
    name=package_name,
    version='0.1.0',
    packages=['pzb_traffic_scripts'],
    package_dir={'pzb_traffic_scripts': 'scripts'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.[yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='sonidavid46@gmail.com',
    description='Traffic light color detection and FSM for half-term challenge.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'color_detector_node = pzb_traffic_scripts.color_detector_node:main',
            'traffic_light_fsm_node = pzb_traffic_scripts.traffic_light_fsm_node:main',
        ],
    },
)
