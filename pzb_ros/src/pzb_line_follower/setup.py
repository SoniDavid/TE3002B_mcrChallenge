from glob import glob
import os
from setuptools import setup

package_name = 'pzb_line_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=['pzb_line_follower_scripts'],
    package_dir={'pzb_line_follower_scripts': 'scripts'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.[yma]*')),
        # Teach-by-demonstration sign actions (recorded /cmd_vel CSVs, ROUND 9).
        (os.path.join('share', package_name, 'config', 'sign_actions'),
         glob('config/sign_actions/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@example.com',
    description='Image-based line follower using Otsu thresholding for Puzzlebot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_follower_node = pzb_line_follower_scripts.line_follower_node:main',
        ],
    },
)
