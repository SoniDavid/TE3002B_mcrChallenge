from glob import glob
import os
from setuptools import setup


package_name = 'pzb_control'


setup(
    name=package_name,
    version='0.1.0',
    packages=['pzb_control_scripts'],
    package_dir={'pzb_control_scripts': 'scripts'},
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
    description='Puzzlebot Week 2 open-loop mini challenge controller.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mini_challenge_ctrl = pzb_control_scripts.mini_challenge_ctrl:main',
        ],
    },
)
