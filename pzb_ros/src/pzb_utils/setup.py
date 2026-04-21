from glob import glob
import os
from setuptools import setup


package_name = 'pzb_utils'


setup(
    name=package_name,
    version='0.1.0',
    packages=['pzb_utils_scripts'],
    package_dir={'pzb_utils_scripts': 'scripts'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@example.com',
    description='Puzzlebot utility nodes (emergency stop).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emergency_stop = pzb_utils_scripts.emergency_stop:main',
            'twist_slew_limiter = pzb_utils_scripts.twist_slew_limiter:main',
        ],
    },
)
