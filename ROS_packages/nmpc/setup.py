from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nmpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include message files
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deck',
    maintainer_email='krisj21@student.sdu.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nmpc_node = nmpc.nmpc_node:main',
            'optitrack_estimator = nmpc.optitrack_estimator:main',
            'state_estimator = nmpc.state_estimator:main',
            'slung_load_nmpc_node = nmpc.slung_load_nmpc_node:main',
            'logger_node = nmpc.full_log:main',
            'slung_load_uncompensated_node = nmpc.slung_load_uncompensated_node:main',
            'e_stop_uncompensated_node = nmpc.e_stop_uncompensated_node:main',
            'e_stop_compensated_node = nmpc.e_stop_compensated_node:main',
            'ukf_node_ros_bag = nmpc.ukf_node_ros_bag:main',
            'ukf_node = nmpc.ukf_node:main',
            'rosbag2csv_node = nmpc.rosbag2csv_node:main',
            'state_estimator_camera = nmpc.state_estimator_camera:main',
        ],
    },
)
