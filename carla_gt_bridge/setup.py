import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'carla_gt_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Region tables and cluster maps are inputs the runtime nodes load, so
        # they must be installed, not just present in the source tree.
        # cones.*.json and props.*.json must be listed: `mission.launch.py` passes
        # them as `prop_tables`, and if they are not installed `gt_cue_node` finds no
        # region families and every landmark except cones falls through to
        # /carla/objects, which does not list static props.
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml') + glob('config/*.npz') + glob('config/mission.*.json')
         + glob('config/objects.*.json') + glob('config/cones.*.json')
         + glob('config/props.*.json') + glob('config/landmarks.*.json')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akilasar',
    maintainer_email='akilasar@mit.edu',
    description='Ground-truth CARLA perception substitutes + offline map segmentation.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'gt_cluster_node = carla_gt_bridge.nodes.gt_cluster_node:main',
        'cluster_space_guard = carla_gt_bridge.nodes.cluster_space_guard:main',
        'load_mission_node = carla_gt_bridge.nodes.load_mission_node:main',
        'gt_cue_node = carla_gt_bridge.nodes.gt_cue_node:main',
        'gt_obstacle_node = carla_gt_bridge.nodes.gt_obstacle_node:main',
        'lidar_ranges_node = carla_gt_bridge.nodes.lidar_ranges_node:main',
        'terrain_mask_node = carla_gt_bridge.nodes.terrain_mask_node:main',
    ]},
)
