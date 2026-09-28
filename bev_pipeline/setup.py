from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bev_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy]*'))),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'PyYAML',
        'scikit-learn',
        'torch',
        'torchvision',
        'open3d',
        # pypatchworkpp is pip-only and must be installed manually before use;
        # see README. Declared here for documentation, not auto-installed by colcon.
        'pypatchworkpp',
    ],
    zip_safe=True,
    maintainer='akilasar',
    maintainer_email='akilasar@mit.edu',
    description='Robocentric BEV / 3D-volumetric behavioral-state classification pipeline.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_inference_node = bev_pipeline.nodes.bev_inference_node:main',
        ],
    },
)
