from setuptools import find_packages, setup

package_name = 'dgppo_ros_node_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['plans/*.json'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akilasar',
    maintainer_email='akilasar@mit.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'carla_sampling_mpc_node = dgppo_ros_node_pkg.carla_sampling_mpc_ros_node:main',
            'carla_mpc_node = dgppo_ros_node_pkg.carla_mpc_ros_node:main',
        ],
    },
)
