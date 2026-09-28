from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'clustering'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[py]*'))),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Trained model artifacts, one data_files entry per environment dir.
        # Installed so live_cluster_inference_node can find them when run from an
        # installed workspace rather than from inside clustering/clustering/.
        *[
            (os.path.join('share', package_name, 'encoder_weights',
                          os.path.basename(d.rstrip('/'))),
             [f for f in glob(os.path.join(d, '*'))
              if os.path.isfile(f) and not f.endswith('.png')])
            for d in glob(os.path.join(package_name, 'encoder_weights', '*/'))
        ],
        ],
    install_requires=['setuptools', 'torch'],
    zip_safe=True,
    maintainer='akilasar',
    maintainer_email='akilasar@mit.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This line declares your Python node as an executable
            'carla_scaling = clustering.scaling_carla_inputs:main',
            'live_cluster_inference_node = clustering.live_cluster_inference_node:main',
            'clock_publisher = clustering.clock_publisher:main'
        ],
    },
)
