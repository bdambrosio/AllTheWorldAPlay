from setuptools import setup, find_packages

package_name = 'cognitive_framework'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/cognitive_system.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Developer',
    maintainer_email='developer@example.com',
    description='Basic cognitive framework with sense, memory, and action nodes',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sense_node = cognitive_framework.sense_node:main',
            'memory_node = cognitive_framework.memory_node:main',
            'action_node = cognitive_framework.action_node:main',
        ],
    },
) 