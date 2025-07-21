#!/usr/bin/env python3
"""
Cognitive System Launch File

Launches the complete cognitive framework with all three nodes:
- sense_node: Perception and sensory input
- memory_node: Storage and consolidation
- action_node: Decision making and execution
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for the cognitive system."""
    
    # Declare launch arguments
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level for all nodes'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='cognitive_system',
        description='Namespace for all cognitive nodes'
    )
    
    # Get launch configurations
    log_level = LaunchConfiguration('log_level')
    namespace = LaunchConfiguration('namespace')
    
    # Define nodes
    sense_node = Node(
        package='cognitive_framework',
        executable='sense_node',
        name='sense_node',
        namespace=namespace,
        output='screen',
        parameters=[
            {'log_level': log_level}
        ],
        ros_arguments=['--log-level', log_level]
    )
    
    memory_node = Node(
        package='cognitive_framework',
        executable='memory_node',
        name='memory_node',
        namespace=namespace,
        output='screen',
        parameters=[
            {'log_level': log_level}
        ],
        ros_arguments=['--log-level', log_level]
    )
    
    action_node = Node(
        package='cognitive_framework',
        executable='action_node',
        name='action_node',
        namespace=namespace,
        output='screen',
        parameters=[
            {'log_level': log_level}
        ],
        ros_arguments=['--log-level', log_level]
    )
    
    # Launch information
    launch_info = LogInfo(
        msg=[
            'Starting Cognitive Framework System\n',
            '  - Sense Node: /cognitive/sense_data publisher\n',
            '  - Memory Node: sense subscriber → /cognitive/memory_data publisher\n', 
            '  - Action Node: sense + memory subscribers → /cognitive/action_data publisher\n',
            '  - Namespace: ', namespace, '\n',
            '  - Log Level: ', log_level
        ]
    )
    
    return LaunchDescription([
        # Arguments
        log_level_arg,
        namespace_arg,
        
        # Information
        launch_info,
        
        # Nodes (order matters for dependencies)
        sense_node,
        memory_node, 
        action_node
    ]) 