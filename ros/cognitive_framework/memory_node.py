#!/usr/bin/env python3
"""
Memory Node - Storage and Consolidation

This node receives sensory data, processes it, and maintains memory state.
It publishes memory updates and consolidated information.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from datetime import datetime
from collections import deque


class MemoryNode(Node):
    """
    The Memory node handles information storage and consolidation.
    
    Subscribers:
        /cognitive/sense_data (std_msgs/String): Raw sensory information
    
    Publishers:
        /cognitive/memory_data (std_msgs/String): Processed memory updates
    """
    
    def __init__(self):
        super().__init__('memory_node')
        
        # Subscriber for sense data
        self.sense_subscriber = self.create_subscription(
            String,
            '/cognitive/sense_data',
            self.sense_data_callback,
            qos_profile=10
        )
        
        # Publisher for memory updates
        self.memory_publisher = self.create_publisher(
            String,
            '/cognitive/memory_data',
            qos_profile=10
        )
        
        # Memory storage - using simple structures for this stub
        self.short_term_memory = deque(maxlen=100)  # Recent experiences
        self.long_term_memory = []  # Consolidated memories
        self.working_memory = {}   # Current active information
        
        # Processing state
        self.last_consolidation = datetime.now()
        self.consolidation_interval = 10.0  # seconds
        self.memory_sequence_id = 0
        
        # Timer for periodic memory consolidation
        self.consolidation_timer = self.create_timer(
            self.consolidation_interval, 
            self.consolidate_memory
        )
        
        self.get_logger().info('Memory Node initialized - ready to process sensory input')
    
    def sense_data_callback(self, msg):
        """
        Process incoming sensory data and update memory structures.
        
        Args:
            msg (std_msgs/String): Sensory data from sense node
        """
        try:
            # Parse incoming sense data
            sense_data = json.loads(msg.data)
            
            # Extract relevant information for memory storage
            memory_entry = {
                'timestamp': sense_data['timestamp'],
                'source_sequence': sense_data['sequence_id'],
                'memory_id': self.memory_sequence_id,
                'content': self._extract_memorable_content(sense_data),
                'importance': self._calculate_importance(sense_data),
                'processed_at': datetime.now().isoformat()
            }
            
            # Store in short-term memory
            self.short_term_memory.append(memory_entry)
            
            # Update working memory with current state
            self._update_working_memory(memory_entry)
            
            # Publish memory update
            self._publish_memory_update(memory_entry)
            
            self.memory_sequence_id += 1
            
            # Log every 20th message
            if self.memory_sequence_id % 20 == 0:
                self.get_logger().info(
                    f'Processed memory entry #{self.memory_sequence_id}, '
                    f'STM size: {len(self.short_term_memory)}, '
                    f'LTM size: {len(self.long_term_memory)}'
                )
                
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse sense data JSON')
        except Exception as e:
            self.get_logger().error(f'Error processing sense data: {str(e)}')
    
    def _extract_memorable_content(self, sense_data):
        """
        Extract memorable aspects from sensory input.
        
        In a real system, this would:
        - Apply attention mechanisms
        - Extract semantic content
        - Identify novel or important information
        - Apply memory encoding strategies
        
        Args:
            sense_data (dict): Raw sensory data
            
        Returns:
            dict: Extracted memorable content
        """
        data = sense_data.get('data', {})
        
        # Handle both old and new text data formats
        text_data = data.get('text', '')
        if isinstance(text_data, dict):
            # New format with console input
            console_input = text_data.get('console_input')
            last_console = text_data.get('last_console_input')
            simulated_text = text_data.get('simulated', '')
        else:
            # Old format - simple string
            console_input = None
            last_console = None
            simulated_text = text_data
        
        # Extract console text sensor data
        console_sensor = data.get('console_text_sensor', {})
        has_new_input = console_sensor.get('new_input', False)
        
        memorable_content = {
            'visual_features': data.get('visual', ''),
            'audio_features': data.get('audio', ''),
            'text_content': {
                'simulated': simulated_text,
                'console_input': console_input,
                'last_console_input': last_console,
                'has_new_console_input': has_new_input
            },
            'console_interaction': {
                'new_input': has_new_input,
                'current_input': console_input,
                'interaction_timestamp': console_sensor.get('input_timestamp')
            },
            'environmental_context': data.get('environment', {}),
            'summary': f"Sensory experience at {sense_data['timestamp']}"
        }
        
        # Add console input to summary if present
        if console_input:
            memorable_content['summary'] += f' [User said: "{console_input}"]'
        
        return memorable_content
    
    def _calculate_importance(self, sense_data):
        """
        Calculate the importance/salience of sensory data for memory encoding.
        
        Args:
            sense_data (dict): Raw sensory data
            
        Returns:
            float: Importance score (0.0 to 1.0)
        """
        data = sense_data.get('data', {})
        
        # Check for console input - this is very important!
        console_sensor = data.get('console_text_sensor', {})
        has_console_input = console_sensor.get('new_input', False)
        
        if has_console_input:
            # Console input gets very high importance (0.8-1.0)
            console_input = console_sensor.get('current_input', '')
            input_length = len(console_input)
            
            # Longer inputs get slightly higher importance
            console_importance = 0.8 + min(0.2, input_length / 100)
            return min(1.0, console_importance)
        
        # For non-console data, use environmental factors
        env = data.get('environment', {})
        
        # Higher importance for unusual environmental conditions
        base_importance = 0.3
        light_variation = abs(env.get('light_level', 0.5) - 0.5) * 2
        noise_variation = abs(env.get('noise_level', 0.25) - 0.25) * 4
        
        importance = min(1.0, base_importance + light_variation + noise_variation)
        return importance
    
    def _update_working_memory(self, memory_entry):
        """
        Update working memory with current active information.
        
        Args:
            memory_entry (dict): New memory entry
        """
        self.working_memory = {
            'current_focus': memory_entry['content'],
            'last_update': memory_entry['timestamp'],
            'active_memories': len(self.short_term_memory),
            'current_importance': memory_entry['importance']
        }
    
    def _publish_memory_update(self, memory_entry):
        """
        Publish memory update to other cognitive components.
        
        Args:
            memory_entry (dict): Memory entry to publish
        """
        memory_update = {
            'type': 'memory_update',
            'timestamp': datetime.now().isoformat(),
            'memory_entry': memory_entry,
            'working_memory': self.working_memory,
            'memory_stats': {
                'short_term_count': len(self.short_term_memory),
                'long_term_count': len(self.long_term_memory)
            }
        }
        
        msg = String()
        msg.data = json.dumps(memory_update, indent=2)
        self.memory_publisher.publish(msg)
    
    def consolidate_memory(self):
        """
        Periodic memory consolidation - transfer important memories to long-term storage.
        
        In a real system, this would:
        - Apply consolidation algorithms
        - Compress and generalize memories
        - Form associations and patterns
        - Prune less important memories
        """
        if len(self.short_term_memory) == 0:
            return
            
        # Simple consolidation: move high-importance memories to long-term storage
        consolidation_threshold = 0.7
        
        consolidated_count = 0
        for memory in list(self.short_term_memory):
            if memory['importance'] >= consolidation_threshold:
                # Create consolidated memory entry
                ltm_entry = {
                    'consolidated_at': datetime.now().isoformat(),
                    'original_memory': memory,
                    'consolidation_type': 'importance_based',
                    'ltm_id': len(self.long_term_memory)
                }
                
                self.long_term_memory.append(ltm_entry)
                consolidated_count += 1
        
        if consolidated_count > 0:
            self.get_logger().info(
                f'Consolidated {consolidated_count} memories to long-term storage. '
                f'LTM size: {len(self.long_term_memory)}'
            )
            
            # Publish consolidation update
            consolidation_update = {
                'type': 'memory_consolidation',
                'timestamp': datetime.now().isoformat(),
                'consolidated_count': consolidated_count,
                'ltm_size': len(self.long_term_memory)
            }
            
            msg = String()
            msg.data = json.dumps(consolidation_update, indent=2)
            self.memory_publisher.publish(msg)


def main(args=None):
    """Main entry point for the memory node."""
    rclpy.init(args=args)
    
    memory_node = MemoryNode()
    
    try:
        rclpy.spin(memory_node)
    except KeyboardInterrupt:
        memory_node.get_logger().info('Memory Node shutting down...')
    finally:
        memory_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 