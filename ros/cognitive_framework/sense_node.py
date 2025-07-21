#!/usr/bin/env python3
"""
Sense Node - Perception and Input Processing

This node simulates sensory input and publishes perception data.
In a real system, this would interface with cameras, microphones, etc.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import threading
import queue
import sys
from datetime import datetime


class SenseNode(Node):
    """
    The Sense node handles perception and sensory input processing.
    
    Publishers:
        /cognitive/sense_data (std_msgs/String): Raw sensory information
    """
    
    def __init__(self):
        super().__init__('sense_node')
        
        # Publisher for sense data
        self.sense_publisher = self.create_publisher(
            String,
            '/cognitive/sense_data',
            qos_profile=10
        )
        
        # Timer for periodic sensing (simulate 10Hz sensor)
        self.timer = self.create_timer(0.1, self.sense_callback)
        
        # Internal state
        self.sequence_id = 0
        
        # Console input handling
        self.text_input_queue = queue.Queue()
        self.last_text_input = None
        self.input_thread = threading.Thread(target=self._console_input_thread, daemon=True)
        self.input_thread.start()
        
        self.get_logger().info('Sense Node initialized - starting perception loop')
        self.get_logger().info('Console Text Sensor ready - type messages and press Enter to send to cognitive system')
    
    def sense_callback(self):
        """
        Main sensing loop - called periodically to generate sensor data.
        
        In a real implementation, this would:
        - Interface with actual sensors (cameras, microphones, etc.)
        - Process raw sensor data
        - Extract relevant features
        - Publish structured perception data
        """
        
        # Get console input if available
        console_input = self._get_console_input()
        
        # Generate simulated sensor data
        sense_data = {
            'timestamp': datetime.now().isoformat(),
            'sequence_id': self.sequence_id,
            'sensor_type': 'hybrid',  # Mix of simulated and real console input
            'data': {
                'visual': f'visual_input_{self.sequence_id}',
                'audio': f'audio_input_{self.sequence_id}',
                'text': {
                    'simulated': f'text_input_{self.sequence_id}',
                    'console_input': console_input,
                    'last_console_input': self.last_text_input
                },
                'console_text_sensor': {
                    'new_input': console_input is not None,
                    'current_input': console_input,
                    'last_input': self.last_text_input,
                    'input_timestamp': datetime.now().isoformat() if console_input else None
                },
                'environment': {
                    'light_level': 0.5 + 0.3 * (self.sequence_id % 10) / 10,
                    'noise_level': 0.2 + 0.1 * (self.sequence_id % 5) / 5,
                    'temperature': 20.0 + 2.0 * (self.sequence_id % 8) / 8
                }
            }
        }
        
        # Create message
        msg = String()
        msg.data = json.dumps(sense_data, indent=2)
        
        # Publish
        self.sense_publisher.publish(msg)
        
        # Log every 10th message to avoid spam, or when console input is present
        if self.sequence_id % 10 == 0 or console_input is not None:
            log_msg = f'Published sense data #{self.sequence_id}'
            if console_input is not None:
                log_msg += f' [Console input: "{console_input}"]'
            self.get_logger().info(log_msg)
        
        self.sequence_id += 1
    
    def _console_input_thread(self):
        """
        Background thread that reads console input and queues it for processing.
        
        This allows the main ROS2 loop to continue while waiting for user input.
        """
        try:
            while True:
                try:
                    # Prompt for input (will only show in terminal, not in logs)
                    user_input = input()
                    if user_input.strip():  # Only process non-empty input
                        self.text_input_queue.put(user_input.strip())
                        # Log to ROS2 instead of printing directly
                        self.get_logger().info(f'Text input received: "{user_input.strip()}"')
                except EOFError:
                    # Handle Ctrl+D gracefully
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C gracefully
                    break
        except Exception as e:
            self.get_logger().error(f'Console input thread error: {str(e)}')
    
    def _get_console_input(self):
        """
        Get the most recent console input if available.
        
        Returns:
            str or None: Recent text input or None if no new input
        """
        text_input = None
        
        # Get the most recent input (drain queue to get latest)
        while not self.text_input_queue.empty():
            try:
                text_input = self.text_input_queue.get_nowait()
            except queue.Empty:
                break
        
        # Update last input if we got new input
        if text_input is not None:
            self.last_text_input = text_input
            
        return text_input


def main(args=None):
    """Main entry point for the sense node."""
    rclpy.init(args=args)
    
    sense_node = SenseNode()
    
    try:
        rclpy.spin(sense_node)
    except KeyboardInterrupt:
        sense_node.get_logger().info('Sense Node shutting down...')
    finally:
        sense_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 