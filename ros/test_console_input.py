#!/usr/bin/env python3
"""
Console Input Test for Cognitive Framework

This script demonstrates how to use the console text sensor.
Run this while the cognitive system is running to test console input.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time


class ConsoleInputTester(Node):
    """Test node to monitor console input processing."""
    
    def __init__(self):
        super().__init__('console_input_tester')
        
        # Subscribe to all cognitive topics to see input processing
        self.sense_subscriber = self.create_subscription(
            String,
            '/cognitive/sense_data',
            self.sense_callback,
            qos_profile=10
        )
        
        self.memory_subscriber = self.create_subscription(
            String,
            '/cognitive/memory_data',
            self.memory_callback,
            qos_profile=10
        )
        
        self.action_subscriber = self.create_subscription(
            String,
            '/cognitive/action_data',
            self.action_callback,
            qos_profile=10
        )
        
        self.get_logger().info('Console Input Tester started')
        self.get_logger().info('Monitoring cognitive system for console input processing...')
        self.get_logger().info('Switch to the sense_node terminal and type some text!')
        
    def sense_callback(self, msg):
        """Monitor sense data for console input."""
        try:
            data = json.loads(msg.data)
            console_sensor = data.get('data', {}).get('console_text_sensor', {})
            
            if console_sensor.get('new_input'):
                input_text = console_sensor.get('current_input', '')
                self.get_logger().info(f'🎤 SENSE detected console input: "{input_text}"')
        except:
            pass
    
    def memory_callback(self, msg):
        """Monitor memory processing of console input."""
        try:
            data = json.loads(msg.data)
            if data.get('type') == 'memory_update':
                memory_entry = data.get('memory_entry', {})
                content = memory_entry.get('content', {})
                
                # Check if this memory contains console input
                if 'console_input' in str(content):
                    self.get_logger().info(f'🧠 MEMORY processed console input with importance: {memory_entry.get("importance", 0):.2f}')
        except:
            pass
    
    def action_callback(self, msg):
        """Monitor action responses to console input."""
        try:
            data = json.loads(msg.data)
            action_info = data.get('action', {})
            source = data.get('source', '')
            
            if action_info.get('priority', 0) > 0.5:  # Significant actions
                self.get_logger().info(f'⚡ ACTION triggered: {action_info.get("action", "unknown")} (source: {source}, priority: {action_info.get("priority", 0):.2f})')
        except:
            pass


def main(args=None):
    """Main entry point for console input tester."""
    # Check if ROS2 environment is available
    try:
        import os
        if 'ROS_DISTRO' not in os.environ:
            print("❌ ROS2 environment not sourced!")
            print("Please run: source /opt/ros/jazzy/setup.bash")
            return 1
        
        rclpy.init(args=args)
        
    except Exception as e:
        print(f"❌ Failed to initialize ROS2: {e}")
        print("Make sure ROS2 is installed and environment is sourced:")
        print("  source /opt/ros/jazzy/setup.bash")
        return 1
    
    tester = None
    try:
        tester = ConsoleInputTester()
        
        print("\n" + "="*60)
        print("🎯 CONSOLE INPUT TEST")
        print("="*60)
        print("Instructions:")
        print("1. Make sure the cognitive system is running:")
        print("   ros2 launch cognitive_framework cognitive_system.launch.py")
        print("2. In the sense_node terminal, type messages and press Enter")
        print("3. Watch this terminal for cognitive processing updates")
        print("4. Press Ctrl+C to stop monitoring")
        print("="*60 + "\n")
        
        rclpy.spin(tester)
    except KeyboardInterrupt:
        if tester:
            tester.get_logger().info('Console input test stopped')
        else:
            print("Console input test stopped")
    except SystemExit:
        # Handle timeout or other system exits gracefully
        if tester:
            tester.get_logger().info('Console input test terminated')
        else:
            print("Console input test terminated")
    except Exception as e:
        # Only show error if it's not an empty string (timeout signal)
        if str(e):
            print(f"❌ Error during console input test: {e}")
            return 1
        else:
            # Silent termination (likely from timeout signal)
            if tester:
                tester.get_logger().info('Console input test ended')
            else:
                print("Console input test ended")
    finally:
        # Clean shutdown
        if tester:
            try:
                tester.destroy_node()
            except:
                pass
        
        # Only shutdown if context was initialized
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main()) 