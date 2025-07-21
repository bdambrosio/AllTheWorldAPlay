#!/usr/bin/env python3
"""
Test Script for Cognitive Framework

This script tests the basic functionality of the cognitive framework by:
1. Subscribing to all output topics
2. Monitoring message flow
3. Validating communication between nodes
4. Reporting system health
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from datetime import datetime, timedelta
import threading


class CognitiveSystemTester(Node):
    """Test node to validate cognitive framework functionality."""
    
    def __init__(self):
        super().__init__('cognitive_system_tester')
        
        # Subscribers for all cognitive topics
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
        
        # Test state tracking
        self.test_start_time = datetime.now()
        self.test_duration = timedelta(seconds=30)  # 30 second test
        
        # Message counters
        self.sense_count = 0
        self.memory_count = 0  
        self.action_count = 0
        
        # Message timestamps
        self.last_sense_time = None
        self.last_memory_time = None
        self.last_action_time = None
        
        # Validation flags
        self.sense_active = False
        self.memory_active = False
        self.action_active = False
        
        # Timer for periodic health checks
        self.health_timer = self.create_timer(5.0, self.health_check)
        
        # Timer for final report
        self.final_timer = self.create_timer(30.0, self.final_report)
        
        self.get_logger().info('Cognitive System Tester started - monitoring for 30 seconds...')
        
    def sense_callback(self, msg):
        """Process sense data messages."""
        try:
            data = json.loads(msg.data)
            self.sense_count += 1
            self.last_sense_time = datetime.now()
            self.sense_active = True
            
            # Validate sense data structure
            required_fields = ['timestamp', 'sequence_id', 'sensor_type', 'data']
            for field in required_fields:
                if field not in data:
                    self.get_logger().warning(f'Missing required field in sense data: {field}')
                    
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in sense data')
        except Exception as e:
            self.get_logger().error(f'Error processing sense data: {str(e)}')
    
    def memory_callback(self, msg):
        """Process memory data messages."""
        try:
            data = json.loads(msg.data)
            self.memory_count += 1
            self.last_memory_time = datetime.now()
            self.memory_active = True
            
            # Validate memory data structure
            if data.get('type') == 'memory_update':
                required_fields = ['memory_entry', 'working_memory', 'memory_stats']
                for field in required_fields:
                    if field not in data:
                        self.get_logger().warning(f'Missing required field in memory data: {field}')
            
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in memory data')
        except Exception as e:
            self.get_logger().error(f'Error processing memory data: {str(e)}')
    
    def action_callback(self, msg):
        """Process action data messages."""
        try:
            data = json.loads(msg.data)
            self.action_count += 1
            self.last_action_time = datetime.now()
            self.action_active = True
            
            # Validate action data structure
            required_fields = ['action_id', 'timestamp', 'source', 'action']
            for field in required_fields:
                if field not in data:
                    self.get_logger().warning(f'Missing required field in action data: {field}')
            
            # Log significant actions
            action_info = data.get('action', {})
            action_type = action_info.get('type', 'unknown')
            priority = action_info.get('priority', 0)
            
            if priority > 0.7:  # High priority actions
                self.get_logger().info(f'High priority action: {action_type} (priority: {priority:.2f})')
                
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action data')
        except Exception as e:
            self.get_logger().error(f'Error processing action data: {str(e)}')
    
    def health_check(self):
        """Periodic health check and status report."""
        current_time = datetime.now()
        elapsed = current_time - self.test_start_time
        
        self.get_logger().info(
            f'Health Check ({elapsed.seconds}s elapsed):\n'
            f'  Sense: {self.sense_count} messages (active: {self.sense_active})\n'
            f'  Memory: {self.memory_count} messages (active: {self.memory_active})\n' 
            f'  Action: {self.action_count} messages (active: {self.action_active})'
        )
        
        # Check for inactive nodes
        now = datetime.now()
        inactive_threshold = timedelta(seconds=10)
        
        warnings = []
        if self.last_sense_time and (now - self.last_sense_time) > inactive_threshold:
            warnings.append('Sense node may be inactive')
        if self.last_memory_time and (now - self.last_memory_time) > inactive_threshold:
            warnings.append('Memory node may be inactive')
        if self.last_action_time and (now - self.last_action_time) > inactive_threshold:
            warnings.append('Action node may be inactive')
            
        for warning in warnings:
            self.get_logger().warning(warning)
    
    def final_report(self):
        """Generate final test report."""
        current_time = datetime.now()
        total_elapsed = current_time - self.test_start_time
        
        # Calculate rates
        sense_rate = self.sense_count / total_elapsed.total_seconds() if total_elapsed.total_seconds() > 0 else 0
        memory_rate = self.memory_count / total_elapsed.total_seconds() if total_elapsed.total_seconds() > 0 else 0
        action_rate = self.action_count / total_elapsed.total_seconds() if total_elapsed.total_seconds() > 0 else 0
        
        # Determine overall health
        health_score = 0
        max_score = 3
        
        if self.sense_active and self.sense_count > 50:  # Expected ~10Hz for 30s = ~300 messages
            health_score += 1
        if self.memory_active and self.memory_count > 10:  # Expected some memory updates
            health_score += 1  
        if self.action_active and self.action_count > 5:   # Expected some actions
            health_score += 1
            
        health_percentage = (health_score / max_score) * 100
        
        # Generate report
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'COGNITIVE SYSTEM TEST REPORT\n'
            f'{"="*60}\n'
            f'Test Duration: {total_elapsed.total_seconds():.1f} seconds\n'
            f'\nMessage Counts:\n'
            f'  Sense Messages: {self.sense_count} ({sense_rate:.1f}/sec)\n'
            f'  Memory Messages: {self.memory_count} ({memory_rate:.1f}/sec)\n'
            f'  Action Messages: {self.action_count} ({action_rate:.1f}/sec)\n'
            f'\nNode Status:\n'
            f'  Sense Node: {"✓ ACTIVE" if self.sense_active else "✗ INACTIVE"}\n'
            f'  Memory Node: {"✓ ACTIVE" if self.memory_active else "✗ INACTIVE"}\n'
            f'  Action Node: {"✓ ACTIVE" if self.action_active else "✗ INACTIVE"}\n'
            f'\nOverall Health: {health_percentage:.0f}% ({health_score}/{max_score} components functional)\n'
            f'{"="*60}'
        )
        
        # Recommendations
        recommendations = []
        if not self.sense_active:
            recommendations.append('Check if sense_node is running')
        if not self.memory_active:
            recommendations.append('Check if memory_node is running and receiving sense data')
        if not self.action_active:
            recommendations.append('Check if action_node is running and receiving sense+memory data')
        if sense_rate < 5:
            recommendations.append('Sense data rate is low - check sense_node timer')
        if self.memory_count == 0 and self.sense_count > 0:
            recommendations.append('Memory node not processing sense data - check subscriptions')
        if self.action_count == 0 and self.memory_count > 0:
            recommendations.append('Action node not generating actions - check decision thresholds')
            
        if recommendations:
            self.get_logger().info(
                f'Recommendations:\n' + 
                '\n'.join(f'  - {rec}' for rec in recommendations)
            )
        else:
            self.get_logger().info('✓ All systems appear to be functioning correctly!')
            
        # Shutdown after report
        rclpy.shutdown()


def main(args=None):
    """Main entry point for the test script."""
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
        tester = CognitiveSystemTester()
        rclpy.spin(tester)
    except KeyboardInterrupt:
        if tester:
            tester.get_logger().info('Test interrupted by user')
        else:
            print("Test interrupted by user")
    except SystemExit:
        # Handle timeout or other system exits gracefully
        if tester:
            tester.get_logger().info('Test terminated')
        else:
            print("Test terminated")
    except Exception as e:
        # Only show error if it's not an empty string (timeout signal)
        if str(e):
            print(f"❌ Error during test: {e}")
            return 1
        else:
            # Silent termination (likely from timeout signal)
            if tester:
                tester.get_logger().info('Test ended')
            else:
                print("Test ended")
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