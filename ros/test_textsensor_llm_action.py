#!/home/bruce/Downloads/AllTheWorldAPlay/src/owl/bin/python3
"""
TextSensor-to-LLM-Action Pattern Demonstration

This script demonstrates the specific pattern requested:
1. Sense node receives text input (textSensor)
2. Action node gets the sense message
3. Action node extracts textSensor content
4. Action node calls LLM with that content
5. Action node creates and publishes action based on LLM response

Run this to see the complete flow in action!
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import threading
from datetime import datetime


class TextSensorLLMActionTester(Node):
    """
    Test node that monitors the textSensor -> LLM -> Action pattern.
    
    Subscribes to all topics to show the complete message flow.
    """
    
    def __init__(self):
        super().__init__('textsensor_llm_action_tester')
        
        # Subscribe to all relevant topics
        self.sense_subscriber = self.create_subscription(
            String,
            '/cognitive/sense_data',
            self.sense_callback,
            qos_profile=10
        )
        
        self.llm_request_subscriber = self.create_subscription(
            String,
            '/cognitive/llm_request',
            self.llm_request_callback,
            qos_profile=10
        )
        
        self.llm_response_subscriber = self.create_subscription(
            String,
            '/cognitive/llm_response',
            self.llm_response_callback,
            qos_profile=10
        )
        
        self.action_subscriber = self.create_subscription(
            String,
            '/cognitive/action_data',
            self.action_callback,
            qos_profile=10
        )
        
        # Track the flow
        self.tracked_flows = {}  # text_input -> flow_data
        self.flow_counter = 0
        
        self.get_logger().info('🎯 TextSensor-LLM-Action Pattern Tester started')
        self.get_logger().info('📝 Type in the sense node terminal to trigger the pattern!')
        self.get_logger().info('')
        self.get_logger().info('Expected flow:')
        self.get_logger().info('  1. User types in sense node terminal')
        self.get_logger().info('  2. Sense node publishes message with textSensor data')
        self.get_logger().info('  3. Action node receives sense message')
        self.get_logger().info('  4. Action node extracts textSensor content')
        self.get_logger().info('  5. Action node calls LLM with text content')
        self.get_logger().info('  6. LLM service processes request')
        self.get_logger().info('  7. LLM service returns response')
        self.get_logger().info('  8. Action node creates action based on LLM response')
        self.get_logger().info('  9. Action node publishes action')
        self.get_logger().info('')
        self.get_logger().info('🔍 Monitoring all topics for this pattern...')
        self.get_logger().info('=' * 70)
    
    def sense_callback(self, msg):
        """Monitor sense messages for textSensor input."""
        try:
            sense_data = json.loads(msg.data)
            console_sensor = sense_data.get('data', {}).get('console_text_sensor', {})
            
            if console_sensor.get('new_input'):
                text_input = console_sensor.get('current_input', '')
                if text_input.strip():
                    flow_id = f"flow_{self.flow_counter}"
                    self.flow_counter += 1
                    
                    self.tracked_flows[text_input] = {
                        'flow_id': flow_id,
                        'text_input': text_input,
                        'sense_timestamp': sense_data.get('timestamp'),
                        'sense_received': datetime.now().isoformat(),
                        'llm_request_sent': None,
                        'llm_response_received': None,
                        'action_published': None,
                        'completed': False
                    }
                    
                    self.get_logger().info('')
                    self.get_logger().info(f'🎯 STEP 1-2: TextSensor Input Detected [{flow_id}]')
                    self.get_logger().info(f'    📝 User input: "{text_input}"')
                    self.get_logger().info(f'    ⏰ Sense timestamp: {sense_data.get("timestamp")}')
                    self.get_logger().info(f'    📨 Waiting for action node to process this...')
                    
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.get_logger().error(f'Error monitoring sense data: {e}')
    
    def llm_request_callback(self, msg):
        """Monitor LLM requests triggered by textSensor input."""
        try:
            request_data = json.loads(msg.data)
            prompt = request_data.get('prompt', '')
            
            # Look for our tracked text inputs in the prompt
            for text_input, flow_data in self.tracked_flows.items():
                if (not flow_data['completed'] and 
                    text_input in prompt and 
                    flow_data['llm_request_sent'] is None):
                    
                    flow_data['llm_request_sent'] = datetime.now().isoformat()
                    flow_data['request_id'] = request_data.get('request_id')
                    
                    self.get_logger().info(f'🎯 STEP 3-5: LLM Request Sent [{flow_data["flow_id"]}]')
                    self.get_logger().info(f'    🤖 Action node extracted: "{text_input}"')
                    self.get_logger().info(f'    📤 Request ID: {request_data.get("request_id")}')
                    self.get_logger().info(f'    🧠 Prompt: "{prompt[:100]}..."')
                    break
                    
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.get_logger().error(f'Error monitoring LLM request: {e}')
    
    def llm_response_callback(self, msg):
        """Monitor LLM responses."""
        try:
            response_data = json.loads(msg.data)
            request_id = response_data.get('request_id')
            
            # Find matching flow
            for text_input, flow_data in self.tracked_flows.items():
                if (not flow_data['completed'] and 
                    flow_data.get('request_id') == request_id and
                    flow_data['llm_response_received'] is None):
                    
                    flow_data['llm_response_received'] = datetime.now().isoformat()
                    flow_data['llm_response'] = response_data.get('response', '')
                    flow_data['processing_time'] = response_data.get('processing_time', 0)
                    
                    self.get_logger().info(f'🎯 STEP 6-7: LLM Response Received [{flow_data["flow_id"]}]')
                    self.get_logger().info(f'    ✅ Success: {response_data.get("success")}')
                    self.get_logger().info(f'    ⏱️  Processing time: {flow_data["processing_time"]:.2f}s')
                    self.get_logger().info(f'    🧠 LLM response: "{flow_data["llm_response"][:100]}..."')
                    self.get_logger().info(f'    📨 Waiting for action to be created...')
                    break
                    
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.get_logger().error(f'Error monitoring LLM response: {e}')
    
    def action_callback(self, msg):
        """Monitor actions created from LLM responses."""
        try:
            action_data = json.loads(msg.data)
            trigger = action_data.get('trigger', {})
            
            if trigger.get('source') == 'textSensor':
                trigger_input = trigger.get('input', '')
                
                # Find matching flow
                for text_input, flow_data in self.tracked_flows.items():
                    if (not flow_data['completed'] and 
                        trigger_input == text_input and
                        flow_data['action_published'] is None):
                        
                        flow_data['action_published'] = datetime.now().isoformat()
                        flow_data['action_type'] = action_data.get('type')
                        flow_data['action_subtype'] = action_data.get('subtype')
                        flow_data['action_id'] = action_data.get('action_id')
                        flow_data['completed'] = True
                        
                        # Calculate total flow time
                        start_time = datetime.fromisoformat(flow_data['sense_received'])
                        end_time = datetime.fromisoformat(flow_data['action_published'])
                        total_time = (end_time - start_time).total_seconds()
                        
                        self.get_logger().info(f'🎯 STEP 8-9: Action Created & Published [{flow_data["flow_id"]}]')
                        self.get_logger().info(f'    🎬 Action type: {action_data.get("type")} ({action_data.get("subtype")})')
                        self.get_logger().info(f'    🆔 Action ID: {action_data.get("action_id")}')
                        self.get_logger().info(f'    🎯 Priority: {action_data.get("action_plan", {}).get("priority", "unknown")}')
                        self.get_logger().info(f'    ⏱️  Total flow time: {total_time:.2f}s')
                        
                        # Print flow summary
                        self._print_flow_summary(flow_data)
                        break
                        
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.get_logger().error(f'Error monitoring action data: {e}')
    
    def _print_flow_summary(self, flow_data):
        """Print a summary of the complete flow."""
        self.get_logger().info('')
        self.get_logger().info('🎉 COMPLETE FLOW SUMMARY:')
        self.get_logger().info(f'    📝 User Input: "{flow_data["text_input"]}"')
        self.get_logger().info(f'    🤖 LLM Analysis: "{flow_data.get("llm_response", "")[:80]}..."')
        self.get_logger().info(f'    🎬 Resulting Action: {flow_data["action_type"]} (ID: {flow_data["action_id"]})')
        
        if flow_data.get('processing_time'):
            self.get_logger().info(f'    ⏱️  LLM Processing: {flow_data["processing_time"]:.2f}s')
        
        start_time = datetime.fromisoformat(flow_data['sense_received'])
        end_time = datetime.fromisoformat(flow_data['action_published'])
        total_time = (end_time - start_time).total_seconds()
        self.get_logger().info(f'    ⏱️  Total Flow: {total_time:.2f}s')
        
        self.get_logger().info('')
        self.get_logger().info('✅ Pattern completed successfully!')
        self.get_logger().info('🔍 Type another message to see the pattern again...')
        self.get_logger().info('=' * 70)


def main(args=None):
    """Main entry point for the pattern tester."""
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
        tester = TextSensorLLMActionTester()
        
        print("\n" + "="*70)
        print("🎯 TEXTSENSOR-TO-LLM-ACTION PATTERN TESTER")
        print("="*70)
        print("This demonstrates the complete flow:")
        print("  User Input → Sense Node → Action Node → LLM → Action")
        print("")
        print("Requirements:")
        print("1. Sense node must be running (for textSensor input)")
        print("2. LLM service node must be running (for LLM processing)")
        print("3. Simple LLM action example must be running (for pattern demo)")
        print("")
        print("To start the required nodes:")
        print("  ros2 run cognitive_framework sense_node")
        print("  ros2 run cognitive_framework llm_service_node") 
        print("  ros2 run cognitive_framework simple_llm_action_example")
        print("")
        print("Or use VSCode launch configuration: 'Text-to-LLM-Action Example'")
        print("")
        print("Then type messages in the sense node terminal and watch this!")
        print("="*70 + "\n")
        
        rclpy.spin(tester)
        
    except KeyboardInterrupt:
        if tester:
            tester.get_logger().info('Pattern tester stopped')
        else:
            print("Pattern tester stopped")
    except SystemExit:
        if tester:
            tester.get_logger().info('Pattern tester terminated')
        else:
            print("Pattern tester terminated")
    except Exception as e:
        if str(e):
            print(f"❌ Error during pattern test: {e}")
            return 1
        else:
            if tester:
                tester.get_logger().info('Pattern test ended')
            else:
                print("Pattern test ended")
    finally:
        if tester:
            try:
                tester.destroy_node()
            except:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main()) 