#!/home/bruce/Downloads/AllTheWorldAPlay/src/owl/bin/python3
"""
Simple LLM Action Example

This is a focused example showing how an action node can call the LLM 
with textSensor contents when it receives a sense message containing text input.

This demonstrates the specific pattern requested:
1. Action node receives sense message
2. Extracts textSensor content 
3. Calls LLM with that content
4. Uses LLM response to determine action
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from datetime import datetime

# Import our LLM client
try:
    # Try relative import first (when running as ROS2 package)
    from .llm_client import LLMClient, LLMResponse
except ImportError:
    # Fall back to absolute import (when running directly)
    from llm_client import LLMClient, LLMResponse


class SimpleLLMActionExample(Node):
    """
    Simple example action node that processes textSensor input with LLM.
    
    Flow:
    1. Receives sense_data message
    2. Checks if it contains textSensor input
    3. If yes, calls LLM to analyze the text
    4. Creates and publishes action based on LLM response
    """
    
    def __init__(self):
        super().__init__('simple_llm_action_example')
        
        # Subscribe to sense data (to get textSensor input)
        self.sense_subscriber = self.create_subscription(
            String,
            '/cognitive/sense_data',
            self.sense_data_callback,
            qos_profile=10
        )
        
        # Publisher for our actions
        self.action_publisher = self.create_publisher(
            String,
            '/cognitive/action_data',
            qos_profile=10
        )
        
        # Initialize LLM client
        self.llm_client = LLMClient(self, service_timeout=15.0)
        
        # Simple state tracking
        self.action_counter = 0
        
        self.get_logger().info('🤖 Simple LLM Action Example initialized')
        self.get_logger().info('📝 Waiting for textSensor input to process with LLM...')
    
    def sense_data_callback(self, msg):
        """
        Process incoming sense data and check for textSensor input.
        
        This is the main callback that demonstrates the requested pattern.
        """
        try:
            # Parse the sense data
            sense_data = json.loads(msg.data)
            
            # Extract textSensor information
            data = sense_data.get('data', {})
            console_text_sensor = data.get('console_text_sensor', {})
            
            # Check if we have new text input
            if console_text_sensor.get('new_input', False):
                current_input = console_text_sensor.get('current_input', '')
                
                if current_input.strip():
                    self.get_logger().info(f'📥 Received textSensor input: "{current_input}"')
                    
                    # This is the key pattern: call LLM with textSensor content
                    self._process_text_input_with_llm(current_input, sense_data)
                    
        except json.JSONDecodeError as e:
            self.get_logger().error(f'❌ Failed to parse sense data JSON: {e}')
        except Exception as e:
            self.get_logger().error(f'❌ Error processing sense data: {e}')
    
    def _process_text_input_with_llm(self, text_input: str, sense_data: dict):
        """
        Process textSensor input using LLM and create appropriate action.
        
        This method demonstrates both blocking and non-blocking LLM usage patterns.
        
        Args:
            text_input: The text from the textSensor
            sense_data: Full sense data context for additional info
        """
        self.get_logger().info(f'🧠 Processing textSensor input with LLM: "{text_input}"')
        
        # Example 1: BLOCKING LLM call for immediate action
        self._blocking_llm_example(text_input, sense_data)
        
        # Example 2: NON-BLOCKING LLM call for background analysis
        self._non_blocking_llm_example(text_input, sense_data)
    
    def _blocking_llm_example(self, text_input: str, sense_data: dict):
        """
        Example of BLOCKING LLM call - use when you need immediate response.
        """
        # Create system prompt for the LLM
        system_prompt = """You are a cognitive action system. A user has provided text input to your sensors.
Analyze their input and suggest a specific action the system should take.
Respond with a clear, actionable directive in 1-2 sentences."""
        
        # Create user prompt with the textSensor content
        user_prompt = f"""User input from textSensor: "{text_input}"

Additional context from sensors:
- Timestamp: {sense_data.get('timestamp', 'unknown')}
- Sequence ID: {sense_data.get('sequence_id', 'unknown')}
- Has environmental data: {bool(sense_data.get('data', {}).get('environment'))}

What action should the cognitive system take in response to this input?"""
        
        try:
            # BLOCKING call - wait for LLM response
            self.get_logger().info('⏳ Making BLOCKING LLM call...')
            
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=100,
                temperature=0.7,
                timeout=10.0
            )
            
            if response.success:
                self.get_logger().info(f'✅ LLM response received: "{response.text}"')
                
                # Create action based on LLM response
                action = self._create_action_from_llm_response(
                    text_input, 
                    response.text, 
                    'blocking_llm',
                    response.processing_time
                )
                
                # Publish the action
                self._publish_action(action)
                
            else:
                self.get_logger().error(f'❌ LLM call failed: {response.error}')
                
                # Create fallback action
                fallback_action = self._create_fallback_action(text_input, 'llm_failure')
                self._publish_action(fallback_action)
                
        except Exception as e:
            self.get_logger().error(f'❌ Error in blocking LLM call: {e}')
            fallback_action = self._create_fallback_action(text_input, 'exception')
            self._publish_action(fallback_action)
    
    def _non_blocking_llm_example(self, text_input: str, sense_data: dict):
        """
        Example of NON-BLOCKING LLM call - for background analysis.
        """
        # Create a different prompt for background analysis
        analysis_prompt = f"""Analyze this user input for deeper cognitive insights: "{text_input}"

Consider:
- Intent and emotional tone
- Complexity and implications  
- Long-term goals or patterns
- Potential follow-up actions

Provide insights for the cognitive system's strategic planning."""
        
        system_prompt = """You are a cognitive analyst providing strategic insights for an AI system.
Focus on understanding deeper patterns and long-term implications."""
        
        try:
            # NON-BLOCKING call with callback
            self.get_logger().info('🚀 Starting NON-BLOCKING LLM analysis...')
            
            def analysis_callback(response: LLMResponse):
                if response.success:
                    self.get_logger().info(f'🔍 Background analysis complete: "{response.text}"')
                    
                    # Create strategic action based on analysis
                    strategic_action = self._create_action_from_llm_response(
                        text_input,
                        response.text,
                        'strategic_analysis',
                        response.processing_time
                    )
                    
                    # Publish strategic action
                    self._publish_action(strategic_action)
                    
                else:
                    self.get_logger().warning(f'⚠️ Background analysis failed: {response.error}')
            
            # Start non-blocking analysis
            future = self.llm_client.generate_async(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                max_tokens=120,
                temperature=0.8,
                callback=analysis_callback
            )
            
            self.get_logger().info(f'📋 Background analysis started (ID: {future.request_id})')
            
        except Exception as e:
            self.get_logger().error(f'❌ Error starting non-blocking LLM call: {e}')
    
    def _create_action_from_llm_response(self, 
                                       original_input: str, 
                                       llm_response: str, 
                                       action_type: str,
                                       processing_time: float) -> dict:
        """
        Create an action based on LLM response.
        
        Args:
            original_input: The original textSensor input
            llm_response: The LLM's response
            action_type: Type of action (blocking_llm, strategic_analysis, etc.)
            processing_time: How long the LLM took to respond
            
        Returns:
            Action dictionary ready to publish
        """
        action = {
            'action_id': self.action_counter,
            'timestamp': datetime.now().isoformat(),
            'type': 'llm_guided_action',
            'subtype': action_type,
            'trigger': {
                'source': 'textSensor',
                'input': original_input
            },
            'llm_analysis': {
                'response': llm_response,
                'processing_time_seconds': processing_time
            },
            'action_plan': {
                'description': f'Action based on LLM analysis of: "{original_input}"',
                'llm_guidance': llm_response,
                'priority': 0.8 if action_type == 'blocking_llm' else 0.6,
                'urgency': 0.9 if action_type == 'blocking_llm' else 0.4
            },
            'metadata': {
                'node_name': self.get_name(),
                'llm_enhanced': True,
                'text_input_length': len(original_input),
                'llm_response_length': len(llm_response)
            }
        }
        
        self.action_counter += 1
        return action
    
    def _create_fallback_action(self, text_input: str, failure_reason: str) -> dict:
        """Create a fallback action when LLM is not available."""
        action = {
            'action_id': self.action_counter,
            'timestamp': datetime.now().isoformat(),
            'type': 'fallback_action',
            'trigger': {
                'source': 'textSensor',
                'input': text_input
            },
            'action_plan': {
                'description': f'Fallback response to: "{text_input}"',
                'fallback_reason': failure_reason,
                'priority': 0.5,
                'urgency': 0.7
            },
            'metadata': {
                'node_name': self.get_name(),
                'llm_enhanced': False,
                'fallback': True
            }
        }
        
        self.action_counter += 1
        return action
    
    def _publish_action(self, action: dict):
        """Publish an action to the action topic."""
        try:
            # Create and publish message
            msg = String()
            msg.data = json.dumps(action, indent=2)
            self.action_publisher.publish(msg)
            
            # Log action
            action_type = action.get('type', 'unknown')
            action_id = action.get('action_id', 'unknown')
            trigger_input = action.get('trigger', {}).get('input', 'unknown')
            priority = action.get('action_plan', {}).get('priority', 0)
            
            emoji = '🤖' if action.get('metadata', {}).get('llm_enhanced') else '⚡'
            
            self.get_logger().info(
                f'{emoji} Published {action_type} action #{action_id}: '
                f'Response to "{trigger_input[:30]}..." (priority: {priority:.2f})'
            )
            
        except Exception as e:
            self.get_logger().error(f'❌ Error publishing action: {e}')


def main(args=None):
    """Main entry point for the simple LLM action example."""
    rclpy.init(args=args)
    
    simple_action_node = SimpleLLMActionExample()
    
    try:
        rclpy.spin(simple_action_node)
    except KeyboardInterrupt:
        simple_action_node.get_logger().info('Simple LLM Action Example shutting down...')
    finally:
        simple_action_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 