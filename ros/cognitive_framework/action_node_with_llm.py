#!/home/bruce/Downloads/AllTheWorldAPlay/src/owl/bin/python3
"""
Enhanced Action Node with LLM Integration

This is an example of how to integrate the LLM client into a cognitive node.
It demonstrates both blocking and non-blocking LLM usage patterns.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from datetime import datetime
from collections import defaultdict

# Import our LLM client
try:
    # Try relative import first (when running as ROS2 package)
    from .llm_client import LLMClient, LLMResponse
except ImportError:
    # Fall back to absolute import (when running directly)
    from llm_client import LLMClient, LLMResponse


class EnhancedActionNode(Node):
    """
    Enhanced Action node that uses LLM for cognitive reasoning and decision making.
    
    Demonstrates:
    - Blocking LLM calls for immediate decisions
    - Non-blocking LLM calls for background reasoning
    - Callback-based LLM usage for fire-and-forget analysis
    """
    
    def __init__(self):
        super().__init__('enhanced_action_node')
        
        # Subscribers (same as original action node)
        self.sense_subscriber = self.create_subscription(
            String,
            '/cognitive/sense_data',
            self.sense_data_callback,
            qos_profile=10
        )
        
        self.memory_subscriber = self.create_subscription(
            String,
            '/cognitive/memory_data',
            self.memory_data_callback,
            qos_profile=10
        )
        
        # Publisher for actions
        self.action_publisher = self.create_publisher(
            String,
            '/cognitive/action_data',
            qos_profile=10
        )
        
        # Initialize LLM client
        self.llm_client = LLMClient(self, service_timeout=15.0)
        
        # Decision-making state
        self.current_sense_data = None
        self.current_memory_data = None
        self.last_sense_timestamp = None
        self.last_memory_timestamp = None
        
        # Action history and planning
        self.action_history = []
        self.current_goals = []
        self.action_sequence_id = 0
        
        # LLM-enhanced reasoning state
        self.background_reasoning = {}  # Track ongoing LLM requests
        self.llm_insights = []          # Store LLM-generated insights
        self.last_llm_decision_time = 0
        
        # Decision parameters
        self.decision_threshold = 0.5
        self.max_action_rate = 2.0
        self.last_action_time = 0
        self.llm_decision_interval = 10.0  # Use LLM every 10 seconds for strategic thinking
        
        # Timer for periodic decision making
        self.decision_timer = self.create_timer(0.5, self.make_decision)
        
        # Timer for strategic LLM-based reasoning
        self.strategic_timer = self.create_timer(self.llm_decision_interval, self.strategic_reasoning)
        
        self.get_logger().info('🤖 Enhanced Action Node with LLM initialized')
    
    def sense_data_callback(self, msg):
        """Process incoming sensory data, including console input."""
        try:
            sense_data = json.loads(msg.data)
            self.current_sense_data = sense_data
            self.last_sense_timestamp = sense_data['timestamp']
            
            # Check for console input - this might need immediate LLM reasoning
            console_sensor = sense_data.get('data', {}).get('console_text_sensor', {})
            if console_sensor.get('new_input'):
                console_input = console_sensor.get('current_input', '')
                self._handle_console_input_with_llm(console_input)
            
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse sense data JSON')
        except Exception as e:
            self.get_logger().error(f'Error processing sense data: {str(e)}')
    
    def memory_data_callback(self, msg):
        """Process memory updates."""
        try:
            memory_data = json.loads(msg.data)
            self.current_memory_data = memory_data
            
            if 'memory_entry' in memory_data:
                self.last_memory_timestamp = memory_data['memory_entry']['timestamp']
                
                # Check if this is high-importance memory that might need LLM analysis
                memory_entry = memory_data['memory_entry']
                importance = memory_entry.get('importance', 0)
                
                if importance > 0.8:  # Very important memory
                    self._analyze_important_memory_with_llm(memory_entry)
            
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse memory data JSON')
        except Exception as e:
            self.get_logger().error(f'Error processing memory data: {str(e)}')
    
    def _handle_console_input_with_llm(self, user_input: str):
        """
        Handle console input using LLM for intelligent response.
        
        This demonstrates BLOCKING LLM usage for immediate responses.
        """
        self.get_logger().info(f'🧠 Analyzing console input with LLM: "{user_input}"')
        
        # Create context-aware prompt
        system_prompt = """You are a cognitive AI agent. A human has provided input to your sensory system. 
Analyze their input and suggest appropriate cognitive actions or responses. 
Be concise but thoughtful. Consider the input's intent, emotional content, and what kind of response would be most helpful."""
        
        prompt = f"Human input: '{user_input}'\n\nAnalyze this input and suggest how I should respond or what actions I should take."
        
        try:
            # BLOCKING call - we want immediate response to user input
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=100,
                temperature=0.7,
                timeout=10.0
            )
            
            if response.success:
                self.get_logger().info(f'✅ LLM analysis: {response.text}')
                
                # Create action based on LLM response
                llm_action = {
                    'type': 'llm_guided_response',
                    'action': 'respond_to_user',
                    'user_input': user_input,
                    'llm_guidance': response.text,
                    'priority': 0.9,  # High priority for user interactions
                    'reason': f'LLM-guided response to user input'
                }
                
                self._execute_action(llm_action, source='llm_blocking')
                
            else:
                self.get_logger().error(f'❌ LLM analysis failed: {response.error}')
                
                # Fallback action without LLM
                fallback_action = {
                    'type': 'fallback_response',
                    'action': 'acknowledge_user_input',
                    'user_input': user_input,
                    'priority': 0.7,
                    'reason': 'LLM unavailable, using fallback response'
                }
                
                self._execute_action(fallback_action, source='fallback')
                
        except Exception as e:
            self.get_logger().error(f'❌ Error in LLM console input analysis: {e}')
    
    def _analyze_important_memory_with_llm(self, memory_entry):
        """
        Analyze important memories using LLM in the background.
        
        This demonstrates NON-BLOCKING LLM usage with callbacks.
        """
        memory_content = memory_entry.get('content', {})
        summary = memory_content.get('summary', 'Unknown memory')
        
        self.get_logger().info(f'🧠 Starting background LLM analysis of important memory')
        
        # Create analysis prompt
        system_prompt = """You are a cognitive AI agent analyzing important memories. 
Look for patterns, implications, and potential actions that should be taken based on this memory.
Consider both immediate and long-term implications."""
        
        prompt = f"Important memory detected: {summary}\n\nAnalyze this memory for patterns, implications, and suggest potential actions or goals."
        
        # NON-BLOCKING call with callback
        def memory_analysis_callback(response: LLMResponse):
            if response.success:
                self.get_logger().info(f'✅ Background memory analysis complete: {response.text}')
                
                # Store insight for future use
                insight = {
                    'type': 'memory_analysis',
                    'memory_summary': summary,
                    'llm_insight': response.text,
                    'timestamp': datetime.now().isoformat(),
                    'importance': memory_entry.get('importance', 0)
                }
                
                self.llm_insights.append(insight)
                
                # Keep only recent insights
                if len(self.llm_insights) > 10:
                    self.llm_insights = self.llm_insights[-5:]
                
                # Create potential goal based on insight
                self._create_goal_from_insight(insight)
                
            else:
                self.get_logger().error(f'❌ Background memory analysis failed: {response.error}')
        
        # Start non-blocking analysis
        future = self.llm_client.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=120,
            temperature=0.6,
            callback=memory_analysis_callback
        )
        
        # Track the request
        self.background_reasoning[future.request_id] = {
            'type': 'memory_analysis',
            'start_time': time.time(),
            'memory_entry': memory_entry
        }
    
    def strategic_reasoning(self):
        """
        Periodic strategic reasoning using LLM.
        
        This demonstrates FIRE-AND-FORGET LLM usage for ongoing strategic thinking.
        """
        current_time = time.time()
        
        # Only do strategic reasoning if we have recent activity
        if (self.current_sense_data is None or 
            current_time - self.last_llm_decision_time < self.llm_decision_interval):
            return
        
        self.last_llm_decision_time = current_time
        
        # Create context summary
        recent_actions = self.action_history[-3:] if self.action_history else []
        active_goals = len(self.current_goals)
        recent_insights = self.llm_insights[-2:] if self.llm_insights else []
        
        context = {
            'recent_actions': len(recent_actions),
            'active_goals': active_goals,
            'recent_insights': len(recent_insights),
            'has_sense_data': self.current_sense_data is not None,
            'has_memory_data': self.current_memory_data is not None
        }
        
        self.get_logger().info('🎯 Starting strategic reasoning with LLM')
        
        # Create strategic prompt
        system_prompt = """You are a cognitive AI agent doing strategic reasoning about your current situation. 
Consider your recent actions, goals, and insights to suggest strategic directions or adjustments."""
        
        prompt = f"""Current situation:
- Recent actions: {context['recent_actions']}
- Active goals: {context['active_goals']}
- Recent insights: {context['recent_insights']}
- Has sensory data: {context['has_sense_data']}
- Has memory data: {context['has_memory_data']}

Based on this context, what strategic actions or goal adjustments should I consider?"""
        
        # Fire-and-forget strategic analysis
        def strategic_callback(response: LLMResponse):
            if response.success:
                self.get_logger().info(f'🎯 Strategic insight: {response.text}')
                
                # Create strategic goal based on LLM suggestion
                strategic_goal = {
                    'goal': 'strategic_adjustment',
                    'llm_guidance': response.text,
                    'priority': 0.6,
                    'created_at': datetime.now().isoformat(),
                    'reason': 'LLM strategic reasoning'
                }
                
                self.current_goals.append(strategic_goal)
                
            else:
                self.get_logger().warning(f'⚠️ Strategic reasoning failed: {response.error}')
        
        # Start strategic analysis (fire-and-forget)
        self.llm_client.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=100,
            temperature=0.8,
            callback=strategic_callback
        )
    
    def _create_goal_from_insight(self, insight):
        """Create actionable goals based on LLM insights."""
        goal = {
            'goal': f'act_on_insight',
            'insight_type': insight['type'],
            'llm_guidance': insight['llm_insight'],
            'priority': min(0.8, insight['importance']),
            'created_at': datetime.now().isoformat(),
            'reason': f'Goal created from LLM {insight["type"]}'
        }
        
        self.current_goals.append(goal)
        self.get_logger().info(f'📋 Created new goal from LLM insight')
    
    def make_decision(self):
        """Enhanced decision making that can use both regular logic and LLM insights."""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_action_time < (1.0 / self.max_action_rate):
            return
        
        # Need data to make decisions
        if self.current_sense_data is None or self.current_memory_data is None:
            return
        
        # Enhanced situation assessment that includes LLM insights
        situation_assessment = self._assess_situation_with_llm()
        
        # Select action based on assessment
        planned_action = self._select_action(situation_assessment)
        
        if planned_action:
            self._execute_action(planned_action, source='enhanced_deliberative')
            self.last_action_time = current_time
    
    def _assess_situation_with_llm(self):
        """Enhanced situation assessment that incorporates LLM insights."""
        # Base assessment (same as original)
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.0,
            'urgency': 0.0,
            'opportunities': [],
            'threats': [],
            'active_goals': len(self.current_goals),
            'llm_insights_available': len(self.llm_insights)
        }
        
        # Include LLM insights in assessment
        if self.llm_insights:
            recent_insights = [insight for insight in self.llm_insights 
                             if (datetime.now() - datetime.fromisoformat(insight['timestamp'])).seconds < 300]
            
            assessment['recent_llm_insights'] = len(recent_insights)
            assessment['confidence'] += len(recent_insights) * 0.1  # LLM insights boost confidence
            
            # Extract opportunities/threats from LLM insights
            for insight in recent_insights:
                if 'opportunity' in insight['llm_insight'].lower():
                    assessment['opportunities'].append('llm_identified_opportunity')
                if 'threat' in insight['llm_insight'].lower() or 'concern' in insight['llm_insight'].lower():
                    assessment['threats'].append('llm_identified_concern')
        
        # Regular sensory/memory analysis (simplified from original)
        if self.current_sense_data:
            env = self.current_sense_data.get('data', {}).get('environment', {})
            console_sensor = self.current_sense_data.get('data', {}).get('console_text_sensor', {})
            
            if console_sensor.get('new_input'):
                assessment['urgency'] += 0.4  # User input is urgent
                assessment['opportunities'].append('user_interaction')
            
            assessment['confidence'] += 0.3
        
        if self.current_memory_data:
            working_memory = self.current_memory_data.get('working_memory', {})
            importance = working_memory.get('current_importance', 0)
            assessment['confidence'] += importance * 0.3
        
        # Normalize
        assessment['confidence'] = min(1.0, assessment['confidence'])
        assessment['urgency'] = min(1.0, assessment['urgency'])
        
        return assessment
    
    def _select_action(self, situation_assessment):
        """Action selection enhanced with LLM guidance."""
        # Priority 1: LLM-guided goals
        llm_goals = [goal for goal in self.current_goals if 'llm_guidance' in goal]
        if llm_goals and situation_assessment['confidence'] > 0.6:
            highest_llm_goal = max(llm_goals, key=lambda g: g['priority'])
            return {
                'type': 'llm_guided',
                'action': 'execute_llm_goal',
                'target': highest_llm_goal['goal'],
                'llm_guidance': highest_llm_goal.get('llm_guidance', ''),
                'priority': highest_llm_goal['priority'],
                'reason': f'Executing LLM-guided goal: {highest_llm_goal["goal"]}'
            }
        
        # Priority 2: Regular goal-directed action (same as original logic)
        if self.current_goals:
            highest_goal = max(self.current_goals, key=lambda g: g['priority'])
            return {
                'type': 'goal_directed',
                'action': 'work_toward_goal',
                'target': highest_goal['goal'],
                'priority': highest_goal['priority'],
                'reason': f"Working toward goal: {highest_goal['goal']}"
            }
        
        # Priority 3: Default exploratory
        return {
            'type': 'exploratory',
            'action': 'gather_information',
            'target': 'environment',
            'priority': 0.3,
            'reason': 'No specific goals, exploring environment'
        }
    
    def _execute_action(self, action, source='enhanced'):
        """Execute actions with LLM integration tracking."""
        # Create action command (same as original but with LLM info)
        action_command = {
            'action_id': self.action_sequence_id,
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'action': action,
            'execution_status': 'initiated',
            'llm_guided': 'llm' in source or 'llm_guidance' in action
        }
        
        # Add to action history
        self.action_history.append(action_command)
        
        # Keep action history manageable
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-50:]
        
        # Remove completed goals
        if action.get('type') in ['goal_directed', 'llm_guided']:
            self.current_goals = [g for g in self.current_goals 
                                 if g['goal'] != action.get('target')]
        
        # Publish action
        msg = String()
        msg.data = json.dumps(action_command, indent=2)
        self.action_publisher.publish(msg)
        
        llm_indicator = '🤖' if action_command['llm_guided'] else '⚡'
        self.get_logger().info(
            f'{llm_indicator} Executed {source} action #{self.action_sequence_id}: '
            f'{action.get("action", "unknown")} (priority: {action.get("priority", 0):.2f})'
        )
        
        self.action_sequence_id += 1


def main(args=None):
    """Main entry point for the enhanced action node."""
    rclpy.init(args=args)
    
    enhanced_action_node = EnhancedActionNode()
    
    try:
        rclpy.spin(enhanced_action_node)
    except KeyboardInterrupt:
        enhanced_action_node.get_logger().info('Enhanced Action Node shutting down...')
    finally:
        enhanced_action_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 