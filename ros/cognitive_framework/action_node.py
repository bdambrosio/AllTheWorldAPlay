#!/usr/bin/env python3
"""
Action Node - Decision Making and Execution

This node receives both sensory data and memory updates, 
makes decisions, and publishes action commands.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from datetime import datetime
from collections import defaultdict


class ActionNode(Node):
    """
    The Action node handles decision making and action execution.
    
    Subscribers:
        /cognitive/sense_data (std_msgs/String): Direct sensory information
        /cognitive/memory_data (std_msgs/String): Processed memory information
    
    Publishers:
        /cognitive/action_data (std_msgs/String): Action commands and decisions
    """
    
    def __init__(self):
        super().__init__('action_node')
        
        # Subscribers
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
        
        # Decision-making state
        self.current_sense_data = None
        self.current_memory_data = None
        self.last_sense_timestamp = None
        self.last_memory_timestamp = None
        
        # Action history and planning
        self.action_history = []
        self.current_goals = []
        self.action_sequence_id = 0
        
        # Decision parameters
        self.decision_threshold = 0.5  # Minimum confidence to act
        self.max_action_rate = 2.0     # Max actions per second
        self.last_action_time = 0
        
        # Timer for periodic decision making
        self.decision_timer = self.create_timer(0.5, self.make_decision)
        
        self.get_logger().info('Action Node initialized - ready to process inputs and make decisions')
    
    def sense_data_callback(self, msg):
        """
        Process incoming sensory data for immediate action consideration.
        
        Args:
            msg (std_msgs/String): Direct sensory data
        """
        try:
            sense_data = json.loads(msg.data)
            self.current_sense_data = sense_data
            self.last_sense_timestamp = sense_data['timestamp']
            
            # Check for immediate action triggers
            self._check_reactive_actions(sense_data)
            
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse sense data JSON')
        except Exception as e:
            self.get_logger().error(f'Error processing sense data: {str(e)}')
    
    def memory_data_callback(self, msg):
        """
        Process memory updates for deliberative action planning.
        
        Args:
            msg (std_msgs/String): Memory data from memory node
        """
        try:
            memory_data = json.loads(msg.data)
            self.current_memory_data = memory_data
            
            if 'memory_entry' in memory_data:
                self.last_memory_timestamp = memory_data['memory_entry']['timestamp']
            
            # Update planning based on memory insights
            self._update_planning(memory_data)
            
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse memory data JSON')
        except Exception as e:
            self.get_logger().error(f'Error processing memory data: {str(e)}')
    
    def _check_reactive_actions(self, sense_data):
        """
        Check for immediate reactive actions based on sensory input.
        
        Args:
            sense_data (dict): Current sensory information
        """
        # Simple reactive behaviors based on environmental conditions
        env = sense_data.get('data', {}).get('environment', {})
        
        # Example reactive rules
        reactions = []
        
        # React to extreme light conditions
        light_level = env.get('light_level', 0.5)
        if light_level > 0.9:
            reactions.append({
                'type': 'reactive',
                'action': 'adjust_brightness',
                'urgency': 'high',
                'reason': 'excessive_light'
            })
        elif light_level < 0.1:
            reactions.append({
                'type': 'reactive',
                'action': 'increase_illumination',
                'urgency': 'medium',
                'reason': 'insufficient_light'
            })
        
        # React to high noise levels
        noise_level = env.get('noise_level', 0.25)
        if noise_level > 0.4:
            reactions.append({
                'type': 'reactive',
                'action': 'reduce_noise_sensitivity',
                'urgency': 'medium',
                'reason': 'high_ambient_noise'
            })
        
        # Execute reactive actions immediately
        for reaction in reactions:
            self._execute_action(reaction, source='reactive')
    
    def _update_planning(self, memory_data):
        """
        Update action planning based on memory insights.
        
        Args:
            memory_data (dict): Memory information
        """
        if memory_data.get('type') == 'memory_consolidation':
            # Long-term memory updates might change our goals
            ltm_size = memory_data.get('ltm_size', 0)
            if ltm_size > 0 and ltm_size % 10 == 0:  # Every 10 consolidated memories
                self.current_goals.append({
                    'goal': 'analyze_memory_patterns',
                    'priority': 0.6,
                    'created_at': datetime.now().isoformat(),
                    'reason': f'Reached {ltm_size} consolidated memories'
                })
        
        elif memory_data.get('type') == 'memory_update':
            # Working memory updates affect immediate planning
            working_memory = memory_data.get('working_memory', {})
            importance = working_memory.get('current_importance', 0)
            
            if importance > 0.7:  # High importance memory
                self.current_goals.append({
                    'goal': 'investigate_important_stimulus',
                    'priority': importance,
                    'created_at': datetime.now().isoformat(),
                    'reason': 'High importance memory detected'
                })
    
    def make_decision(self):
        """
        Main decision-making loop - called periodically to evaluate situation and plan actions.
        
        This implements a simplified cognitive decision-making process:
        1. Assess current situation (sense + memory)
        2. Evaluate goals and priorities
        3. Select appropriate actions
        4. Execute actions
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_action_time < (1.0 / self.max_action_rate):
            return
        
        # Need both sense and memory data to make informed decisions
        if self.current_sense_data is None or self.current_memory_data is None:
            return
        
        # Assess current situation
        situation_assessment = self._assess_situation()
        
        # Select action based on assessment
        planned_action = self._select_action(situation_assessment)
        
        if planned_action:
            self._execute_action(planned_action, source='deliberative')
            self.last_action_time = current_time
    
    def _assess_situation(self):
        """
        Assess the current situation based on available sensory and memory data.
        
        Returns:
            dict: Situation assessment
        """
        # Combine sensory and memory information
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.0,
            'urgency': 0.0,
            'opportunities': [],
            'threats': [],
            'active_goals': len(self.current_goals)
        }
        
        # Analyze sensory data
        if self.current_sense_data:
            env = self.current_sense_data.get('data', {}).get('environment', {})
            
            # Environmental stability assessment
            light_variance = abs(env.get('light_level', 0.5) - 0.5)
            noise_variance = abs(env.get('noise_level', 0.25) - 0.25)
            
            if light_variance > 0.3 or noise_variance > 0.15:
                assessment['urgency'] += 0.3
                assessment['threats'].append('environmental_instability')
            else:
                assessment['confidence'] += 0.4
                assessment['opportunities'].append('stable_environment')
        
        # Analyze memory data
        if self.current_memory_data:
            working_memory = self.current_memory_data.get('working_memory', {})
            importance = working_memory.get('current_importance', 0)
            
            assessment['confidence'] += importance * 0.6
            
            if importance > 0.8:
                assessment['opportunities'].append('high_value_information')
        
        # Goal-based assessment
        if self.current_goals:
            highest_priority = max(goal['priority'] for goal in self.current_goals)
            assessment['urgency'] += highest_priority * 0.4
        
        assessment['confidence'] = min(1.0, assessment['confidence'])
        assessment['urgency'] = min(1.0, assessment['urgency'])
        
        return assessment
    
    def _select_action(self, situation_assessment):
        """
        Select an appropriate action based on situation assessment.
        
        Args:
            situation_assessment (dict): Current situation analysis
            
        Returns:
            dict or None: Selected action or None if no action needed
        """
        # Only act if we have sufficient confidence
        if situation_assessment['confidence'] < self.decision_threshold:
            return None
        
        # Priority-based action selection
        urgency = situation_assessment['urgency']
        opportunities = situation_assessment['opportunities']
        threats = situation_assessment['threats']
        
        # Handle threats first
        if threats and urgency > 0.6:
            return {
                'type': 'defensive',
                'action': 'mitigate_threat',
                'target': threats[0],
                'priority': urgency,
                'reason': f'Responding to threat: {threats[0]}'
            }
        
        # Pursue opportunities
        if opportunities and situation_assessment['confidence'] > 0.7:
            return {
                'type': 'explorative',
                'action': 'pursue_opportunity',
                'target': opportunities[0],
                'priority': situation_assessment['confidence'],
                'reason': f'Pursuing opportunity: {opportunities[0]}'
            }
        
        # Goal-directed action
        if self.current_goals:
            highest_goal = max(self.current_goals, key=lambda g: g['priority'])
            return {
                'type': 'goal_directed',
                'action': 'work_toward_goal',
                'target': highest_goal['goal'],
                'priority': highest_goal['priority'],
                'reason': f"Working toward goal: {highest_goal['goal']}"
            }
        
        # Default exploratory action
        return {
            'type': 'exploratory',
            'action': 'gather_information',
            'target': 'environment',
            'priority': 0.3,
            'reason': 'No specific goals, exploring environment'
        }
    
    def _execute_action(self, action, source='deliberative'):
        """
        Execute the selected action and publish action command.
        
        Args:
            action (dict): Action to execute
            source (str): Source of the action ('reactive' or 'deliberative')
        """
        # Create action command
        action_command = {
            'action_id': self.action_sequence_id,
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'action': action,
            'execution_status': 'initiated'
        }
        
        # Add to action history
        self.action_history.append(action_command)
        
        # Keep action history manageable
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-50:]
        
        # Remove completed goals
        if action.get('type') == 'goal_directed':
            self.current_goals = [g for g in self.current_goals 
                                 if g['goal'] != action.get('target')]
        
        # Publish action
        msg = String()
        msg.data = json.dumps(action_command, indent=2)
        self.action_publisher.publish(msg)
        
        self.get_logger().info(
            f'Executed {source} action #{self.action_sequence_id}: '
            f'{action.get("action", "unknown")} (priority: {action.get("priority", 0):.2f})'
        )
        
        self.action_sequence_id += 1


def main(args=None):
    """Main entry point for the action node."""
    rclpy.init(args=args)
    
    action_node = ActionNode()
    
    try:
        rclpy.spin(action_node)
    except KeyboardInterrupt:
        action_node.get_logger().info('Action Node shutting down...')
    finally:
        action_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 