import unittest
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chat.OwlCoT import OwlInnerVoice
from chat.react import Actor
from utils.Messages import UserMessage
from utils.workingMemory import WorkingMemory
from utils.llm_api import LLM
from unittest.mock import Mock

# Add missing action string constants
act_prefix_string = """Available actions:
1. Answer: Provide a direct response
2. Library: Search academic sources
3. Google: Search web sources
4. Article: Retrieve specific article
5. Review: Review existing information
"""

act_string = """Choose an action and respond in format:
<Act>action_name</Act>
<Target>recipient_name</Target>
<Content>action content</Content>
"""

class TestReact(unittest.TestCase):
    def setUp(self):
        """Set up test environment with real LLM"""
        self.llm = LLM('local')
        
        # Create minimal OwlCoT mock with required methods
        self.cot = OwlInnerVoice()
        self.cot.llm = self.llm
        
        # Create test actor
        self.actor = Actor(
            name="TestActor",
            cot=self.cot,
            character_description="A helpful AI assistant focused on clear communication and accurate information."
        )
        
        # Initialize working memory with name
        self.actor.wm = WorkingMemory(name="TestActor_wm")
        self.actor.library_questions = []

    def test_task_flow(self):
        """Test basic task handling with OTP"""
        sender = type('Sender', (), {'name': 'User'})()
        response = self.actor.task(
            sender=sender,
            act="says",
            task_text="Hello there"
        )
        
        self.assertTrue(self.mock_cot.llm.ask.called)
        self.assertIn("says to", response)

    def test_research_flow(self):
        """Test research task handling"""
        sender = type('Sender', (), {'name': 'User'})()
        response = self.actor.task(
            sender=sender,
            act="asks",
            task_text="Complex research question"
        )
        
        # Should trigger research or response
        self.assertTrue(
            "library search" in response.lower() or 
            "says to" in response.lower()
        )

    def test_action_selection(self):
        """Test action selection and execution"""
        sender = type('Sender', (), {'name': 'User'})()
        response = self.actor.task(
            sender=sender,
            act="says",
            task_text="Test query"
        )
        
        # Should result in some kind of response
        self.assertTrue(
            "says to" in response or
            "library search" in response or
            "research" in response
        )

    def test_memory_integration(self):
        """Test memory integration"""
        sender = type('Sender', (), {'name': 'User'})()
        
        # Add test memory and wait for state update
        self.actor.add_to_history("Previous conversation")
        self.actor.generate_state()  # Ensure state is updated
        
        # Test memory recall
        memories = self.actor.selective_recall("conversation")
        self.assertIn("Previous conversation", memories)

if __name__ == '__main__':
    unittest.main() 