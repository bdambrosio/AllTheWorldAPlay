import unittest
from unittest.mock import Mock, patch
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chat.OwlCoT import OwlInnerVoice

from chat.react import Actor
from utils.Messages import UserMessage
from utils.workingMemory import WorkingMemory
from utils.llm_api import LLM

class TestReact(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_cot = OwlInnerVoice()
        self.mock_cot.llm = LLM('local')
        self.mock_cot.script = Mock()
        
        # Create test actor
        self.actor = Actor(
            name="TestActor",
            cot=self.mock_cot,
            character_description="Test character"
        )
        

    def test_task_flow(self):
        """Test basic task handling with OTP"""
        # Test simple response task
        response = self.actor.task(
            sender=Mock(name="Sender"),
            act="says",
            task_text="Hello there"
        )
        
        # Verify OTP flow
        self.assertTrue(self.mock_cot.llm.ask.called)
        self.assertIn("says to", response)

    def test_research_flow(self):
        """Test research task handling"""
        # Set up mock to indicate need for research
        self.mock_cot.llm.ask.return_value = """
<Orient>
Insufficient information available.
</Orient>
<Thoughts>
Need to research topic.
</Thoughts>
"""
        
        # Mock library search results
        self.mock_cot.script.s2_search.return_value = (["test result"], ["test fact"])
        
        response = self.actor.task(
            sender=Mock(name="Sender"),
            act="asks",
            task_text="Complex research question"
        )
        
        # Verify research flow
        self.assertTrue(self.mock_cot.script.s2_search.called)
        self.assertIn("library search results", str(response))

    def test_action_selection(self):
        """Test action selection and execution"""
        
        response = self.actor.task(
            sender=Mock(name="Sender"),
            act="says",
            task_text="Test query"
        )
        
        # Verify action selection
        self.assertIn("Test response", response)

    def test_memory_integration(self):
        """Test memory integration"""
        # Add test memory
        self.actor.add_to_history("Previous conversation")
        
        # Test memory recall
        memories = self.actor.selective_recall("conversation")
        self.assertIn("Previous conversation", memories)
        
        # Test memory use in task
        response = self.actor.task(
            sender=Mock(name="Sender"),
            act="says",
            task_text="What did we discuss?"
        )
        
        self.assertTrue(self.mock_cot.llm.ask.called)
        self.assertIn("says to", response)

if __name__ == '__main__':
    unittest.main() 