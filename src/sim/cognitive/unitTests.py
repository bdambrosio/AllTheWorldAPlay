import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import unittest
from datetime import datetime
from typing import List, Dict
from utils.llm_api import LLM
from sim.cognitive.state import StateSystem, StateAssessment
from sim.cognitive.priority import PrioritySystem
from sim.memory.core import StructuredMemory, MemoryEntry

class TestCognitiveSystem(unittest.TestCase):
    def setUp(self):
        self.llm = LLM('local')
        self.character = "Test character description"
        self.memory = StructuredMemory()
        
        # Add some test memories
        self.memory.add_entry(MemoryEntry(
            text="Feeling hungry",
            timestamp=datetime.now(),
            importance=0.8,
            confidence=1.0
        ))
        self.memory.add_entry(MemoryEntry(
            text="Found some food",
            timestamp=datetime.now(),
            importance=0.6,
            confidence=1.0
        ))
        
        self.state_system = StateSystem(self.llm, self.character)
        self.priority_system = PrioritySystem(self.llm, self.character)
        
    def test_state_assessment(self):
        """Test state assessment generation"""
        drives = ["hunger", "rest"]
        situation = "In a forest with food nearby"
        
        state = self.state_system.generate_state(
            drives,
            situation,
            self.memory
        )
        
        # Verify state structure
        self.assertIsInstance(state, dict)
        for term, info in state.items():
            self.assertIn("drive", info)
            self.assertIn("state", info)
            self.assertIn("trigger", info)
            self.assertIn("termination_check", info)
            
    def test_priority_generation(self):
        """Test priority task generation"""
        drives = ["hunger", "rest"]
        state = {
            "hunger": {
                "drive": "hunger",
                "state": "high",
                "trigger": "haven't eaten",
                "termination_check": "food consumed"
            }
        }
        situation = "In a forest with food nearby"
        
        priorities = self.priority_system.update_priorities(
            drives,
            state,
            self.memory,
            situation
        )
        
        # Verify priorities
        self.assertIsInstance(priorities, list)
        self.assertTrue(len(priorities) > 0)
        # Each priority should be a valid XML plan
        for plan in priorities:
            self.assertIn("<Plan>", plan)
            self.assertIn("<Name>", plan)
            self.assertIn("<Steps>", plan)
            
    def test_error_handling(self):
        """Test system handles errors gracefully"""
        # Test with empty drives
        state = self.state_system.generate_state(
            [],
            "test situation",
            self.memory
        )
        self.assertEqual(state, {})
        
        # Test with invalid memories
        bad_memory = StructuredMemory()
        state = self.state_system.generate_state(
            ["test"],
            "test",
            bad_memory
        )
        self.assertIsInstance(state, dict)

if __name__ == '__main__':
    unittest.main() 