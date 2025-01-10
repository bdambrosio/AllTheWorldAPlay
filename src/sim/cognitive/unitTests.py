import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import unittest
from datetime import datetime, timedelta
from typing import List, Dict
from utils.llm_api import LLM
from sim.cognitive.state import StateSystem, StateAssessment
from sim.cognitive.priority import PrioritySystem
from sim.cognitive.processor import CognitiveProcessor, CognitiveState
from sim.memory.core import StructuredMemory, MemoryEntry
from sim.context import Context

class MockContext:
    def __init__(self, time=None):
        self.simulation_time = time or datetime(2024, 1, 1, 12, 0)

class MockAgent:
    def __init__(self, context):
        self.context = context

class TestCognitiveSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.llm = LLM('local')
        self.character = "Test character description"
        
        # Set up context with simulation time
        self.context = MockContext()
        self.agent = MockAgent(self.context)
        self.memory = StructuredMemory(owner=self.agent)
        
        # Add test memories at different simulation times
        base_time = self.context.simulation_time
        self.memory.add_entry(MemoryEntry(
            text="Feeling hungry",
            timestamp=base_time - timedelta(hours=1),
            importance=0.8,
            confidence=1.0
        ))
        self.memory.add_entry(MemoryEntry(
            text="Found some food",
            timestamp=base_time - timedelta(minutes=30),
            importance=0.6,
            confidence=1.0
        ))
        
        # Initialize cognitive systems
        self.state_system = StateSystem(self.llm, self.character)
        self.priority_system = PrioritySystem(self.llm, self.character)
        self.processor = CognitiveProcessor(self.llm, self.character)

    def test_state_assessment_timing(self):
        """Test state assessment with simulation time"""
        drives = ["hunger", "rest"]
        situation = "In a forest with food nearby"
        
        state = self.state_system.generate_state(
            drives=drives,
            situation=situation,
            memory=self.memory
        )
        
        # Verify state timestamps
        for term, info in state.items():
            self.assertIn("timestamp", info)
            self.assertEqual(info["timestamp"], self.context.simulation_time)

    def test_priority_time_window(self):
        """Test priority system uses correct time window"""
        drives = ["hunger"]
        state = {
            "hunger": {
                "drive": "hunger",
                "state": "high",
                "trigger": "haven't eaten",
                "termination_check": "food consumed",
                "timestamp": self.context.simulation_time
            }
        }
        
        priorities = self.priority_system.update_priorities(
            drives=drives,
            state=state,
            memory=self.memory,
            situation="Test situation"
        )
        
        # Verify priorities generated
        self.assertIsInstance(priorities, list)
        self.assertTrue(len(priorities) > 0)

    def test_cognitive_processor_integration(self):
        """Test cognitive processor handles time and memory correctly"""
        cognitive_state = CognitiveState(
            state={},
            active_priorities=["hunger", "rest"]  # Initial drives/priorities
        )
        
        new_state = self.processor.process_cognitive_update(
            cognitive_state=cognitive_state,
            memory=self.memory,
            current_situation="Test situation",
            step="4 hours"
        )
        
        # Verify state and priorities updated
        self.assertTrue(new_state.state)
        self.assertTrue(new_state.active_priorities)

    def test_memory_time_filtering(self):
        """Test memory filtering by simulation time"""
        # Add a future memory that shouldn't be considered
        future_time = self.context.simulation_time + timedelta(hours=1)
        self.memory.add_entry(MemoryEntry(
            text="Future event",
            timestamp=future_time,
            importance=0.5,
            confidence=1.0
        ))
        
        drives = ["hunger"]
        state = self.state_system.generate_state(
            drives=drives,
            situation="Test situation",
            memory=self.memory
        )
        
        # Future memories shouldn't influence current state
        self.assertNotIn("future", str(state).lower())

    def test_state_transition(self):
        """Test state transitions over simulation time"""
        # Initial state - make hunger more severe
        self.memory.add_entry(MemoryEntry(
            text="Haven't eaten all day, feeling very hungry and weak",
            timestamp=self.context.simulation_time - timedelta(hours=6),
            importance=0.9,
            confidence=1.0
        ))
        
        drives = ["hunger"]
        initial_state = self.state_system.generate_state(
            drives=drives,
            situation="Very hungry in forest, no food for hours",
            memory=self.memory
        )
        
        # Verify we got an initial state
        self.assertTrue(initial_state, "Should have initial state")
        initial_hunger = next((info for key, info in initial_state.items() 
                              if info['drive'] == 'hunger'), None)
        self.assertIsNotNone(initial_hunger, "Should have hunger state")
        
        # Add multiple memories about satisfying hunger
        self.memory.add_entry(MemoryEntry(
            text="Found lots of ripe berries",
            timestamp=self.context.simulation_time,
            importance=0.7,
            confidence=1.0
        ))
        self.memory.add_entry(MemoryEntry(
            text="Ate many berries until full",
            timestamp=self.context.simulation_time + timedelta(minutes=10),
            importance=0.8,
            confidence=1.0
        ))
        self.memory.add_entry(MemoryEntry(
            text="Feeling satisfied after eating",
            timestamp=self.context.simulation_time + timedelta(minutes=15),
            importance=0.6,
            confidence=1.0
        ))
        
        # Advance simulation time
        self.context.simulation_time += timedelta(minutes=30)
        
        # Get new state
        new_state = self.state_system.generate_state(
            drives=drives,
            situation="In forest after eating many berries, feeling full",
            memory=self.memory
        )
        
        # Verify new state
        self.assertTrue(new_state, "Should have new state")
        new_hunger = next((info for key, info in new_state.items() 
                           if info['drive'] == 'hunger'), None)
        self.assertIsNotNone(new_hunger, "Should have hunger state")
        
        # States should be different
        self.assertNotEqual(
            initial_hunger['state'],
            new_hunger['state'],
            "Hunger state should change after eating"
        )

if __name__ == '__main__':
    unittest.main() 