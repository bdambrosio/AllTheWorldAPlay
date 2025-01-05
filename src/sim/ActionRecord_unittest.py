import unittest
from time import time
from sim.ActionRecord import (
    ActionMemory, 
    ActionRecord,
    StateSnapshot,
    Mode,
    create_action_record
)

class TestActionMemory(unittest.TestCase):
    def setUp(self):
        self.memory = ActionMemory()
        
        # Create sample states
        self.state1 = StateSnapshot(
            values={
                "find food": "high",
                "find shelter": "medium",
                "explore area": "low"
            },
            timestamp=time()
        )
        
        self.state2 = StateSnapshot(
            values={
                "find food": "medium",  # Improved (lower is better)
                "find shelter": "medium",  # Same
                "explore area": "high"   # Worse
            },
            timestamp=time()
        )
        
    def test_state_comparison(self):
        """Test state comparison logic"""
        helped, hindered = self.memory.compare_states(self.state1, self.state2)
        self.assertIn("find food", helped)
        self.assertNotIn("find shelter", helped)
        self.assertNotIn("find shelter", hindered)
        self.assertIn("explore area", hindered)
        
    def test_record_creation_and_storage(self):
        """Test basic record creation and storage"""
        record = ActionRecord(
            mode=Mode.DO,
            action_text="gather berries",
            source_task="find food",
            timestamp=time(),
            target=None,
            context_feedback="Found some berries",
            energy_used=2.0,
            state_before=self.state1,
            state_after=self.state2,
            helped_states=[],
            hindered_states=[]
        )
        
        self.memory.add_record(record)
        self.assertEqual(len(self.memory.records), 1)
        self.assertIn("find food", self.memory.task_sequences)
        
    def test_effectiveness_calculation(self):
        """Test task effectiveness scoring"""
        # Create a sequence of records with mixed results
        for i in range(5):
            record = ActionRecord(
                mode=Mode.DO,
                action_text=f"action {i}",
                source_task="find food",
                timestamp=time(),
                target=None,
                context_feedback="",
                energy_used=1.0,
                state_before=StateSnapshot(
                    values={"find food": "high"},
                    timestamp=time()
                ),
                state_after=StateSnapshot(
                    values={"find food": "medium" if i < 2 else "high"},
                    timestamp=time()
                ),
                helped_states=["find food"] if i < 2 else [],
                hindered_states=[]
            )
            self.memory.add_record(record)
            
        effectiveness = self.memory.get_task_effectiveness("find food")
        self.assertLess(effectiveness, 1.0)  # Should be reduced due to failed attempts
        
    def test_repetition_detection(self):
        """Test detection of repetitive actions"""
        # Add several similar actions
        for i in range(3):
            record = ActionRecord(
                mode=Mode.DO,
                action_text="gather wood for shelter",
                source_task="build shelter",
                timestamp=time(),
                target=None,
                context_feedback="",
                energy_used=1.0,
                state_before=StateSnapshot(
                    values={"find shelter": "high"},
                    timestamp=time()
                ),
                state_after=StateSnapshot(
                    values={"find shelter": "high"},  # No improvement
                    timestamp=time()
                ),
                helped_states=[],
                hindered_states=[]
            )
            self.memory.add_record(record)
            
        # Test if similar action would be considered repetitive
        is_repetitive = self.memory.is_action_repetitive(
            Mode.DO, 
            "gather more wood for shelter",
            "build shelter"
        )
        self.assertTrue(is_repetitive)

class TestAgentIntegration(unittest.TestCase):
    def setUp(self):
        # Create a minimal test agent
        class TestAgent:
            def __init__(self):
                self.name = "TestAgent"
                self.state = {
                    "find food": {"state": "high"},
                    "find shelter": {"state": "medium"}
                }
                self.active_task = ["find food"]
                self.show = ""
                self.action_memory = ActionMemory()
                
        self.agent = TestAgent()
        
    def test_record_creation(self):
        """Test creating action record from agent state"""
        record = create_action_record(
            agent=self.agent,
            mode=Mode.DO,
            action_text="search for berries",
            task_name="find food"
        )
        
        self.assertEqual(record.source_task, "find food")
        self.assertEqual(record.mode, Mode.DO)
        self.assertIn("find food", record.state_before.values)

if __name__ == '__main__':
    unittest.main()
    