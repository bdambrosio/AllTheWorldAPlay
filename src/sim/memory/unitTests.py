import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
from sim.memory.core import MemoryEntry, StructuredMemory
from sim.memory.consolidation import MemoryConsolidator
from sim.memory.retrieval import MemoryRetrieval
from sim.agh import Agh

class TestMemorySystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.llm_mock = Mock()
        self.llm_mock.ask.return_value = """<Analysis>
            <Abstractions>
                <Group>
                    <Timestamps>2024-01-01 10:00:00,2024-01-01 10:01:00</Timestamps>
                    <Abstract>Combined memory of events</Abstract>
                </Group>
            </Abstractions>
            <Links>
                <Link>2024-01-01 10:00:00,2024-01-01 10:02:00</Link>
            </Links>
        </Analysis>"""
        
        # Create test agent with mock LLM
        self.agh = Agh("TestAgent", "Test description", server='local')
        self.agh.llm = self.llm_mock
        
        # Create test memories
        self.memories = [
            MemoryEntry(
                text="Saw a bug on the floor",
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
                importance=0.5,
                confidence=1.0
            ),
            MemoryEntry(
                text="Chased the bug around",
                timestamp=datetime(2024, 1, 1, 10, 1, 0),
                importance=0.6,
                confidence=1.0
            ),
            MemoryEntry(
                text="Caught the bug",
                timestamp=datetime(2024, 1, 1, 10, 2, 0),
                importance=0.8,
                confidence=1.0
            )
        ]
        
        for memory in self.memories:
            self.agh.structured_memory.add_entry(memory)

    def test_memory_retrieval(self):
        """Test basic memory retrieval as used by Agh"""
        # Test get_recent
        recent = self.agh.structured_memory.get_recent(5)
        self.assertLessEqual(len(recent), 5)
        
        # Test memory consolidation with drives
        self.agh.memory_consolidator.consolidate(
            self.agh.structured_memory,
            self.agh.drives,
            self.agh.character
        )

    def test_memory_formatting(self):
        """Test memory text formatting as used by Agh"""
        recent_memories = self.agh.structured_memory.get_recent(5)
        memory_text = '\n'.join(memory.text for memory in recent_memories)
        self.assertIsInstance(memory_text, str)

if __name__ == '__main__':
    unittest.main()