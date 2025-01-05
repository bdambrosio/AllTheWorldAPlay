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

    def test_memory_retrieval_by_drive(self):
        """Test retrieving memories relevant to a drive"""
        drive = "hunting bugs. exploring"
        memories = self.agh.memory_retrieval.get_by_drive(
            self.agh.structured_memory,
            drive,
            threshold=0.5
        )
        self.assertTrue(len(memories) > 0)
        self.assertTrue(any("bug" in mem.text for mem in memories))

    def test_memory_retrieval_by_context(self):
        """Test retrieving memories by context"""
        context = "catching insects"
        memories = self.agh.memory_retrieval.get_by_context(
            self.agh.structured_memory,
            context,
            threshold=0.5
        )
        self.assertTrue(len(memories) > 0)
        self.assertTrue(any("bug" in mem.text for mem in memories))

    def test_memory_consolidation(self):
        """Test memory consolidation process"""
        initial_count = len(self.agh.structured_memory.get_all())
        
        # Run consolidation
        self.agh.memory_consolidator.consolidate(
            self.agh.structured_memory,
            ["hunting bugs. exploring"],
            self.agh.character
        )
        
        # Check consolidation effects
        consolidated = self.agh.structured_memory.get_all()
        self.assertGreater(len(consolidated), initial_count, 
                          "Consolidation should create new abstract memories")
        
        # Check memory links
        memory_links = sum(len(mem.related_memories) for mem in consolidated)
        self.assertGreater(memory_links, 0, 
                          "Consolidation should create memory links")

    def test_memory_importance(self):
        """Test memory importance scoring"""
        high_importance_memory = MemoryEntry(
            text="Found a dangerous spider",
            timestamp=datetime.now(),
            importance=0.9,
            confidence=1.0
        )
        self.agh.structured_memory.add_entry(high_importance_memory)
        
        # Test that high importance memories are prioritized in retrieval
        memories = self.agh.memory_retrieval.get_by_context(
            self.agh.structured_memory,
            "dangerous creatures",
            threshold=0.5
        )
        self.assertTrue(any(mem.importance > 0.8 for mem in memories))

    def test_recent_memory_retrieval(self):
        """Test retrieving recent memories"""
        old_memory = MemoryEntry(
            text="Old memory from yesterday",
            timestamp=datetime.now() - timedelta(days=1),
            importance=0.5,
            confidence=1.0
        )
        self.agh.structured_memory.add_entry(old_memory)
        
        recent_memories = self.agh.memory_retrieval.get_recent_relevant(
            self.agh.structured_memory,
            "any context",
            time_window=timedelta(hours=1)
        )
        
        self.assertTrue(all(
            (datetime.now() - mem.timestamp) < timedelta(hours=1)
            for mem in recent_memories
        ))

    def test_memory_embedding(self):
        """Test memory embedding generation and similarity"""
        memory1 = MemoryEntry(
            text="Playing with a red ball",
            timestamp=datetime.now(),
            importance=0.5,
            confidence=1.0
        )
        memory2 = MemoryEntry(
            text="Found a blue toy ball",
            timestamp=datetime.now(),
            importance=0.5,
            confidence=1.0
        )
        
        self.agh.structured_memory.add_entry(memory1)
        self.agh.structured_memory.add_entry(memory2)
        
        # Test that similar memories have high similarity scores
        memories = self.agh.memory_retrieval.get_by_context(
            self.agh.structured_memory,
            "playing with balls",
            threshold=0.5
        )
        self.assertGreaterEqual(len(memories), 2)

if __name__ == '__main__':
    unittest.main()