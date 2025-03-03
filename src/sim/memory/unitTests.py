import os, sys

from src.sim.cognitive.driveSignal import Drive
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sim.agh import Agh
from sim.context import Context

import unittest
from datetime import datetime, timedelta
import numpy as np
from utils.llm_api import LLM
from sim.memory.core import MemoryEntry, AbstractMemory, StructuredMemory
from sim.memory.consolidation import MemoryConsolidator
from sim.memory.core import MemoryRetrieval
from sim.memory.core import NarrativeSummary

class TestMemorySystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.memory = StructuredMemory()
        self.memory.owner = Agh('TestChar','I am a test character',server='local')
        meow_meow = Agh('Meow-Meow','I am a cat',server='local')
        self.memory.owner.context = Context([self.memory.owner, meow_meow], situation='test', step='1 hour')
        self.llm = LLM('local')
        self.consolidator = MemoryConsolidator(self.llm, self.memory.owner.context)
        
        # Create test drives
        self.test_drives = [
            Drive("find food and water"),
            Drive("stay safe from threats"),
            Drive("explore the environment")
        ]
        
        # Create test memories about a coherent activity using simulation time
        base_time = datetime(2024, 1, 1, 12, 0)  # Fixed simulation start time
        self.test_memories = [
            ("I see a mouse in the corner", 0.5),
            ("I slowly creep toward the mouse", 0.6),
            ("The mouse is eating some crumbs", 0.5),
            ("I pounce at the mouse but miss", 0.7),
            ("The mouse escapes under the cabinet", 0.6)
        ]
        
        self.memory_ids = []
        for i, (text, importance) in enumerate(self.test_memories):
            mem = MemoryEntry(
                text=text,
                timestamp=base_time + timedelta(minutes=i*5),
                importance=importance,
                confidence=1.0
            )
            self.memory_ids.append(self.memory.add_entry(mem))

    def test_concrete_memory_operations(self):
        """Test basic concrete memory operations"""
        # Test adding and retrieving
        self.assertEqual(len(self.memory.get_all()), 5)
        
        # Test recent memory retrieval
        recent = self.memory.get_recent(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].text, "The mouse escapes under the cabinet")
        
        # Test memory linking
        self.assertTrue(self.memory.link_memories(self.memory_ids[0], self.memory_ids[1]))
        mem1 = self.memory.get_by_id(self.memory_ids[0])
        self.assertIn(self.memory_ids[1], mem1.related_memories)

    def test_abstract_memory_formation(self):
        """Test abstract memory creation and management"""
        # Should have created abstract memories from test data
        current = self.memory.get_active_abstraction()
        self.assertIsNotNone(current)
        
        # Add a related memory
        similar_memory = MemoryEntry(
            text="I look around for where the mouse went",
            timestamp=datetime.now(),
            importance=0.6,
            confidence=1.0
        )
        self.memory.add_entry(similar_memory)
        
        # Should still be part of same abstraction
        current = self.memory.get_active_abstraction()
        self.assertIn(similar_memory.memory_id, current.instances)
        
        # Add very different memory to force new abstraction
        different_memory = MemoryEntry(
            text="I decide to take a nap in the sunbeam",
            timestamp=datetime.now(),
            importance=0.5,
            confidence=1.0
        )
        self.memory.add_entry(different_memory)
        
        # Should have closed previous abstraction
        abstractions = self.memory.get_recent_abstractions()
        self.assertGreater(len(abstractions), 0)
        
        # Find the mouse-hunting abstraction
        mouse_abstraction = next((abs for abs in abstractions if "mouse" in abs.summary.lower()), None)
        self.assertIsNotNone(mouse_abstraction, "Mouse-hunting abstraction not found")
        self.assertFalse(mouse_abstraction.is_active)

    def test_consolidation(self):
        """Test memory consolidation operations"""
        # Run consolidation
        self.consolidator.consolidate(self.memory)
        
        # Check abstractions were processed
        abstractions = self.memory.get_recent_abstractions()
        self.assertGreater(len(abstractions), 0)
        
        # First abstraction should be about mouse hunting and either:
        # - still active (current) OR
        # - completed with 3+ instances
        first_abstract = abstractions[0]
        self.assertTrue(
            first_abstract.is_active or len(first_abstract.instances) >= 3,
            "Abstraction should either be active or have 3+ instances"
        )
        self.assertIn("mouse", first_abstract.summary.lower())

        # If it's completed, verify it has an end time
        if not first_abstract.is_active:
            self.assertIsNotNone(first_abstract.end_time)

    def test_retrieval(self):
        """Test memory retrieval system"""
        retrieval = MemoryRetrieval()
        
        # Test semantic search
        results = retrieval.get_relevant_memories(
            memory=self.memory,
            query="hunting mice",
            threshold=0.3
        )
        
        # Should find both concrete and abstract memories about the mouse
        self.assertGreater(len(results['concrete']), 0)
        self.assertGreater(len(results['abstract']), 0)

    def test_memory_cleanup(self):
        """Test cleanup of concrete memories after abstraction"""
        # Initial count
        initial_count = len(self.memory.concrete_memories)
        
        # Force memories to be "old" relative to simulation time
        old_time = datetime(2024, 1, 1, 1, 0)  # 11 hours before base_time
        for mem in self.memory.concrete_memories:
            mem.timestamp = old_time
        
        # Add one high importance memory that should be preserved
        preserved_mem = MemoryEntry(
            text="A very important mouse observation",
            timestamp=old_time,
            importance=0.9,  # High importance
            confidence=1.0
        )
        self.memory.add_entry(preserved_mem)
        
        # Add new memory to force abstraction closure
        new_mem = MemoryEntry(
            text="I decide to take a nap",
            timestamp=datetime(2024, 1, 1, 13, 0),
            importance=0.5,
            confidence=1.0
        )
        self.memory.add_entry(new_mem)
        
        # Run consolidation
        self.consolidator.consolidate(self.memory)
        
        # Check cleanup
        remaining = len(self.memory.concrete_memories)
        self.assertLess(remaining, initial_count + 2,  # +2 for the memories we added
                        "Some concrete memories should be cleaned up")
        
        # Verify high importance memory remains
        self.assertTrue(any(mem.importance >= 0.9 for mem in self.memory.concrete_memories),
                       "High importance memory should be preserved")

    def test_abstract_memory_validation(self):
        """Test abstract memory formation and concrete cleanup"""
        # Get initial abstract memory
        abstracts = self.memory.get_recent_abstractions()
        self.assertEqual(len(abstracts), 1)
        
        # Add unrelated memory to force new abstraction
        new_mem = MemoryEntry(
            text="I decide to take a nap",
            timestamp=datetime(2024, 1, 1, 13, 0),
            importance=0.5,
            confidence=1.0
        )
        self.memory.add_entry(new_mem)
        
        # Previous abstraction should be closed
        mouse_abstract = next(abs for abs in self.memory.get_recent_abstractions() 
                             if "mouse" in abs.summary.lower())
        self.assertFalse(mouse_abstract.is_active)
        
        # Original concrete memories should be marked for cleanup
        self.assertTrue(len(self.memory.pending_cleanup) > 0)
        
        # Run consolidation
        self.consolidator.consolidate(self.memory)
        
        # Check cleanup occurred
        self.assertEqual(len(self.memory.pending_cleanup), 0)

    def test_narrative_creation_and_update(self):
        """Test narrative creation and updating"""
        # Create initial narrative
        base_time = datetime(2024, 1, 1, 12, 0)
        narrative = NarrativeSummary(
            recent_events="",
            ongoing_activities="",
            background="",
            last_update=base_time - timedelta(hours=5),  # Force update needed
            active_drives=["explore", "socialize"]
        )
        
        # Add some test memories with character interactions
        interaction_memories = [
            ("I met Meow-Meow in the garden", 0.7),
            ("Meow-Meow showed me a butterfly", 0.6),
            ("I played chase with Meow-Meow", 0.8),
            ("Meow-Meow needed a rest from playing", 0.6),
            ("I watched Meow-Meow nap", 0.5)
        ]
        
        for i, (text, importance) in enumerate(interaction_memories):
            mem = MemoryEntry(
                text=text,
                timestamp=base_time + timedelta(minutes=i*30),
                importance=importance,
                confidence=1.0
            )
            self.memory.add_entry(mem)
        
        # Create character description
        character_desc = "I am a pale grey kitten named Lemon. I love playing and exploring."
        
        # Update narrative
        self.consolidator.update_cognitive_model(
            memory=self.memory,
            narrative=narrative,
            knownActorManager=self.memory.owner.known_actors,
            current_time=base_time + timedelta(hours=4),
            character_desc=character_desc
        )
        
        # Test narrative components
        self.assertTrue(narrative.background, "Background should be populated")
        self.assertTrue(narrative.recent_events, "Recent events should be populated")
        self.assertTrue(narrative.ongoing_activities, "Activities should be populated")
        
        # Test relationship tracking
        self.assertIn("Meow-Meow", narrative.key_relationships,
                     "Should track relationship with Meow-Meow")
        
        # Test narrative coherence
        self.assertIn("Meow-Meow", narrative.recent_events,
                     "Recent events should mention key character")
        
        # Test update timing
        self.assertEqual(
            narrative.last_update,
            base_time + timedelta(hours=4),
            "Update time should be set correctly"
        )

    def test_narrative_update_conditions(self):
        """Test conditions for narrative updates"""
        base_time = datetime(2024, 1, 1, 12, 0)
        narrative = NarrativeSummary(
            recent_events="Original events",
            ongoing_activities="Original activities",
            background="Original background",
            last_update=base_time,
            active_drives=[]
        )
        
        # Try update too soon (within update_interval)
        self.consolidator.update_cognitive_model(
            memory=self.memory,
            narrative=narrative,
            knownActorManager=self.memory.owner.known_actors,
            current_time=base_time + timedelta(hours=1),
            character_desc="Test character"
        )
        
        # Should not update
        self.assertEqual(narrative.recent_events, "Original events",
                        "Should not update before interval")
        
        # Try update after interval
        self.consolidator.update_cognitive_model(
            memory=self.memory,
            narrative=narrative,
            knownActorManager=self.memory.owner.known_actors,
            current_time=base_time + timedelta(hours=5),
            character_desc="Test character"
        )
        
        # Should update
        self.assertNotEqual(narrative.recent_events, "Original events",
                           "Should update after interval")

    def test_drive_memory_retrieval(self):
        """Test retrieving memories related to drives"""
        # Now uses memory_retrieval directly
        retrieval = self.memory.owner.memory_retrieval
        
        food_memories = retrieval.get_by_drive(
            memory=self.memory,
            drive=self.test_drives[0],
            threshold=0.1
        )
        
        # Should find mouse-related memories
        self.assertTrue(any("mouse" in mem.text.lower() for mem in food_memories))
        
        # Test age weighting
        newest_first = retrieval.get_by_drive(
            memory=self.memory,
            drive=self.test_drives[0],
            threshold=0.1
        )
        
        # More recent memories should have higher scores
        self.assertEqual(newest_first[0].text, "The mouse is eating some crumbs")

if __name__ == '__main__':
    unittest.main()