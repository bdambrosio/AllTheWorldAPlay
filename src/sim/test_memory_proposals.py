#!/usr/bin/env python3
"""
Test script for the new memory consolidation proposals:
1. Auto-abstraction on add_entry
2. Cleanup of old concrete memories
3. Unified return types for all retrieval methods
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime, timedelta
from sim.memory.core import StructuredMemory, MemoryEntry, AbstractMemory
from sim.context import Context

def test_auto_abstraction():
    """Test that abstractions are created automatically on add_entry"""
    print("Testing auto-abstraction on add_entry...")
    
    # Create a mock context with simulation time
    class MockContext:
        def __init__(self):
            self.simulation_time = datetime.now()
    
    class MockOwner:
        def __init__(self):
            self.context = MockContext()
    
    # Create memory system
    memory = StructuredMemory(owner=MockOwner())
    
    # Add similar memories
    base_time = datetime.now()
    
    # Add first memory
    entry1 = MemoryEntry(
        text="Maya paints a landscape in her studio",
        timestamp=base_time,
        importance=0.8,
        confidence=0.9
    )
    memory.add_entry(entry1)
    
    # Add similar memory
    entry2 = MemoryEntry(
        text="Maya works on her painting with careful brushstrokes",
        timestamp=base_time + timedelta(minutes=30),
        importance=0.7,
        confidence=0.8
    )
    memory.add_entry(entry2)
    
    # Add dissimilar memory
    entry3 = MemoryEntry(
        text="Maya goes to the grocery store to buy milk",
        timestamp=base_time + timedelta(hours=1),
        importance=0.5,
        confidence=0.9
    )
    memory.add_entry(entry3)
    
    # Add another similar memory to painting
    entry4 = MemoryEntry(
        text="Maya cleans her brushes after painting session",
        timestamp=base_time + timedelta(hours=2),
        importance=0.6,
        confidence=0.8
    )
    memory.add_entry(entry4)
    
    print(f"Added {len(memory.concrete_memories)} concrete memories")
    print(f"Created {len(memory.abstract_memories)} abstract memories")
    
    # Check that abstractions were created
    assert len(memory.abstract_memories) > 0, "Should have created at least one abstraction"
    
    # Print abstractions
    for i, abstract in enumerate(memory.abstract_memories):
        print(f"Abstract {i+1}: {abstract.summary}")
        print(f"  Instances: {len(abstract.instances)}")
        print(f"  Time range: {abstract.start_time} to {abstract.end_time}")
    
    print("✓ Auto-abstraction test passed\n")

def test_cleanup():
    """Test cleanup of old concrete memories with importance preservation"""
    print("Testing cleanup of old concrete memories...")
    
    class MockContext:
        def __init__(self):
            self.simulation_time = datetime.now()
    
    class MockOwner:
        def __init__(self):
            self.context = MockContext()
    
    memory = StructuredMemory(owner=MockOwner())
    memory.cleanup_age_hours = 1  # Set short cleanup time for testing
    
    # Add old memories with different importance levels
    old_time = datetime.now() - timedelta(hours=2)
    
    # Low importance - should be cleaned up
    entry1 = MemoryEntry(
        text="Old low-importance memory about painting",
        timestamp=old_time,
        importance=0.5,  # Below 0.67 threshold
        confidence=0.8
    )
    memory.add_entry(entry1)
    
    # Medium importance - should be cleaned up
    entry2 = MemoryEntry(
        text="Another old medium-importance memory about painting",
        timestamp=old_time + timedelta(minutes=10),
        importance=0.6,  # Below 0.67 threshold
        confidence=0.7
    )
    memory.add_entry(entry2)
    
    # High importance - should be preserved even if old
    entry3 = MemoryEntry(
        text="Old high-importance memory about painting",
        timestamp=old_time + timedelta(minutes=20),
        importance=0.8,  # Above 0.67 threshold
        confidence=0.9
    )
    memory.add_entry(entry3)
    
    # Add recent memory
    recent_entry = MemoryEntry(
        text="Recent memory about cooking",
        timestamp=datetime.now() - timedelta(minutes=10),
        importance=0.8,
        confidence=0.9
    )
    memory.add_entry(recent_entry)
    
    print(f"Total concrete memories after cleanup: {len(memory.concrete_memories)}")
    print(f"Total abstract memories: {len(memory.abstract_memories)}")
    
    # Check that recent memory still exists
    recent_found = any(mem.text == "Recent memory about cooking" for mem in memory.concrete_memories)
    assert recent_found, "Recent memory should still exist"
    
    # Check that high-importance old memory is preserved
    high_importance_found = any(mem.text == "Old high-importance memory about painting" for mem in memory.concrete_memories)
    assert high_importance_found, "High-importance old memory should be preserved"
    
    print("✓ Cleanup test passed\n")

def test_unified_retrieval():
    """Test that all retrieval methods return consistent types"""
    print("Testing unified retrieval interface...")
    
    class MockContext:
        def __init__(self):
            self.simulation_time = datetime.now()
    
    class MockOwner:
        def __init__(self):
            self.context = MockContext()
    
    memory = StructuredMemory(owner=MockOwner())
    
    # Add several memories to create abstractions
    base_time = datetime.now()
    
    for i in range(6):
        entry = MemoryEntry(
            text=f"Memory {i+1} about daily activities",
            timestamp=base_time + timedelta(minutes=i*10),
            importance=0.6,
            confidence=0.8
        )
        memory.add_entry(entry)
    
    # Test get_all returns mixed list
    all_memories = memory.get_all()
    print(f"get_all() returned {len(all_memories)} memories")
    assert isinstance(all_memories, list), "get_all should return list"
    
    # Test get_recent returns mixed list
    recent_memories = memory.get_recent(4)
    print(f"get_recent(4) returned {len(recent_memories)} memories")
    assert isinstance(recent_memories, list), "get_recent should return list"
    
    # Test to_string works on mixed list
    memory_strings = [mem.to_string() for mem in recent_memories]
    print(f"to_string() worked on {len(memory_strings)} memories")
    assert all(isinstance(s, str) for s in memory_strings), "All to_string results should be strings"
    
    # Test get_by_criteria
    high_importance = memory.get_by_criteria(min_importance=0.7)
    print(f"get_by_criteria(min_importance=0.7) returned {len(high_importance)} memories")
    assert isinstance(high_importance, list), "get_by_criteria should return list"
    
    # Test backward compatibility methods still work
    concrete_only = memory.get_recent_concrete(3)
    assert isinstance(concrete_only, list), "get_recent_concrete should return list"
    assert all(isinstance(mem, MemoryEntry) for mem in concrete_only), "get_recent_concrete should return only MemoryEntry objects"
    
    print("✓ Unified retrieval test passed\n")

def test_to_string():
    """Test to_string methods for both memory types"""
    print("Testing to_string methods...")
    
    # Test MemoryEntry to_string
    entry = MemoryEntry(
        text="Maya paints a beautiful landscape",
        timestamp=datetime.now(),
        importance=0.8,
        confidence=0.9
    )
    
    assert entry.to_string() == "Maya paints a beautiful landscape", "MemoryEntry to_string should return text"
    
    # Test AbstractMemory to_string
    abstract = AbstractMemory(
        summary="Pattern of painting activities",
        start_time=datetime.now(),
        instances=[1, 2, 3],
        importance=0.7
    )
    
    assert abstract.to_string() == "Pattern of painting activities", "AbstractMemory to_string should return summary"
    
    # Test mixed list processing
    mixed_memories = [entry, abstract]
    string_representations = [mem.to_string() for mem in mixed_memories]
    
    expected = ["Maya paints a beautiful landscape", "Pattern of painting activities"]
    assert string_representations == expected, "Mixed list should convert properly"
    
    print("✓ to_string test passed\n")

if __name__ == "__main__":
    print("Testing Memory Consolidation Proposals\n")
    print("=" * 50)
    
    try:
        test_auto_abstraction()
        test_cleanup()
        test_unified_retrieval()
        test_to_string()
        
        print("=" * 50)
        print("🎉 All tests passed! Memory consolidation proposals work correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 