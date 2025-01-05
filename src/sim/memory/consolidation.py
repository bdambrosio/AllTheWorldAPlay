# memory/consolidation.py
from typing import List, Dict, Tuple
from datetime import datetime
from utils.Messages import UserMessage
import utils.xml_utils as xml_utils
from .core import StructuredMemory, MemoryEntry
from .retrieval import MemoryRetrieval
import numpy as np
import utils.xml_utils as xml

class MemoryConsolidator:
    """Handles memory consolidation, abstraction, and organization"""
    
    def __init__(self, llm):
        self.llm = llm
        self.retrieval = MemoryRetrieval()
        
    def consolidate(self, 
                   memory: StructuredMemory,
                   drives: List[str],
                   character: str) -> None:
        """Consolidate recent memories and update memory structure"""
        
        # Skip if no new memories to consolidate
        if not memory.memory_buffer:
            return
            
        # Get memories to consolidate 
        memories_text = self._format_memories_for_llm(memory.memory_buffer)
        
        # For each drive, analyze relevant memories
        for drive in drives:
            # Get memories relevant to this drive's phrases
            drive_memories = self.retrieval.get_by_drive(memory, drive)
            if not drive_memories:
                continue
                
            # Generate consolidation analysis for this drive
            analysis = self._analyze_memories_for_drive(
                memories_text,
                drive,
                character
            )
            
            # Apply consolidation updates
            self._apply_consolidation(memory, analysis, drive_memories)
            
        # Clear buffer after consolidation
        memory.memory_buffer.clear()
        
    def _analyze_memories_for_drive(self, 
                                  memories_text: str,
                                  drive: str,
                                  character: str) -> str:
        """Generate consolidation analysis for memories related to a drive"""
        
        prompt = [UserMessage(content="""Analyze these recent memories in the context of the drive: "{drive}"

<Memories>
{{$memories}}
</Memories>

<Character>
{{$character}}
</Character>

Analyze these memories and suggest consolidation actions in XML format:

<Analysis>
  <Abstractions>
    <!-- Groups of memories that can be combined into higher-level memories -->
    <Group>
      <Timestamps>time1,time2,...</Timestamps>
      <Abstract>summarized higher-level memory</Abstract>
    </Group>
    ...
  </Abstractions>
  
  <Links>
    <!-- Pairs of related memories that should be linked -->
    <Link>time1,time2</Link>
    ...
  </Links>
  
  <ImportanceUpdates>
    <!-- Memories whose importance should be adjusted -->
    <Update>
      <Timestamp>time</Timestamp>
      <Adjustment>increase/decrease</Adjustment>
      <Reason>why importance should change</Reason>
    </Update>
    ...
  </ImportanceUpdates>
</Analysis>
""")]

        return self.llm.ask({
            "memories": memories_text,
            "character": character
        }, prompt)

    def _apply_consolidation(self, 
                           memory: StructuredMemory,
                           analysis: str,
                           drive_memories: List[MemoryEntry]) -> None:
        """Apply consolidation updates from analysis"""
                   
        # Handle abstractions
        for group in xml.findall('<Group>', analysis):
            timestamps = xml.find('<Timestamps>', group)
            abstract = xml.find('<Abstract>', group)
            
            if abstract:
                # Find all memories that match these timestamps
                candidate_memories = []
                for ts in timestamps.split(","):
                    for mem in drive_memories:
                        if str(mem.timestamp) == ts.strip():
                            candidate_memories.append(mem)
                
                if candidate_memories:
                    # Create new abstract memory
                    new_memory = MemoryEntry(
                        text=abstract,
                        timestamp=datetime.now(),
                        importance=0.8,  # Abstract memories start important
                        confidence=1.0
                    )
                    memory_id = memory.add_entry(new_memory)
                    
                    # Find best similarity match for each candidate
                    for mem in candidate_memories:
                        best_sim = 0.0
                        best_id = None
                        
                        # Compare with all existing abstractions
                        for existing_mem in memory.get_all():
                            if existing_mem.embedding is None:
                                continue
                            sim = np.dot(mem.embedding, existing_mem.embedding) / (
                                np.linalg.norm(mem.embedding) * np.linalg.norm(existing_mem.embedding)
                            )
                            if sim > best_sim:
                                best_sim = sim
                                best_id = existing_mem.memory_id
                        
                        # Link to best match if above threshold
                        if best_sim >= 0.5:  # Lower threshold for finding matches
                            if best_id == memory_id or best_sim > 0.7:  # Prefer new abstract if similarity very high
                                memory.link_memories(memory_id, mem.memory_id)
                            elif best_id is not None:
                                memory.link_memories(best_id, mem.memory_id)
        
        # Handle importance updates
        for update in xml.findall('<Update>', analysis):
            ts = xml.find('<Timestamp>', update)
            adjustment = xml.find('<Adjustment>', update)
            
            for mem in drive_memories:
                if str(mem.timestamp) == ts.strip():
                    if adjustment == "increase":
                        mem.importance = min(1.0, mem.importance * 1.2)
                    else:
                        mem.importance = max(0.1, mem.importance * 0.8)
                        
        # Handle memory linking
        for link in xml.findall('<Link>', analysis):
            if link:  # xml_utils returns None if not found
                times = link.split(",")
                if len(times) == 2:
                    mem1 = mem2 = None
                    for mem in drive_memories:
                        if str(mem.timestamp) == times[0].strip():
                            mem1 = mem
                        elif str(mem.timestamp) == times[1].strip():
                            mem2 = mem
                    if mem1 and mem2:
                        memory.link_memories(mem1.memory_id, mem2.memory_id)

    def _format_memories_for_llm(self, memories: List[MemoryEntry]) -> str:
        """Format memories for LLM prompt"""
        return "\n".join([
            f"[{mem.timestamp}] (importance: {mem.importance:.1f}): {mem.text}"
            for mem in memories
        ])