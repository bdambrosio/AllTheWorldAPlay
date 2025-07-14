# memory/consolidation.py
from __future__ import annotations
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sim.cognitive.knownActor import KnownActorManager
from utils.Messages import UserMessage
from .core import StructuredMemory, MemoryEntry, AbstractMemory, NarrativeSummary
from sim.utility.semantic_clustering import SemanticClusterManager, SemanticCluster
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.cognitive.driveSignal import Drive
    from sim.cognitive.EmotionalStance import EmotionalStance
    from sim.cognitive.driveSignal import SignalCluster
import numpy as np

class MemoryConsolidator:
    def __init__(self, owner, llm, context):
        self.owner = owner
        self.llm = llm
        self.context = context
        # Add semantic clustering manager
        self.semantic_manager = SemanticClusterManager(
            similarity_threshold=0.6,  # Slightly lower threshold for memories
            clustering_eps=0.35
        )
        self.last_semantic_clustering = None

    def set_llm(self, llm):
        self.llm = llm

    def consolidate(self, memory: StructuredMemory):
        """Periodic memory maintenance and optimization with semantic clustering"""
        current_time = self.owner.context.simulation_time
        
        # Original drive-based consolidation
        self._consolidate_by_drives(memory)
        
        # New semantic clustering consolidation
        self._consolidate_by_semantics(memory, current_time)
        
        # Enhanced summaries and cleanup
        self._enhance_summaries(memory)
        self._merge_related_abstractions(memory)
        self._cleanup_abstractions(memory)
        self._cleanup_concrete_memories(memory)

    def _consolidate_by_drives(self, memory: StructuredMemory):
        """Original drive-based consolidation"""
        # Get memories by drive using retrieval system
        drive_memories = {}
        for drive in self.owner.drives:  # Now using Drive objects
            memories = self.owner.memory_retrieval.get_by_drive(
                memory=memory,
                drive=drive,
                threshold=0.1,
                max_results=5  # Limit to most relevant memories
            )
            if memories:
                drive_memories[drive] = memories
        
        # Create abstractions from drive-related memories
        for drive, memories in drive_memories.items():
            if len(memories) >= 4:  # Need at least 2 memories to form pattern
                self._create_abstraction(memory, memories, drive)

    def _consolidate_by_semantics(self, memory: StructuredMemory, current_time: datetime):
        """New semantic clustering consolidation"""
        # Only run semantic clustering periodically to avoid overhead
        if (self.last_semantic_clustering is None or 
            (current_time - self.last_semantic_clustering).total_seconds() > 600):  # Every 15 minutes
            
            # Get recent memories for clustering (last 48 hours)
            recent_cutoff = current_time - timedelta(hours=12)
            recent_memories = [
                mem for mem in memory.concrete_memories 
                if mem.timestamp >= recent_cutoff
            ]
            
            if len(recent_memories) >= 3:  # Need minimum memories for clustering
                self._semantic_cluster_memories(memory, recent_memories, current_time)
                self.last_semantic_clustering = current_time

    def _semantic_cluster_memories(self, memory: StructuredMemory, memories: List[MemoryEntry], current_time: datetime):
        """Perform semantic clustering on memories"""
        # Prepare memories with embeddings
        items_with_embeddings = []
        for mem in memories:
            # Get or create embedding for memory text
            if not hasattr(mem, 'embedding') or mem.embedding is None:
                embedding = self.semantic_manager.get_embedding(mem.to_string())
                mem.embedding = embedding
            else:
                embedding = mem.embedding
            items_with_embeddings.append((mem, embedding))
        
        # Perform clustering
        self.semantic_manager.recluster(
            items_with_embeddings=items_with_embeddings,
            current_time=current_time,
            min_samples=2  # At least 2 memories per cluster
        )
        
        # Create abstractions from semantic clusters
        scored_clusters = self.semantic_manager.score_clusters(current_time)
        
        for cluster, score in scored_clusters:
            if len(cluster.items) >= 2 and score > 1.0:  # Significant clusters only
                self._create_semantic_abstraction(memory, cluster, current_time)

    def _create_semantic_abstraction(self, memory: StructuredMemory, cluster: SemanticCluster, current_time: datetime):
        """Create abstraction from semantic cluster"""
        cluster_memories = cluster.items
        
        # Generate semantic summary using LLM
        memory_texts = [mem.to_string() for mem in cluster_memories]
        
        prompt = [UserMessage(content=f"""Analyze this sequence of related memories and create a concise activity summary.

Related Memories:
{chr(10).join(f"- {text}" for text in memory_texts)}

Create a brief summary 5-15 words) that captures:
1. The common theme or activity
2. The progression or pattern
3. The significance to the character

Focus on what the character was doing and why it forms a coherent pattern.
Respond with just the summary, no reasoning, introductory, or additional text.
End with:
</end>
""")]
        
        summary = self.llm.ask({}, prompt, stops=["</end>"], tag='MemoryConsolidator.semantic_summary')
        if not summary:
            summary = f"Related activity involving {len(cluster_memories)} events"
        else:
            summary = summary.strip()
        
        # Determine if this is cross-drive or single-drive
        memory_drives = set()
        for mem in cluster_memories:
            if hasattr(mem, 'related_drives'):
                memory_drives.update(mem.related_drives)
        
        drive_text = "cross-drive activity" if len(memory_drives) > 1 else memory_drives.pop() if memory_drives else "general activity"
        
        # Create abstraction
        abstraction = AbstractMemory(
            summary=summary,
            start_time=min(mem.timestamp for mem in cluster_memories),
            end_time=max(mem.timestamp for mem in cluster_memories),
            instances=[mem.memory_id for mem in cluster_memories],
            drive=drive_text,
            is_active=False
        )
        
        # Add semantic metadata
        abstraction.cluster_score = cluster.score
        abstraction.cluster_size = len(cluster_memories)
        abstraction.abstraction_type = "semantic"
        
        # Check if this abstraction overlaps with existing ones
        if not self._overlaps_existing_abstraction(memory, abstraction):
            memory.abstract_memories.append(abstraction)
            # Mark concrete memories for potential cleanup
            memory.pending_cleanup.extend([mem.memory_id for mem in cluster_memories])

    def _overlaps_existing_abstraction(self, memory: StructuredMemory, new_abstraction: AbstractMemory) -> bool:
        """Check if new abstraction significantly overlaps with existing ones"""
        new_instances = set(new_abstraction.instances)
        
        for existing in memory.abstract_memories:
            existing_instances = set(existing.instances)
            overlap = len(new_instances & existing_instances)
            
            # If more than 50% overlap, consider it duplicate
            if overlap / len(new_instances) > 0.5:
                return True
                
        return False

    def _enhance_summaries(self, memory: StructuredMemory):
        """Improve summaries of abstract memories using LLM"""
        for abstract in memory.get_recent_abstractions():
            if len(abstract.instances) >= 5:  # Enough instances for better summary
                concrete_instances = [
                    memory.get_by_id(mid).to_string() 
                    for mid in abstract.instances 
                    if memory.get_by_id(mid)
                ]
                
                prompt = [UserMessage(content=f"""
Summarize this sequence of related events into a concise description of the activity:

Events:
{chr(10).join(concrete_instances)}

Respond with just the summary, no additional text.
End your response with:
</end>
""")]
                
                summary = self.llm.ask({}, prompt, stops=["</end>"])
                if summary:
                    abstract.summary = summary.strip()

    def _merge_related_abstractions(self, memory: StructuredMemory):
        """Merge abstract memories that represent the same ongoing activity"""
        recent = memory.get_recent_abstractions()
        recent_copy = list(recent)  # Iterate over a copy to safely modify the original
        
        for i, abs1 in enumerate(recent_copy):
            if not abs1.is_active and abs1 in memory.abstract_memories:  # Only merge completed abstractions that still exist
                for abs2 in recent_copy[i+1:]:
                    if not abs2.is_active and abs2 in memory.abstract_memories:
                        if self._should_merge(abs1, abs2):
                            self._merge_abstractions(memory, abs1, abs2)
                            break  # Break inner loop since abs1 has been merged/removed

    def _should_merge(self, abs1: AbstractMemory, abs2: AbstractMemory) -> bool:
        """Check if two abstract memories should be merged"""
        # Time proximity
        time_gap = abs2.start_time - abs1.end_time
        if time_gap > timedelta(hours=1):
            return False
            
        # Content similarity
        if abs1.embedding is not None and abs2.embedding is not None:
            similarity = np.dot(abs1.embedding, abs2.embedding) / (
                np.linalg.norm(abs1.embedding) * np.linalg.norm(abs2.embedding)
            )
            return similarity > 0.7
            
        return False

    def _merge_abstractions(self, memory: StructuredMemory, abs1: AbstractMemory, abs2: AbstractMemory):
        """Merge two abstract memories"""
        merged = AbstractMemory(
            summary=self._merge_summaries(abs1, abs2),
            start_time=min(abs1.start_time, abs2.start_time),
            end_time=max(abs1.end_time, abs2.end_time),
            instances=abs1.instances + abs2.instances,
            drive=abs1.drive,  # Keep drive from first abstraction
            is_active=False
        )
        
        # Remove old abstractions and add merged one - check if they're still in the list first
        if abs1 in memory.abstract_memories:
            memory.abstract_memories.remove(abs1)
        if abs2 in memory.abstract_memories:
            memory.abstract_memories.remove(abs2)
        memory.abstract_memories.append(merged)

    def _merge_summaries(self, abs1: AbstractMemory, abs2: AbstractMemory) -> str:
        """Merge summaries of two abstractions"""
        return f"{abs1.summary}; {abs2.summary}"

    def _cleanup_abstractions(self, memory: StructuredMemory):
        """Remove or archive old/unimportant abstract memories"""
        cutoff = self.owner.context.simulation_time - timedelta(days=7)  # Keep week of abstractions
        memory.abstract_memories = [
            mem for mem in memory.abstract_memories
            if mem.is_active or (mem.end_time is not None and mem.end_time > cutoff)
        ]

    def _cleanup_concrete_memories(self, memory: StructuredMemory):
        """Remove concrete memories that have been abstracted"""
        if not memory.pending_cleanup:
            return
            
        # Keep very recent memories regardless
        recent_cutoff = self.owner.context.simulation_time - timedelta(hours=1)
        
        # Remove concrete memories that are:
        # 1. In pending_cleanup
        # 2. Not from the last hour
        # 3. Not highly important (keep significant memories)
        memory.concrete_memories = [
            mem for mem in memory.concrete_memories
            if (mem.memory_id not in memory.pending_cleanup or
                mem.timestamp > recent_cutoff or
                mem.importance > 0.8)
        ]
        
        # Clear pending cleanup
        memory.pending_cleanup = []

    # Future pattern learning stubs
    def _detect_patterns(self, memory: StructuredMemory):
        """Detect behavioral/event patterns across abstract memories"""
        pass

    def update_cognitive_model(self, memory: StructuredMemory, narrative: NarrativeSummary, knownActorManager: KnownActorManager,
                        current_time: datetime, character_desc: str, cycle: int = 0, relationsOnly: bool = True) -> None:
        """Update narrative summary based on recent memories and abstractions"""
        
        if not narrative.needs_update(current_time):
            return
        
        narrative.last_update = current_time
        # Update ongoing activities
        active_abstract = memory.get_active_abstraction()
        prompt = [UserMessage(content=f"""Given a character's current state, summarize their ongoing activities and immediate goals.

Character Description:
{character_desc}

Current Drives:
{chr(10).join(f"- {drive}" for drive in narrative.active_drives)}

Active Activity:
{active_abstract.summary if active_abstract else "No current activity"}

Recent Abstractions:
{chr(10).join(f"- {abs.summary}" for abs in memory.get_recent_abstractions(4))}

Recent Concrete Memories:
        {chr(10).join(f"- {mem.to_string()}" for mem in memory.get_recent(20))}

Create a concise summary (about 100 words) that describes:
1. What the character is currently doing
2. Their immediate goals and intentions
3. How this relates to their drives
4. Any ongoing social interactions

Focus on active pursuits and immediate intentions.
Respond with just the activity summary, no additional text.
End with:
</End>
""")]

        if not relationsOnly:
            narrative.last_update = current_time
            new_activities = self.llm.ask({}, prompt, stops=["</End>"], tag='MemoryConsolidator.ongoing_activities')
            if new_activities:
                narrative.ongoing_activities = new_activities.strip()
        
        # Update recent events
        recent_window = current_time - timedelta(hours=2)  # Last 4 hours
        recent_abstracts = [abs for abs in memory.get_recent_abstractions(20)
                           if abs.start_time >= recent_window]
        recent_concretes = [mem for mem in memory.get_recent(40)
                           if mem.timestamp >= recent_window]
        
        prompt = [UserMessage(content=f"""Create a detailed narrative of recent events (last 2 hours) for this character.

Character Description:
{character_desc}

Current Activity:
{active_abstract.summary if active_abstract else "No current activity"}

Recent Event Sequence:
{chr(10).join(f"- {mem.to_string()}" for mem in sorted(recent_concretes, key=lambda x: x.timestamp))}

Recent Activity Patterns:
{chr(10).join(f"- {abs.summary}" for abs in recent_abstracts)}

Create a flowing narrative (about 150 words) that:
1. Describes events in chronological order
2. Maintains causal relationships between events
3. Includes character's reactions and decisions
4. Shows how events connect to current situation

Write in past tense, focus on specific details and immediate consequences.
Respond with just the event narrative, no additional text.
End with:
</end>
""")]

        if not relationsOnly:
            new_events = self.llm.ask({}, prompt, stops=["</end>"], tag='MemoryConsolidator.recent_events', max_tokens=240)
            if new_events:
                narrative.recent_events = new_events.strip()
        
        # Update key relationships
        valid_chars = [a.name for a in self.context.actors + self.context.npcs
                      if a.name != self.owner.name]
        
        # Extract character names from memory text
        recent_window = current_time - timedelta(hours=2)  # Look back 2 hour
        recent_memories = [mem for mem in memory.get_recent_concrete(40) 
                          if mem.timestamp >= recent_window]
        all_texts = [mem.to_string() for mem in recent_memories]
        
        # Find character names in recent texts (simple approach)
        # Look for capitalized words that aren't at start of sentence
        potential_chars = set()
        for text in all_texts:
            words = text.split()
            for i, word in enumerate(words):
                if (word[0].isupper() and  # Capitalized
                    len(word) > 1 and      # Not single letter
                    i > 0 and word.lower() not in {'i', 'me', 'my', 'mine','you', 'your', 'yours', 'he', 'she', 'it', 'they', 'them', 'their', 'theirs'}):  # Not pronouns
                    if word not in potential_chars:
                        potential_chars.add(word)
        
        # Only use valid characters from potential and existing chars
        all_char_names = set(valid_chars) & (set(knownActorManager.names()) | potential_chars)
        knownActorManager.update_all_relationships(all_texts, all_char_names)
        
        

    def _create_abstraction(self, memory: StructuredMemory, memories: List[MemoryEntry], drive: Drive) -> None:
        """Create a new abstract memory from related concrete memories"""
        # Get average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)
        
        # Create summary using first memory as template
        summary = f"Activity related to {drive.text}: {memories[0].to_string()}"
        
        # Create new abstraction
        abstraction = AbstractMemory(
            summary=summary,
            start_time=min(m.timestamp for m in memories),
            end_time=max(m.timestamp for m in memories),
            instances=[m.memory_id for m in memories],
            drive=drive.text,
            is_active=False
        )
        
        # Mark as drive-based abstraction
        abstraction.abstraction_type = "drive-based"
        
        # Add to memory
        memory.abstract_memories.append(abstraction)

    class Pattern:
        """Future: Represent learned patterns"""
        def __init__(self):
            self.description: str
            self.supporting_abstractions: List[int]  # Abstract memory IDs
            self.confidence: float
            self.last_validated: datetime