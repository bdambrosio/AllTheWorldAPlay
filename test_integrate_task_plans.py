#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import numpy as np
from unittest.mock import Mock, MagicMock
from src.sim.context import Context

class MockTask:
    _id_counter = 0
    
    def __init__(self, name, description="test task", reason="test reason"):
        MockTask._id_counter += 1
        self.id = f"task_{MockTask._id_counter}"
        self.name = name
        self.description = description
        self.reason = reason
        
    def to_string(self):
        return f"{self.name}: {self.description}. because {self.reason}"

class MockGoal:
    def __init__(self, task_plan):
        self.task_plan = task_plan

class MockActor:
    def __init__(self, name, tasks):
        self.name = name
        self.__class__.__name__ = 'NarrativeCharacter'
        self.focus_goal = MockGoal([MockTask(f"{name}_{t}") for t in tasks])

def test_integrate_task_plans():
    # Create mock context
    context = Mock(spec=Context)
    context.scene_task_embeddings = []
    context.scene_integrated_task_plan = []
    
    # Create mock actors with tasks
    actor1 = MockActor("Alice", ["greet", "explore", "decide"])
    actor2 = MockActor("Bob", ["listen", "respond", "act"])  
    actor3 = MockActor("Charlie", ["observe", "plan"])
    
    actors = {"Alice": actor1, "Bob": actor2, "Charlie": actor3}
    
    # Mock the get_actor_by_name method
    context.get_actor_by_name = lambda name: actors.get(name)
    
    # Mock embed_task to return random embeddings
    def mock_embed_task(task):
        embedding = np.random.rand(384)  # Random 384-dim vector
        context.scene_task_embeddings.append(embedding)
        return embedding
    context.embed_task = mock_embed_task
    
    # Mock cluster_tasks to return some clusters
    def mock_cluster_tasks(embeddings):
        if len(embeddings) < 2:
            return []
        # Create some mock clusters - group similar tasks
        clusters = [
            [0, 3],  # Alice_greet + Bob_listen (cluster 0)
            [1],     # Alice_explore (outlier)
            [2, 5],  # Alice_decide + Bob_act (cluster 1) 
            [4],     # Bob_respond (outlier)
            [6],     # Charlie_observe (outlier)
            [7]      # Charlie_plan (outlier)
        ]
        return clusters
    context.cluster_tasks = mock_cluster_tasks
    
    # Mock evaluate_task_criticality
    context.evaluate_task_criticality = lambda task: 0.5
    
    # Create test scene
    scene = {
        'actors': ['Alice', 'Bob', 'Charlie'],
        'actor_order': ['Alice', 'Bob', 'Charlie']
    }
    
    # Run the method we're testing
    Context.integrate_task_plans(context, scene)
    
    # Print results
    print("=== Test Results ===")
    print(f"Total embeddings created: {len(context.scene_task_embeddings)}")
    print(f"Integrated task plan length: {len(context.scene_integrated_task_plan)}")
    
    print("\nIntegrated task plan:")
    for i, item in enumerate(context.scene_integrated_task_plan):
        actor_name = item['actor'].name
        task_name = item['task'].name
        print(f"  {i+1}. {actor_name}: {task_name}")
    
    # Verify clustering worked - should have fewer integrated tasks than total tasks
    total_tasks = sum(len(actor.focus_goal.task_plan) for actor in actors.values())
    integrated_tasks = len(context.scene_integrated_task_plan)
    
    print(f"\nClustering effectiveness:")
    print(f"  Total input tasks: {total_tasks}")
    print(f"  Integrated tasks: {integrated_tasks}")
    print(f"  Tasks eliminated by clustering: {total_tasks - integrated_tasks}")
    
    # Verify no duplicate clusters
    task_ids_seen = set()
    for item in context.scene_integrated_task_plan:
        task_id = item['task'].id
        if task_id in task_ids_seen:
            print(f"ERROR: Duplicate task ID {task_id}")
        task_ids_seen.add(task_id)
    
    print(f"  Unique tasks in plan: {len(task_ids_seen)}")
    assert len(task_ids_seen) == integrated_tasks, "Duplicate tasks found!"
    
    print("\n✓ Test passed!")

if __name__ == "__main__":
    test_integrate_task_plans() 