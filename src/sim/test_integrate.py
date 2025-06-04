#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.llm_api as LLM_API
from utils.Messages import UserMessage
llm = LLM_API.LLM(server_name='openai')
import json
from datetime import datetime
from character_dataclasses import Task, Goal, datetime_handler

import numpy as np

class MockTask:
    _id_counter = 0
    
    def __init__(self, name, description="test task", reason="test reason", termination="test termination"):
        MockTask._id_counter += 1
        self.id = f"task_{MockTask._id_counter}"
        self.name = name
        self.description = description
        self.reason = reason
        self.termination = termination
        
    def to_string(self):
        return f"{self.name}: {self.description}. because {self.reason}"

class MockGoal:
    def __init__(self, task_plan):
        self.task_plan = task_plan

class MockActor:
    def __init__(self, name, tasks):
        self.name = name
        self.__class__.__name__ = 'NarrativeCharacter'
        self.focus_goal = MockGoal(tasks)

class MockContext:
    def __init__(self):
        self.scene_task_embeddings = []
        self.scene_integrated_task_plan = []
        self.llm = LLM_API.LLM(server_name='openai')
        self.name = 'Context'
        self.act_central_narrative = 'Reach final agreement on how to proceed'
        self.central_narrative = 'Battle against the adversary to reach competitive advantage and dominate the market'
        self.scene_history = 'Alice and Bob have been arguing about how to proceed. Charlie is observing and planning.'
        self.transcript = 'Alice: I think we should go to the apple tree. Bob: I think we should go to the apple tree. Charlie: I think we should go to the apple tree.'
        self.scene_integrated_task_plan = []
        self.actors_by_name = {}
        self.actors = []
    
    def get_actor_by_name(self, name):
        return self.actors_by_name.get(name)
    
    def embed_task(self, task):
        """Embed a task"""
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = _embedding_model.encode(task.name+': '+task.description+'. because '+task.reason+' to achieve '+task.termination)
        self.scene_task_embeddings.append(embedding)
        return embedding
    
    def cluster_tasks(self, task_embeddings):
        """Cluster tasks"""
        from sklearn.cluster import DBSCAN
        
        if not task_embeddings or len(task_embeddings) < 2:
            self.scene_task_clusters = []
            return self.scene_task_clusters
        
        clustering = DBSCAN(eps=0.20, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(task_embeddings)
        
        # Group by cluster labels
        clusters = []
        cluster_dict = {}
        for i, label in enumerate(labels):
            if label == -1:  # Outlier
                clusters.append([i])
            else:
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(i)
        
        clusters.extend(cluster_dict.values())
        self.scene_task_clusters = clusters
        return self.scene_task_clusters
    
    def integrate_task_plans(self, scene):
        """Integrate the task plans of all characters in the scene"""
        actors_in_scene = []
        self.scene_task_embeddings = []
        task_id_to_embedding_index = {}
        total_input_task_count = 0
        total_actor_beats = len(scene['action_order'])


        actor_tasks = {}        # get actors and task_plans
        for actor_name in scene['characters']:
            if actor_name == 'Context':
                continue
            actor = self.get_actor_by_name(actor_name)
            if actor.__class__.__name__ == 'NarrativeCharacter':
                actors_in_scene.append(actor)
                actor_tasks[actor.name] = {}
                actor_tasks[actor.name]['task_plan'] = actor.focus_goal.task_plan if actor.focus_goal and actor.focus_goal.task_plan else []
                total_input_task_count += len(actor_tasks[actor.name]['task_plan'])
                actor_tasks[actor.name]['next_task_index'] = 0
                for n, task in enumerate(actor_tasks[actor.name]['task_plan']):
                    #actor_tasks[actor]['task_plan'][task.id]['criticality'] = self.evaluate_task_criticality(task)
                    task_id_to_embedding_index[task.id] = len(self.scene_task_embeddings)
                    self.embed_task(task)

        prune_level = 0
        if total_input_task_count > 1.5*total_actor_beats:
            prune_level = 1
        if total_input_task_count > 2*total_actor_beats:
            prune_level = 2
        if total_input_task_count > 3*total_actor_beats:
            prune_level = 3
        scene_task_clusters = self.cluster_tasks(self.scene_task_embeddings)

        # Create mapping from task id to cluster index
        task_id_to_cluster = {}
        for cluster_idx, task_indices in enumerate(scene_task_clusters):
            for embedding_idx in task_indices:
                # Find task id that corresponds to this embedding index
                for task_id, emb_idx in task_id_to_embedding_index.items():
                    if emb_idx == embedding_idx:
                        task_id_to_cluster[task_id] = cluster_idx
                        break
        
        used_clusters = set()
        task_index = 0
        self.scene_integrated_task_plan = []
        actors_with_remaining_tasks = actors_in_scene
        while len(actors_with_remaining_tasks) > 0:
            for name in scene['action_order']:
                actor = self.get_actor_by_name(name)
                if actor not in actors_with_remaining_tasks:
                    continue
                next_task_index = actor_tasks[actor.name]['next_task_index']
                current_task = actor.focus_goal.task_plan[next_task_index]
                self.evaluate_commitment_significance(actor, current_task, scene)
                is_critical = next_task_index == len(actor.focus_goal.task_plan) - 1
                
                # Check if this task's cluster has already been used
                cluster_idx = task_id_to_cluster.get(current_task.id, -1)
                # note alternative is to merge tasks in a cluster before reaching here if prune_level > 0 and tasks are not critical
                if cluster_idx not in used_clusters or prune_level < 1 or is_critical:
                    self.scene_integrated_task_plan.append({'actor': actor, 'task': current_task})
                    used_clusters.add(cluster_idx)
                
                task_index += 1
                actor_tasks[actor.name]['next_task_index'] += 1
                if actor_tasks[actor.name]['next_task_index'] >= len(actor_tasks[actor.name]['task_plan']):
                    actors_with_remaining_tasks.remove(actor)
            
        return self.scene_integrated_task_plan

    def evaluate_commitment_significance(self, other_character, commitment_task, current_scene):
        prompt = [UserMessage(content="""Evaluate the significance of a tentative verbal commitment to the current dramatic context
    and determine if it is a significant commitment that should be added to the task plan.
The apparentcommitment is inferred from a verbal statement to act now made by self, {{$name}}, to other, {{$target_name}}.
The apparent commitment is tentative, and may be redundant in the scene you are in.
The apparent commitment may be hypothetical, and may not be a commitment to act now.
The apparent commitment may be redundant in the scene you are in, or simply a restatement of already established plans.
The apparent commitment may simply be 'social noise' - a statement of intent or opinion, not a commitment to act now.
Alternatively, the apparent commitment may be unique andsignificant to the scene you are in or the central dramatic question.

## Dramatic Context
<central_narrative>
{{$central_narrative}}
</central_narrative>

<act_specific_narrative>
{{$act_specific_narrative}}
</act_specific_narrative>

<current_scene>
{{$current_scene}}
</current_scene>

<scene_integrated_task_plan>
{{$scene_integrated_task_plan}}
</scene_integrated_task_plan>

<scene_history> 
{{$scene_history}}
</scene_history>

<transcript>    
{{$transcript}}
</transcript>
        
## Apparent Commitment
<commitment>
{{$commitment_task}}
</commitment>

Your task is to evaluate the significance of the apparent commitment.
Determine if it is 
    social NOISE that should be ignored.
    a RELEVANT task that can be optionally included if it is not redundant and not distracting in the scene you are in, or 
    a SIGNIFICANT commitment that should be added to the task plan,  
Respond with a single word: "SIGNIFICANT" or "RELEVANT" or "NOISE".
Do not include any other text in your response.

End your response with </end>
"""
        )]
        significance = 'NOISE'
        response = self.llm.ask({"name": self.name, 
                               "commitment_task": commitment_task.to_string(),
                               "central_narrative": self.context.central_narrative,
                               "act_central_narrative": self.context.act_central_narrative,
                               "current_scene": json.dumps(current_scene, indent=2, default=datetime_handler) if current_scene else '',
                               "scene_integrated_task_plan": self.context.scene_integrated_task_plan if self.context.scene_integrated_task_plan else self.goal.task_plan,
                               "scene_history": self.context.scene_history,
                               "transcript": self.actor_models.get_actor_model(other_character.name, create_if_missing=True).dialog.get_transcript(6)},
                                prompt, max_tokens=100, stops=['</end>'], tag='evaluate_commitment_significance')
        if response:
            response = response.lower()
            if 'significant' in response:
                return 'SIGNIFICANT'
            elif 'relevant' in response:
                return 'RELEVANT'
            else:
                return 'NOISE'
        return 'NOISE'


def test_integrate_task_plans():
    # Create mock context
    context = MockContext()
    
    # Create mock actors with some similar tasks to force clustering
    actor1 = MockActor("Alice", [MockTask("greet", "greet", "greet", "greet"), 
                                 MockTask("explore", "explore apple tree", "explore", "find apple"), 
                                 MockTask("decide", "decide", "decide", "decide")])
    actor2 = MockActor("Bob", [MockTask("hello", "hello", "hello", "hello"), 
                               MockTask("explore", "search around apple tree", "explore", "find apple"), 
                               MockTask("act", "act", "act", "act")])  # "hello" similar to "greet", "search" similar to "explore"
    actor3 = MockActor("Charlie", [MockTask("observe", "observe", "observe", "observe"), 
                                   MockTask("plan", "plan", "plan", "plan")])
    
    context.actors_by_name = {"Alice": actor1, "Bob": actor2, "Charlie": actor3}
    context.actors = [actor1, actor2, actor3]
    # Create test scene
    scene = {
        'characters': ['Alice', 'Bob', 'Charlie'],
        'action_order': ['Alice', 'Bob', 'Charlie']
    }
    
    # Run the method we're testing
    result = context.integrate_task_plans(scene)
    
    # Print results
    print("=== Test Results ===")
    print(f"Total embeddings created: {len(context.scene_task_embeddings)}")
    print(f"Integrated task plan length: {len(context.scene_integrated_task_plan)}")
    
    print("\nAll tasks by actor:")
    for actor in context.actors:
        tasks = [t.name for t in actor.focus_goal.task_plan]
        print(f"  {actor.name}: {tasks}")
    
    print("\nIntegrated task plan:")
    for i, item in enumerate(context.scene_integrated_task_plan):
        actor_name = item['actor'].name
        task_name = item['task'].name
        task_id = item['task'].id
        print(f"  {i+1}. {actor_name}: {task_name} (id: {task_id})")
    
    # Show which tasks were clustered
    print("\nClustering details:")
    embeddings_used = len(context.scene_task_embeddings)
    clusters = context.cluster_tasks(context.scene_task_embeddings)
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            task_names = []
            for emb_idx in cluster:
                # Find task name for this embedding index
                for actor in context.actors:
                    for j, task in enumerate(actor.focus_goal.task_plan):
                        if j + sum(len(a.focus_goal.task_plan) for a in list(context.actors)[:list([a.name for a in context.actors]).index(actor.name)]) == emb_idx:
                            task_names.append(task.name)
                            break
            print(f"  Cluster {i}: {task_names} (size {len(cluster)})")
    
    # Verify clustering worked
    total_tasks = sum(len(actor.focus_goal.task_plan) for actor in context.actors)
    integrated_tasks = len(context.scene_integrated_task_plan)
    
    print(f"\nClustering effectiveness:")
    print(f"  Total input tasks: {total_tasks}")
    print(f"  Integrated tasks: {integrated_tasks}")
    print(f"  Tasks eliminated by clustering: {total_tasks - integrated_tasks}")
    
    # Verify no duplicate task IDs
    task_ids_seen = set()
    for item in context.scene_integrated_task_plan:
        task_id = item['task'].id
        if task_id in task_ids_seen:
            print(f"ERROR: Duplicate task ID {task_id}")
        task_ids_seen.add(task_id)
    
    print(f"  Unique tasks in plan: {len(task_ids_seen)}")
    assert len(task_ids_seen) == integrated_tasks, "Duplicate tasks found!"
    
    if integrated_tasks < total_tasks:
        print("\n✓ Test passed - clustering successfully eliminated duplicate tasks!")
    else:
        print("\n✓ Test passed - no clusters found, all tasks preserved!")

if __name__ == "__main__":
    test_integrate_task_plans()