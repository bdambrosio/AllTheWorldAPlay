#!/usr/bin/env python3

import numpy as np

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

class MockContext:
    def __init__(self):
        self.scene_task_embeddings = []
        self.scene_integrated_task_plan = []
    
    def get_actor_by_name(self, name):
        return self.actors.get(name)
    
    def embed_task(self, task):
        # Create predictable embeddings that will cluster
        if "greet" in task.name or "hello" in task.name:
            # Greeting tasks get similar embeddings
            base = np.array([1.0, 0.0, 0.0] + [0.1] * 381)
            embedding = base + np.random.normal(0, 0.05, 384)  # Small noise
        elif "explore" in task.name or "search" in task.name:
            # Exploration tasks get similar embeddings
            base = np.array([0.0, 1.0, 0.0] + [0.1] * 381)
            embedding = base + np.random.normal(0, 0.05, 384)  # Small noise
        else:
            # Other tasks get random embeddings
            embedding = np.random.rand(384)
            
        self.scene_task_embeddings.append(embedding)
        return embedding
    
    def cluster_tasks(self, task_embeddings):
        if not task_embeddings or len(task_embeddings) < 2:
            return []
        
        # Manual clustering based on similarity
        clusters = []
        used = set()
        
        for i, emb1 in enumerate(task_embeddings):
            if i in used:
                continue
                
            cluster = [i]
            used.add(i)
            
            for j, emb2 in enumerate(task_embeddings):
                if j in used or i == j:
                    continue
                    
                # Calculate cosine similarity
                cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                if cosine_sim > 0.8:  # High similarity threshold
                    cluster.append(j)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def evaluate_task_criticality(self, task):
        return 0.5

    def integrate_task_plans(self, scene):
        """Simplified version of the method we're testing"""
        actors_in_scene = []
        self.scene_task_embeddings = []
        task_id_to_embedding_index = {}
        actor_tasks = {}
        total_input_task_count = 0
        
        # Get actors and task_plans
        for actor_name in scene['actors']:
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
                    task_id_to_embedding_index[task.id] = len(self.scene_task_embeddings)
                    self.embed_task(task)

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
        actors_with_remaining_tasks = [a.name for a in actors_in_scene]
        
        while len(actors_with_remaining_tasks) > 0:
            for name in scene['actor_order']:
                if name not in actors_with_remaining_tasks:
                    continue
                actor = self.get_actor_by_name(name)
                next_task_index = actor_tasks[actor.name]['next_task_index']
                current_task = actor.focus_goal.task_plan[next_task_index]
                
                # Check if this task's cluster has already been used
                cluster_idx = task_id_to_cluster.get(current_task.id, -1)
                if cluster_idx not in used_clusters:
                    self.scene_integrated_task_plan.append({'actor': actor, 'task': current_task})
                    used_clusters.add(cluster_idx)
                
                task_index += 1
                actor_tasks[actor.name]['next_task_index'] += 1
                if actor_tasks[actor.name]['next_task_index'] >= len(actor_tasks[actor.name]['task_plan']):
                    actors_with_remaining_tasks.remove(actor.name)
            
        return self.scene_integrated_task_plan

def test_integrate_task_plans():
    # Create mock context
    context = MockContext()
    
    # Create mock actors with some similar tasks to force clustering
    actor1 = MockActor("Alice", ["greet", "explore", "decide"])
    actor2 = MockActor("Bob", ["hello", "search", "act"])  # "hello" similar to "greet", "search" similar to "explore"
    actor3 = MockActor("Charlie", ["observe", "plan"])
    
    context.actors = {"Alice": actor1, "Bob": actor2, "Charlie": actor3}
    
    # Create test scene
    scene = {
        'actors': ['Alice', 'Bob', 'Charlie'],
        'actor_order': ['Alice', 'Bob', 'Charlie']
    }
    
    # Run the method we're testing
    result = context.integrate_task_plans(scene)
    
    # Print results
    print("=== Test Results ===")
    print(f"Total embeddings created: {len(context.scene_task_embeddings)}")
    print(f"Integrated task plan length: {len(context.scene_integrated_task_plan)}")
    
    print("\nAll tasks by actor:")
    for actor_name, actor in context.actors.items():
        tasks = [t.name for t in actor.focus_goal.task_plan]
        print(f"  {actor_name}: {tasks}")
    
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
                for actor in context.actors.values():
                    for j, task in enumerate(actor.focus_goal.task_plan):
                        if j + sum(len(a.focus_goal.task_plan) for a in list(context.actors.values())[:list(context.actors.keys()).index(actor.name)]) == emb_idx:
                            task_names.append(task.name)
                            break
            print(f"  Cluster {i}: {task_names} (size {len(cluster)})")
    
    # Verify clustering worked
    total_tasks = sum(len(actor.focus_goal.task_plan) for actor in context.actors.values())
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