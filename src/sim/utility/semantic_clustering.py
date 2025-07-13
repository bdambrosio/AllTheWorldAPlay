# sim/utility/semantic_clustering.py
"""
Shared semantic clustering utilities for drive signals and memory consolidation.
Extracted and adapted from driveSignal.py to enable reuse.
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Any
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sentence_transformers import SentenceTransformer

try:
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
except Exception as e:
    print(f"Warning: Could not load embedding model locally: {e}")
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


class SemanticCluster:
    """Generic semantic cluster for any embeddable content"""
    
    def __init__(self, centroid: np.ndarray, items: List[Any], text: str = ""):
        self.centroid = centroid
        self.items = items
        self.text = text
        self.score = 0.0
        
    def add_item(self, item: Any, embedding: np.ndarray) -> None:
        """Add item to cluster and update centroid"""
        self.items.append(item)
        # Update centroid as running average
        n = len(self.items)
        self.centroid = ((n-1) * self.centroid + embedding) / n
        
    def get_latest_timestamp(self) -> datetime:
        """Get most recent timestamp from items (assumes items have timestamp attr)"""
        if not self.items:
            return datetime.min
        return max(item.timestamp for item in self.items if hasattr(item, 'timestamp'))
        
    def get_importance(self, current_time: datetime, min_age: float = 0, age_range: float = 24) -> float:
        """Calculate cluster importance based on items"""
        if not self.items:
            return 0.0
            
        # Average importance of items (assumes items have importance attr)
        importances = [getattr(item, 'importance', 0.5) for item in self.items]
        avg_importance = sum(importances) / len(importances)
        
        # Apply age decay
        latest_time = self.get_latest_timestamp()
        if latest_time != datetime.min:
            age_hours = (current_time - latest_time).total_seconds() / 3600
            if age_range > 0:
                age_factor = max(0.1, 1.0 - (age_hours - min_age) / age_range)
            else:
                age_factor = 1.0
            return avg_importance * age_factor
        
        return avg_importance


class SemanticClusterManager:
    """Generic semantic clustering manager"""
    
    def __init__(self, similarity_threshold: float = 0.7, clustering_eps: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self.clustering_eps = clustering_eps
        self.clusters: List[SemanticCluster] = []
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        return _embedding_model.encode(text)
        
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def find_closest_cluster(self, embedding: np.ndarray) -> Tuple[int, float]:
        """Find closest cluster to given embedding"""
        if not self.clusters:
            return -1, 0.0
            
        best_idx = -1
        best_similarity = 0.0
        
        for i, cluster in enumerate(self.clusters):
            similarity = self.cosine_similarity(embedding, cluster.centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
                
        return best_idx, best_similarity
        
    def add_to_cluster(self, item: Any, embedding: np.ndarray, text: str = "") -> bool:
        """Add item to existing cluster if similar enough, return True if added"""
        closest_idx, similarity = self.find_closest_cluster(embedding)
        
        if similarity > self.similarity_threshold and closest_idx >= 0:
            self.clusters[closest_idx].add_item(item, embedding)
            return True
        else:
            # Create new cluster
            new_cluster = SemanticCluster(
                centroid=embedding.copy(),
                items=[item],
                text=text
            )
            self.clusters.append(new_cluster)
            return False
            
    def recluster(self, items_with_embeddings: List[Tuple[Any, np.ndarray]], 
                  current_time: datetime, eps: float = -1, min_samples: int = 2) -> None:
        """Rebuild clusters using DBSCAN"""
        if not items_with_embeddings:
            self.clusters = []
            return
            
        if eps == -1:
            eps = self.clustering_eps
            
        items, embeddings = zip(*items_with_embeddings)
        embeddings = np.array(embeddings)
        
        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Group items by cluster label
        new_clusters = defaultdict(list)
        outlier_items = []
        
        for item, embedding, label in zip(items, embeddings, labels):
            if label != -1:
                new_clusters[label].append((item, embedding))
            else:
                outlier_items.append((item, embedding))
                
        # Create new cluster objects
        self.clusters = []
        for cluster_items in new_clusters.values():
            if cluster_items:
                items_only, embeddings_only = zip(*cluster_items)
                # Calculate centroid
                centroid = np.mean(embeddings_only, axis=0)
                
                cluster = SemanticCluster(
                    centroid=centroid,
                    items=list(items_only),
                    text=f"Cluster of {len(items_only)} items"
                )
                self.clusters.append(cluster)
                
        # Handle outliers - preserve important recent ones
        for item, embedding in outlier_items:
            if hasattr(item, 'timestamp') and hasattr(item, 'importance'):
                age_minutes = (current_time - item.timestamp).total_seconds() / 60
                recency = max(0.1, 15 / max(15, age_minutes))
                importance = getattr(item, 'importance', 0.5)
                
                if (recency * importance) ** 0.33 >= 0.5:
                    outlier_cluster = SemanticCluster(
                        centroid=embedding.copy(),
                        items=[item],
                        text=f"Outlier: {getattr(item, 'text', str(item))[:50]}"
                    )
                    self.clusters.append(outlier_cluster)
                    
        # Adjust eps for next clustering
        num_clusters = len(self.clusters)
        num_items = len(items)
        if num_items > 24 and num_clusters > 0:
            eps_adjustment = 0.01 * (num_items ** 0.5 - num_clusters)
            if 0 < self.clustering_eps - eps_adjustment < 0.5:
                self.clustering_eps = self.clustering_eps - eps_adjustment
                
    def score_clusters(self, current_time: datetime) -> List[Tuple[SemanticCluster, float]]:
        """Score clusters by importance, recency, and size"""
        if not self.clusters:
            return []
            
        # Calculate age ranges
        max_age = max((current_time - c.get_latest_timestamp()).total_seconds() / 3600 
                     for c in self.clusters if c.get_latest_timestamp() != datetime.min) + 1
        min_age = min((current_time - c.get_latest_timestamp()).total_seconds() / 3600 
                     for c in self.clusters if c.get_latest_timestamp() != datetime.min)
        age_range = max(max_age - min_age, 0.25)
        
        max_cluster_size = max(len(c.items) for c in self.clusters)
        
        scored_clusters = []
        for cluster in self.clusters:
            try:
                size_ratio = 100 * len(cluster.items) / max_cluster_size
                
                latest_time = cluster.get_latest_timestamp()
                if latest_time != datetime.min:
                    cluster_age = (current_time - latest_time).total_seconds() / 3600
                    if cluster_age < min_age or age_range < 0.25:
                        recency = 100
                    else:
                        recency = 100 - 80 * (cluster_age - min_age) / age_range
                else:
                    recency = 0
                    
                importance = 100 * cluster.get_importance(current_time, min_age, age_range)
                
                score = (importance * size_ratio * recency) ** 0.25
                cluster.score = score
                scored_clusters.append((cluster, score))
                
            except Exception as e:
                print(f"Error scoring cluster: {e}")
                
        return sorted(scored_clusters, key=lambda x: x[1], reverse=True) 