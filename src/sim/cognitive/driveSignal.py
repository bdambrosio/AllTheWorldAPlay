from __future__ import annotations
from datetime import datetime
import math
import numpy as np
from typing import ClassVar, List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import traceback
from utils import hash_utils
import utils.xml_utils as xml
from utils.llm_api import LLM
import sim.context
from utils.Messages import SystemMessage
import utils.llm_api as llm_api
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from weakref import WeakValueDictionary

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking
    from sim.cognitive.EmotionalStance import EmotionalStance

_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@dataclass(frozen=True)
class Drive:
    """Represents a character drive with semantic embedding"""
    _id_counter: ClassVar[int] = 0
    _instances: ClassVar[WeakValueDictionary] = WeakValueDictionary()
    text: str
    embedding: Optional[np.ndarray] = None
    id: str = field(init=False)
    
    def __post_init__(self):
        Drive._id_counter += 1
        id_val = f"d{Drive._id_counter}"
        object.__setattr__(self, 'id', id_val)
        Drive._instances[id_val] = self
        if self.embedding is None:
            object.__setattr__(self, 'embedding', _embedding_model.encode(self.text))
    
    def __hash__(self):
        return hash(self.text)
    
    def __eq__(self, other):
        if not isinstance(other, Drive):
            return False
        return self.text == other.text

    @classmethod
    def get_by_id(cls, id: str):
        return cls._instances.get(id)

@dataclass
class DriveSignal:
    """Represents a detected issue or opportunity related to a drive"""
    text: str                # The text that triggered this signal
    drives: List[Drive]      # Reference to the related drives
    is_opportunity: bool    # True if opportunity, False if issue
    importance: float       # 0-1 scale of importance
    urgency: float         # 0-1 scale of urgency
    timestamp: datetime    # When this was last detected
    embedding: np.ndarray  # Vector embedding of the text

    def to_string(self):
        return f'{self.text}: {"opportunity" if self.is_opportunity else "issue"} {[d.text for d in self.drives]} {self.importance:.2f} {self.urgency:.2f} {self.timestamp}'
    
    def get_age_factor(self, current_time: datetime, min_age: int, age_range: int) -> float:
        """Get age factor for signal"""
        if age_range > 0:
            return math.pow(0.5, ((current_time - self.timestamp).total_seconds()-min_age)/ (age_range * 3600))
        else:
            return 1.0  

@dataclass
class SignalCluster:
    """Represents a cluster of similar drive signals"""
    _id_counter: ClassVar[int] = 0
    _instances: ClassVar[WeakValueDictionary] = WeakValueDictionary()
    manager: 'DriveSignalManager'
    centroid: np.ndarray   # Center of the cluster
    signals: List[DriveSignal]
    drives: List[Drive]     # The drives this cluster relates to
    is_opportunity: bool   # True if opportunity cluster
    text: str            # Label for the cluster
    history: List[str] = field(default_factory=list)
    score: float = 0.0
    emotional_stance: Optional[EmotionalStance] = None
    id: str = field(init=False)
    
    def __post_init__(self):
        SignalCluster._id_counter += 1
        self.id = f"sc{SignalCluster._id_counter}"
        SignalCluster._instances[self.id] = self

    @classmethod
    def get_by_id(cls, id: str):
        try:
            return cls._instances.get(id)
        except:
            return None

    def to_string(self):
        return f'{self.id} {self.text}: {"opportunity" if self.is_opportunity else "issue"} {len(self.signals)} signals, score {self.score}'
    
    def to_full_string(self):
        return f'{self.id} Name: {self.text}\n    {"opportunity" if self.is_opportunity else "issue"};  score {self.score}\n    {self.emotional_stance.to_definition()}\n    signals:{"\n      ".join([s.text for s in self.signals[:10]])}\n'
    
    def add_signal(self, signal: DriveSignal) -> None:
        """Add a signal to the cluster and update centroid"""
        self.signals.append(signal)
        # Update centroid as mean of all embeddings
        for drive in signal.drives:
            for d2 in self.drives:
                if d2.id == drive.id:
                    break
            else:
                self.drives.append(drive)
        embeddings = [s.embedding for s in self.signals]
        self.centroid = np.mean(embeddings, axis=0)

    def cluster_name(self):
        if len(self.signals) == 0:
            self.text = 'NA'
        elif len(self.signals) == 1:
            self.text = self.signals[0].text
        else:
            prompt = [SystemMessage(content="""Given the following texts contained in a cluster, determine the most appropriate label for the cluster.
The label should be 8 - 10 words , and most closely represent the central theme of the cluster: 
                                
<texts>
{{$signals}}
</texts>

Respond in this format:
<label>Label for the cluster</label>

Only respond with the label, no other text.
End your response with:
</end>
""")]
            response = self.manager.llm.ask({"signals": '\n'.join([signal.text for signal in self.signals])}, prompt, temp=0.1, stops=['</end>'], max_tokens=20)
            if xml.find('<label>', response):
                self.text = xml.find('<label>', response).strip()
            else:
                self.text = self.signals[0].text
        return self.text
        
        
    def get_importance(self, current_time: datetime, min_age: int, age_range: int) -> float:
        """Get cluster importance as max of contained signals"""
        if age_range > 0:
            return max(s.importance * s.get_age_factor(current_time=current_time, min_age=min_age, age_range=age_range) for s in self.signals)
        else:
            return max(s.importance for s in self.signals)
        
    def get_urgency(self, current_time: datetime, min_age: int, age_range: int) -> float:
        """Get cluster urgency as max of contained signals"""
        if age_range > 0:
            return max(s.urgency * s.get_age_factor(current_time=current_time, min_age=min_age, age_range=age_range) for s in self.signals)
        else:
            return max(s.urgency for s in self.signals)
        
    def get_latest_timestamp(self) -> datetime:
        """Get most recent timestamp from signals"""
        return max(s.timestamp for s in self.signals)

class DriveSignalManager:
    def __init__(self, llm: LLM, context=None, embedding_dim=384):
        """Initialize detector with given embedding dimension"""
        self.clusters: List[SignalCluster] = []
        self.embedding_dim = embedding_dim
        self.similarity_threshold = 0.60
        self.llm = llm
        self.context = context
        self.current_time = None
        self.clustering_eps = 0.40

    def set_llm(self, llm: LLM):
        self.llm = llm
        
    def set_context(self, context):
        self.context = context
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get vector embedding for text using memory consolidation's embedding"""
        return _embedding_model.encode(text)

    def get_signal_cluster_by_name(self, name: str, create_if_missing: bool = False) -> SignalCluster:
        """Get signal cluster by name"""
        name = name.strip().lower()
        for cluster in self.clusters:
            if cluster.text.strip().lower() == name:
                return cluster
        if create_if_missing:
            return SignalCluster(manager=self, centroid=np.zeros(self.embedding_dim), signals=[], drives=set(), is_opportunity=False, text=name)
        return None
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def _find_closest_cluster(self, signal: DriveSignal) -> Tuple[int, float]:
        """Find index and similarity of closest cluster"""
        if not self.clusters or len([c for c in self.clusters if c.is_opportunity == signal.is_opportunity]) == 0:
            return -1, 0.0
            
        similarities = [self._cosine_similarity(signal.embedding, c.centroid) 
                       for c in self.clusters if c.is_opportunity == signal.is_opportunity]
        closest_idx = np.argmax(similarities)
        return closest_idx, similarities[closest_idx]
        
    def construct_signal(self, signal_hash, drives, current_time):
        signal_type = hash_utils.find('type', signal_hash).strip()
        desc = hash_utils.find('description', signal_hash).strip()
        signal_text = hash_utils.find('signal', signal_hash).strip()
        drive_ids = hash_utils.find('drive_ids', signal_hash).strip()
        if desc is None or desc == '':
            return None
        try:
            importance = float(hash_utils.find('importance', signal_hash).strip())
        except:
            return None
        try:
            urgency = float(hash_utils.find('urgency', signal_hash).strip())
        except:
            return None
                    
        embedding = self._get_embedding(desc)
        drive_ids = drive_ids.split('@')
        drive_ids = [d.strip() for d in drive_ids]
        signal_drives = []
        for id in drive_ids:
            if id in Drive._instances:  
                signal_drives.append(Drive._instances[id])
            else:
                print(f"Warning: Drive {id} not found")
                    
        signal = DriveSignal(
            text=f'{signal_text}: {desc}',
            drives=signal_drives,
            is_opportunity=signal_type == 'opportunity',
            importance=importance,
            urgency=urgency,
            timestamp=current_time,
            embedding=embedding
        )
        return signal

        
    def analyze_text(self, text: str, drives: List[Drive], current_time: datetime) -> List[DriveSignal]:
        """Analyze text for drive-related signals"""

        self.current_time = current_time
        try:
            signals = []

            prompt = [SystemMessage(content="""Analyze the following text for issues or opportunities related to these drives: 
        
<Drives>
{{$drives}}
</Drives>
           
<Text>
{{$text}}
</Text>

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:
                                    
#signal 3-4 words naming the issue or opportunity detected
#type opportunity / issue
#description 6-8 words further detailing the opportunity or issue
#drive_ids a @ separated list of drive ids this signal is related to. A drive id is a string of the form 'd123'
#importance 0-1
#urgency 0-1
##

Only respond if you find a clear and strong signal. Multiple signals can be on separate lines.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]
            
            response = self.llm.ask({"text": text, "drives": '\n'.join([f'{d.id} {d.text}' for d in drives])}, prompt, temp=0.1, stops=['</end>'], max_tokens=180)
            if not response:
                return []
                    
            for signal_hash in hash_utils.findall_forms(response):
                signal = self.construct_signal(signal_hash, drives, current_time)
                if signal:
                    signals.append(signal)
                   
            self.process_signals(signals)
            # print(f"Found {len(signals)} signals")
            return signals
        except Exception as e:
            traceback.print_exc()
            print(f"Error analyzing text: {e}")
            return []
 
    def process_signals(self, signals: List[DriveSignal]):
        """Process new signals and update clusters"""
        updated_clusters = []
        for signal in signals:
            closest_idx, similarity = self._find_closest_cluster(signal)
            
            if similarity > self.similarity_threshold:
                # Add to existing cluster
                self.clusters[closest_idx].add_signal(signal)
                if not any(self.clusters[closest_idx] is cluster for cluster in updated_clusters):
                    updated_clusters.append(self.clusters[closest_idx])
            else:
                # Create new cluster
                new_cluster = SignalCluster(
                    manager=self,
                    centroid=signal.embedding,
                    signals=[signal],
                    drives=signal.drives,
                    is_opportunity=signal.is_opportunity,
                    text=signal.text
                )
                self.clusters.append(new_cluster)
                if not any(new_cluster is cluster for cluster in updated_clusters):
                    updated_clusters.append(new_cluster)

        for cluster in updated_clusters:
            cluster.cluster_name()
        #scored = self.get_scored_clusters()
        #return scored
                
    def get_active_signals(self, max_age_hours: int = 24) -> List[SignalCluster]:
        """Get clusters with recent signals"""
        if self.context:
            current_time = self.context.simulation_time
        else:
            current_time = self.current_time
        return [c for c in self.clusters 
                if (current_time - c.get_latest_timestamp()).total_seconds() / 3600 < max_age_hours] 
    
    def get_clusters(self) -> List[SignalCluster]:
        """Get all clusters"""
        return self.clusters
    
    def get_cluster_count(self) -> int:
        """Get number of clusters"""
        return len(self.clusters)
    
    def get_scored_clusters(self):
        if len(self.clusters) < 1:
            return []
        if self.context:
            current_time = self.context.simulation_time
        else:
            current_time = self.current_time
        """Score a cluster based on urgency, importance, signal count and recency"""
        # Normalize signal count to 0-100
        max_cluster_signals = max(len(c.signals) for c in self.clusters)
        max_age = max((current_time - c.get_latest_timestamp()).total_seconds() / 3600 for c in self.clusters)+1
        min_age = min((current_time - c.get_latest_timestamp()).total_seconds() / 3600 for c in self.clusters)
        age_range = max_age - min_age if max_age > min_age else 0.25 #timestep is 1/4 hour

        scored_clusters = []
        for cluster in self.clusters:
            try:
                signal_ratio = 100*len(cluster.signals) / max_cluster_signals
                # Normalize recency to 0-100 
                min_cluster_signal_age = min((current_time - s.timestamp).total_seconds() / 3600 for s in cluster.signals)
                if min_cluster_signal_age < min_age or age_range < 0.25:
                    recency = 100
                else:
                    recency = 100-80*(min_cluster_signal_age-min_age)/ age_range
                urgency = 100*cluster.get_urgency(current_time=current_time, min_age=min_age, age_range=age_range)
                importance = 100*cluster.get_importance(current_time=current_time, min_age=min_age, age_range=age_range)
                score = math.pow(urgency * importance * signal_ratio * recency, 0.25)
                cluster.score = score
                scored_clusters.append((cluster, score))
            except Exception as e:
                traceback.print_exc()
                print(f"Error scoring cluster: {e}")
        return sorted(scored_clusters, key=lambda x: x[1], reverse=True)
        
    def recluster(self, eps: float=-1, min_samples: int = 2) -> None:
        """Rebuild clusters using DBSCAN to remove outliers and optimize grouping.
        Preserves recent high-importance outliers as single-signal clusters.
        
        Args:
            eps: DBSCAN epsilon parameter for neighborhood size (default 0.3)
            min_samples: Minimum samples for core point (default 2)
        """
        if not self.clusters:
            return
        
        if eps == -1:
            eps = self.clustering_eps

        # Collect all signals and their embeddings
        all_signals = []
        embeddings = []
        for cluster in self.clusters:
            all_signals.extend(cluster.signals)
            embeddings.extend([s.embedding for s in cluster.signals])
        
        # Run DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Group signals by cluster label
        new_clusters = defaultdict(list)
        outlier_signals = []
        current_time = self.context.simulation_time if self.context else datetime.now()

        for signal, label in zip(all_signals, labels):
            if label != -1:
                # Add to regular cluster
                new_clusters[label].append(signal)
            else:
                outlier_signals.append(signal)
            
        # Create new cluster objects
        self.clusters = []
        for signals in new_clusters.values():
            if signals:
                # Create cluster with first signal
                cluster = SignalCluster(
                    manager=self,
                    centroid=signals[0].embedding,
                    signals=[signals[0]],
                    drives=signals[0].drives,
                    is_opportunity=signals[0].is_opportunity,
                    text=signals[0].text
                )
                # Add remaining signals
                for signal in signals[1:]:
                    cluster.add_signal(signal)
                cluster.cluster_name()
                self.clusters.append(cluster)
        
        num_clusters = len(self.clusters)
        num_clustered_signals = sum(len(c.signals) for c in self.clusters) + len(outlier_signals)
        if num_clustered_signals > 24 and num_clusters > 0:
            # if num_clusters < sqrt num_signals, clusters are too large, decrease eps to require closer signals
            eps_adjustment = 0.01*(num_clustered_signals**0.5 - num_clusters)
            if self.clustering_eps - eps_adjustment < 0 or self.clustering_eps - eps_adjustment > 0.5:
                print(f"Warning: eps_adjustment {eps_adjustment} is out of range")
                eps_adjustment = 0.0
            self.clustering_eps = self.clustering_eps - eps_adjustment
            
        # Check if outliers should be preserved
        for outlier_signal in outlier_signals:
            age_minutes = max(30, (current_time - outlier_signal.timestamp).total_seconds() / 60)
            if (30/age_minutes <= 30 * outlier_signal.importance * outlier_signal.urgency)**.33 >= 0.6:
                # Create singleton cluster for important recent outlier
                outlier_cluster = SignalCluster(
                    manager=self,
                    centroid=outlier_signal.embedding,
                    signals=[outlier_signal],
                    drives=outlier_signal.drives,
                    is_opportunity=outlier_signal.is_opportunity,
                    text='outliner_'+outlier_signal.text
                )
                # Add remaining signals
                outlier_cluster.cluster_name()
                self.clusters.append(outlier_cluster)
            
        self.get_scored_clusters() # rank new clusters
        print(f"Reclustered into {len(self.clusters)} clusters")
        
