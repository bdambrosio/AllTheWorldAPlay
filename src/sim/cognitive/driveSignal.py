from datetime import datetime
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
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


_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@dataclass(frozen=True)
class Drive:
    """Represents a character drive with semantic embedding"""
    text: str
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.embedding is None:
            # Use object.__setattr__ to set field of frozen instance
            object.__setattr__(self, 'embedding', 
                _embedding_model.encode(self.text))
    
    def __hash__(self):
        return hash(self.text)
    
    def __eq__(self, other):
        if not isinstance(other, Drive):
            return False
        return self.text == other.text

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

    def get_age_factor(self, current_time: datetime, min_age: int, age_range: int) -> float:
        """Get age factor for signal"""
        if age_range > 0:
            return math.pow(0.5, ((current_time - self.timestamp).total_seconds()-min_age)/ (age_range * 3600))
        else:
            return 1.0  

@dataclass
class SignalCluster:
    """Represents a cluster of similar drive signals"""
    manager: 'DriveSignalManager'
    centroid: np.ndarray   # Center of the cluster
    signals: List[DriveSignal]
    drives: Set[Drive]     # The drives this cluster relates to
    is_opportunity: bool   # True if opportunity cluster
    text: str            # Label for the cluster
    history: List[str] = field(default_factory=list)
    score: float = 0.0
    def to_string(self):
        return f'{self.text}: {"opportunity" if self.is_opportunity else "issue"} {len(self.signals)} signals, score {self.score}'
    
    def add_signal(self, signal: DriveSignal) -> None:
        """Add a signal to the cluster and update centroid"""
        self.signals.append(signal)
        # Update centroid as mean of all embeddings
        self.drives.update(signal.drives)
        embeddings = [s.embedding for s in self.signals]
        self.centroid = np.mean(embeddings, axis=0)
        prompt = [SystemMessage(content="""Given the following texts contained in a cluster, determine the most appropriate label for the cluster.
The label should be of similar length to the individual texts, and most closely represent the central theme of the cluster: 
                                
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
        self.similarity_threshold = 0.6
        self.llm = llm
        self.context = context
        self.current_time = None
    def set_llm(self, llm: LLM):
        self.llm = llm
        
    def set_context(self, context):
        self.context = context
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get vector embedding for text using memory consolidation's embedding"""
        return _embedding_model.encode(text)
        
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
        drive_names = hash_utils.find('drive', signal_hash).strip()
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
        drive_names = drive_names.split('@')
        drive_names = [d.strip() for d in drive_names]
        signal_drives = set([d for d in drives if d.text in drive_names])
                    
        signal = DriveSignal(
            text=f'{signal_text}; {desc}',
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
                                    
#signal
#type opportunity / issue
#description 6-8 words describing the opportunity or issue
#drive a @ separated list of drives this signal is related to
#importance 0-1
#urgency 0-1
##

Only respond if you find a clear and strong signal. Multiple signals can be on separate lines.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
""")]
            
            response = self.llm.ask({"text": text, "drives": '\n'.join([d.text for d in drives])}, prompt, temp=0.1, stops=['</end>'], max_tokens=180)
            if not response:
                return []
                    
            for signal_hash in hash_utils.findall_forms(response):
                signal = self.construct_signal(signal_hash, drives, current_time)
                if signal:
                    signals.append(signal)
                   
            self.process_signals(signals)
            print(f"Found {len(signals)} signals")
            return signals
        except Exception as e:
            traceback.print_exc()
            print(f"Error analyzing text: {e}")
            return []
 
    def process_signals(self, signals: List[DriveSignal]):
        """Process new signals and update clusters"""
        for signal in signals:
            closest_idx, similarity = self._find_closest_cluster(signal)
            
            if similarity > self.similarity_threshold:
                # Add to existing cluster
                self.clusters[closest_idx].add_signal(signal)
            else:
                # Create new cluster
                self.clusters.append(SignalCluster(
                    manager=self,
                    centroid=signal.embedding,
                    signals=[signal],
                    drives=signal.drives,
                    is_opportunity=signal.is_opportunity,
                    text=signal.text
                ))
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
        
    def recluster(self, eps: float = 0.37, min_samples: int = 2) -> None:
        """Rebuild clusters using DBSCAN to remove outliers and optimize grouping.
        Preserves recent high-importance outliers as single-signal clusters.
        
        Args:
            eps: DBSCAN epsilon parameter for neighborhood size (default 0.3)
            min_samples: Minimum samples for core point (default 2)
        """
        if not self.clusters:
            return
        
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
        current_time = self.context.simulation_time if self.context else datetime.now()
        
        for signal, label in zip(all_signals, labels):
            if label != -1:
                # Add to regular cluster
                new_clusters[label].append(signal)
            else:
                # Check if outlier should be preserved
                age_minutes = (current_time - signal.timestamp).total_seconds() / 60
                if (age_minutes <= 30 or 
                    (signal.importance >= 0.8 and 
                    signal.urgency >= 0.8)):
                    # Create singleton cluster for important recent outlier
                    new_clusters[f'outlier_{len(new_clusters)}'].append(signal)
            
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
                self.clusters.append(cluster)
            
        self.get_scored_clusters() # rank new clusters
        print(f"Reclustered into {len(self.clusters)} clusters")
        
