from __future__ import annotations
from itertools import tee
import json
from reprlib import Repr
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
from utils.Messages import SystemMessage, UserMessage
import utils.llm_api as llm_api
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from weakref import WeakValueDictionary
#from sim.prompt import ask
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.agh import Character, Goal  # Only imported during type checking
    from sim.cognitive.EmotionalStance import EmotionalStance

try:
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
except Exception as e:
    print(f"Warning: Could not load embedding model locally: {e}")
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def noisy_add(a: float, b: float) -> float:
    """
    Implements noisy-or operation for probabilities a and b in range [0,1].
    Returns a value that is at least as large as max(a,b) but never exceeds 1.0.
    The result represents the probability of either event occurring.
    
    Args:
        a: First probability (0 <= a <= 1)
        b: Second probability (0 <= b <= 1)
        
    Returns:
        Combined probability (0 <= result <= 1)
    """
    if not (0 <= a <= 1 and 0 <= b <= 1):
        raise ValueError("Inputs must be between 0 and 1")
    
    # If either input is 1.0, result must be 1.0
    if a == 1.0 or b == 1.0:
        return 1.0
        
    # For non-1.0 inputs, use the formula: 1 - (1-a)(1-b)
    # This ensures result is at least max(a,b) but never exceeds 1.0
    return 1.0 - (1.0 - a) * (1.0 - b)

@dataclass
class Drive:
    """Represents a character drive with semantic embedding"""
    _id_counter: ClassVar[int] = 0
    _instances: ClassVar[WeakValueDictionary] = WeakValueDictionary()
    text: str
    activation: float = 1.0
    embedding: Optional[np.ndarray] = None
    attempted_goals: List[Goal] = field(default_factory=list)   
    satisfied_goals: List[Goal] = field(default_factory=list)   

    id: str = field(init=False)
    
    @classmethod
    def get_by_id(cls, id: str):
        if id in cls._instances:
            return cls._instances[id]
        else:
            return None
        
    @classmethod
    def get_by_text(cls, text: str):
        for drive in cls._instances.values():
            if drive.text == text:
                return drive
        return None
    
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
        return self.id == other.id
    
    def update_on_goal_completion(self, character: Character, goal: Goal, completion_statement: str) -> Optional[Drive]:
        """Update drive based on goal completion
        Core Transformation Patterns
1. Scope Transformations
Generalize: Expand concern from individual to collective or systemic level.

Example: "Securing my inheritance" → "Fighting for fair inheritance laws for all farmers"
Trigger: Recognition that personal problems reflect broader patterns

Particularize: Narrow concern from abstract/general to specific/individual.

Example: "Finding justice in society" → "Ensuring my family acknowledges my contributions"
Trigger: Frustration with abstract goals; discovery of specific leverage point

2. Temporal Transformations
Future-Shift: Reorient from past grievances to future possibilities.

Example: "Reclaiming what I deserved" → "Building what I desire"
Trigger: New relationships or opportunities that make future more salient than past

Legacy-Shift: Reorient from immediate outcomes to long-term impact.

Example: "Acquiring farmland" → "Creating an agricultural legacy for generations"
Trigger: Confrontation with mortality or contemplation of one's lasting impact

3. Agency Transformations
Internalize: Shift from external validation to internal standards.

Example: "Gaining others' recognition" → "Achieving personal standards of excellence"
Trigger: Repeated disappointment with external validation sources

Collectivize: Shift from individual to collaborative agency.

Example: "Succeeding through my own efforts" → "Building success through partnership"
Trigger: Powerful cooperative experiences; recognition of interdependence

4. Value Transformations
Means-Ends Inversion: What was once a means becomes an end in itself.

Example: "Working the land to earn a living" → "Finding fulfillment in agricultural craft"
Trigger: Discovery of unexpected satisfaction in process rather than outcome

Value Transcendence: Replace concrete goal with abstract value it represents.

Example: "Owning this specific farm" → "Creating a place that embodies security and belonging"
Trigger: Reflection on deeper motivations underlying concrete desires

5. Compensation Transformations
Substitution: Replace blocked goal with achievable alternative that satisfies same need.

Example: "Inheriting family land" → "Purchasing and revitalizing abandoned farmland"
Trigger: Repeated failure combined with discovery of alternative path

Sublimation: Transform problematic desire into socially valuable alternative.

Example: "Violently confronting those who wronged me" → "Becoming an advocate for justice through legal means"
Trigger: Ethical growth or social pressure against original expression

6. Integration Transformations
Synthesis: Merge competing drives into unified higher-order drive.

Example: "Finding love" + "Securing land" → "Building a family legacy through shared stewardship"
Trigger: Recognition of how seemingly separate goals can mutually reinforce

Differentiation: Break general drive into more specific components.

Example: "Achieving success" → "Mastering agricultural techniques, building community standing, ensuring financial stability"
Trigger: Growing sophistication in understanding of complex goal"""

        prompt = [UserMessage(content="""Given the following goal, update the motivating drive to reflect the goal completion.
        
Goal just satisfied:
{{$goal}}

Drive to update:
{{$drive}}
                              
Goals attempted for this drive: 
{{$attempted_goals}}

Goals known to have been satisfied for this drive: 
{{$satisfied_goals}}

A number of goals attempted but not satisfied may indicate this drive is not a good fit for the character at this time. 
The character might respond, for example, by 
 - becoming more determined, 
 - changing the target to something more abstract (e.g., a thing like this rather than this thing).
 - changing the target to something more specific (e.g., this thing here rather than my ideal thing).
 - changing the target to something more modest (e.g. from 'my ideal house' to 'a house').
 - rejecting the drive (e.g. from 'to be rich' to 'resent the rich' or in the extreme case 'kill the rich').

Alternatively, a number of satisfied goals may indicate the character is ready for expansion of the drive in space, time, or scope.
In this case the character might respond by 
 - broadening the scope of the drive (e.g., from 'my' to 'my family' or 'my community' or 'my society'),
 - deepening the motivation (e.g., from 'to be accepted' to 'to be respected' or 'to be loved' or 'to be happy'),
 - changing the target to something more specific (e.g., this thing here rather than my ideal thing).

Some drives are recurring, and will ebb or strengthen with goal satisfaction.
Other drives evolve over time into new, more ambitious or more modest goals. 
                              
The basic personality of the character for whom you are updating the drive is:

{{$character}}

Use your knowledge of human and animal nature, together with your understanding of story and narrative arc, to create an evolution of this drive.
                              
Respond with the updated drive (8-12 words) only.
Do not include any introductory, explanatory, formative or discursive text.
End your response with:
</end>
""")]       
        result = character.llm.ask({"goal": f'{goal.name}: {goal.description}; termination criterion: {goal.termination}', 
                                    "drive": f'{self.text}', 
                                    "attempted_goals": '\n'.join([f'{g.name}: {g.description}; termination criterion: {g.termination}' for g in self.attempted_goals]),
                                    "satisfied_goals": '\n'.join([f'{g.name}: {g.description}; termination criterion: {g.termination}' for g in self.satisfied_goals]),
                                    "character": f'{character.character}'}, prompt, tag='DriveSignal.update_drive', temp=0.1, stops=['</end>'], max_tokens=30)
        if result:
            try:
                return Drive(text=result.strip(), activation=self.activation*0.9)
            except:
                return None
        else:
            return None
        

    def update_on_goal_completion_draft(self,character: Character, goal, completion_statement: str) -> Optional[Drive]:
        """ Called whenever a goal or task linked to this drive is finished.
        The LLM returns JSON with:
            new_drive        - new wording or same wording
            action           - 'evolve' | 'lower_activation' | 'lower_priority' | 'retire'
            activation_delta - int (optional, default 0)
        """

        # ------ build prompt --------------------------------------------------
        system_msg = SystemMessage(content="You are an expert dramaturg and psychologist.")
        mission = """
A goal or task linked to this drive has just been completed.

##Goal or Task
{{$name}}: {{$description}}  (termination: {{$termination}})

##Completion information
{{$completion_statement}}

##Drive
{{$drive}}
Current activation level: {{$activation}}

Recent attempts for this drive (max 5):
{{$attempts}}

Recently satisfied goals for this drive (max 5):
{{$satisfied}}

Character sketch:
{{$character}}

##Instructions
Please decide how the drive should change.  Reply **only** with a JSON object:
  {
    "new_drive": "8-12 word description",
    "action": "evolve | lower_activation | lower_priority | retire",
    "activation_delta": int   # optional, 0 … +100
  }
End with </end>"""

        prompt = [system_msg,UserMessage(content=mission)]
        bindings = {
                "name": goal.name,
                "description": goal.description,
                "termination": goal.termination,
                "completion_statement": completion_statement,
                "drive": self.text,
                "activation": self.activation_level,
                "attempts": "\n".join(
                    f"• {g.name} ({g.termination})" for g in self.attempted_goals[-5:]
                ) or "—",
                "satisfied": "\n".join(
                    f"• {g.name} ({g.termination})" for g in self.satisfied_goals[-5:]
                ) or "—",
                "character": character.character.strip()[:400]  # truncate safety
            }

        # ------ ask model -----------------------------------------------------
        raw = character.llm.ask(bindings, prompt, tag="Drive.update", temp=0.2, stops=["</end>"], max_tokens=120)
        if not raw:
            return None

        try:
            data = json.loads(raw)
        except Exception:
            # failed to parse; keep drive unchanged
            return None

        # ------ apply result --------------------------------------------------
        action = data.get("action", "evolve")
        new_text = data.get("new_drive", self.text).strip()
        delta = int(data.get("activation_delta", 0))

        if action == "retire":
            # caller removes the drive entirely
            return None

        # create updated Drive object
        updated = Drive(new_text)
        updated.activation_level = max(0, min(100, self.activation_level + delta))

        # caller can inspect `action` to move drive lower in list if needed
        updated.metadata["action"] = action
        return updated


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
    
    def to_full_string(self):
        return f"""{self.text}. Issue or Opportunity: {"opportunity" if self.is_opportunity else "issue"} Importance: {self.importance:.2f} Urgency: {self.urgency:.2f}
{'\n    '.join(['Drive: ' + d.text for d in self.drives])}"""


    def get_age_factor(self, current_time: datetime, min_age: int, age_range: int) -> float:
        """Get age factor for signal"""
        if age_range > 0:
            try:
                return math.pow(0.5, min(0.0, max(10.0, ((current_time - self.timestamp).total_seconds()-min_age)/ (age_range * 3600))))
            except:
                return 0.5
        else:
            return 1.0  

class SignalCluster:
    """Represents a cluster of similar drive signals"""
    _id_counter: ClassVar[int] = 0
    _instances: ClassVar[WeakValueDictionary] = WeakValueDictionary()

    def __init__(self, manager: 'DriveSignalManager', centroid: np.ndarray, signals: List[DriveSignal], drives: List[Drive], is_opportunity: bool, text: str):
        self.manager = manager
        self.centroid = centroid
        self.signals = signals
        self.drives = drives
        self.is_opportunity = is_opportunity
        self.text = text.lstrip(":")
        self.history = []
        self.score = 0.0
        self.emotional_stance = None
        self.new_signal_count = 0 # number of new signals added to this cluster since last cluster name update
        self.id = f"sc{SignalCluster._id_counter}"
        SignalCluster._instances[self.id] = self
        SignalCluster._id_counter += 1

    @classmethod
    def get_by_id(cls, id: str):
        if id in cls._instances:
            return cls._instances[id]
        else:
            return None

    def to_string(self):
        return f"""{self.id} {self.text}: {"opportunity" if self.is_opportunity else "issue"}; {len(self.signals)} signals; score: {self.score}"""
    
    def to_full_string(self):
        return f'{self.id} Name: {self.text}. Issue or Opportunity: {"opportunity" if self.is_opportunity else "issue"};  score {self.score}\n    Emotions: {self.emotional_stance.to_string()}\n'
    
    def add_signal(self, signal: DriveSignal) -> None:
        """Add a signal to the cluster and update centroid"""
        self.signals.append(signal)
        self.new_signal_count += 1
        # Update centroid as mean of all embeddings
        for drive in signal.drives:
            if drive not in self.drives:
                self.drives.append(drive)
        embeddings = [s.embedding for s in self.signals]
        self.centroid = np.mean(embeddings, axis=0)
        if self.new_signal_count > len(self.signals)/5:
            self.cluster_name()
            self.new_signal_count = 0


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

Respond in this hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Close the hash-formatted text with a separate line containing only ##
be careful to insert line breaks only where shown, separating a value from the next tag:

#label Label for the cluster
##

Only respond with the label, no other text.
End your response with:
</end>
""")]
            response = self.manager.llm.ask({"signals": '\n'.join([signal.text for signal in self.signals])}, prompt, tag='SignalCluster.cluster_name', temp=0.1, stops=['</end>'], max_tokens=20)
            if hash_utils.find('label', response):
                self.text = hash_utils.find('label', response).strip()
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
    def __init__(self, owner: Character, llm: LLM, context=None, ask=None, embedding_dim=384):
        """Initialize detector with given embedding dimension"""
        self.clusters: List[SignalCluster] = []
        self.embedding_dim = embedding_dim
        self.similarity_threshold = 0.60
        self.llm = llm
        self.context = context
        self.current_time = None
        self.owner = owner
        self.clustering_eps = 0.40
        self.ask = ask

    def set_llm(self, llm: LLM):
        self.llm = llm
        
    def set_context(self, context):
        self.context = context

    def clear_clusters(self, goals: List[Goal]):
        new_clusters = []
        for cluster in self.clusters:
            if cluster in [g.signalCluster for g in goals]:
                new_clusters.append(cluster)
        self.clusters = new_clusters
        
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
        
        #compute signal importance as raw importance times activation of drives
        drive_ids = drive_ids.split('@')
        drive_ids = [d.strip() for d in drive_ids]
        activation = 0.0
        signal_drives = []
        for id in drive_ids:
            id = id.strip()
            if id in Drive._instances:  
                signal_drives.append(Drive._instances[id])
                activation = noisy_add(activation, Drive._instances[id].activation)
            elif id =='' or id is None:
                continue
            else:
                print(f"Warning: Drive {id} not found")
        importance = activation*importance

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

        #if True:
        #    return []
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

<Surroundings>
{{$surroundings}}
</Surroundings>

Consider <Surroundings> carefully for additional context. 
Signals can originate from elements explicitly mentioned there, especially those related to safety, survival, or immediate opportunities.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
The valid tags in this response are signal, type, description, drive_ids, importance, urgency.
The type tag takes a single work as its content, either issue or opportunity.
be careful to insert line breaks only where shown, separating a value from the next tag, as in the following example:

#signal 3-4 words max tersely naming the key theme or essence (e.g., "Food Source Discovered")
#type issue or opportunity
#description 4-6 words max explicitly identifying the specific detail or actionable aspect of the signal (e.g., "Apple trees nearby provide food").
#drive_ids a @ separated list of drive ids this signal is related to. A drive id is a string of the form 'd123'
#importance 1.0-10.0 (importance of the signal to the character - log scale: 1.0 is not important, 10.0 is life-or-death important.  Most signals are in the 2.0-4.0 range.)
#urgency 1.0-10.0 (urgency of the signal to the character - 1.0 is not urgent, 10.0 demands response within a few seconds. Most signals are in the 2.0-4.0 range.)
##

Only respond if you find a clear and strong signal. Report only the single most urgent importantsignal.
Do not include any introductory, explanatory, or discursive text.
End your response with:
                          
</s>

""")]
            
            response = self.llm.ask({"text": text, "drives": '\n'.join([f'{d.id} {d.text}' for d in drives]), 
                                     "surroundings": self.owner.look_percept}, 
                                     prompt, tag='DriveSignal.analyze_text', temp=0.1, stops=['</s>'], max_tokens=180)
            if not response:
                return []
            print(f'\npercept {text}')        
            for signal_hash in hash_utils.findall_forms(response):
                signal = self.construct_signal(signal_hash, drives, current_time)
                if signal:
                    print(f'    {"opportunity" if signal.is_opportunity else "issue"} {signal.text} ({signal.importance:.2f}, {signal.urgency:.2f})')
                    signals.append(signal)
                   
            self.process_signals(signals)
            # print(f"Found {len(signals)} signals")
            return signals
        except Exception as e:
            traceback.print_exc()
            print(f"Error analyzing text: {e}")
            return []
 
    def check_drive_signal(self, drive: Drive) -> List[DriveSignal]:
        """Analyze drive for self-awareness signals"""

        mission = f"""Given the information following, evaluate the awareness of the owner at the current time with respect to the following need: 
        
{drive.id}: {drive.text}

"""

        suffix = f"""

Report any issues or opportunities you expect the owner is aware of with respect to this need and only this need:

{drive.id}: {drive.text}

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
The valid tags in this response are signal, type, description, drive_ids, importance, urgency.
The type tag takes a single work as its content, either issue or opportunity.
be careful to insert line breaks only where shown, separating a value from the next tag, as in the following example:

#signal 3-4 words briefly naming the key theme or essence (e.g., "Food Source Discovered")
#type issue or opportunity
#description 4-7 words explicitly identifying or elaborating the specific detail or actionable aspect of the signal (e.g., "Apple trees nearby provide food").
#drive_ids {drive.id}
#importance 0.0-1.0
#urgency 0.0-1.0
##

Only respond if you find a clear and strong signal. Multiple signals can be on separate lines. Report at most 2 signals.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
"""
            
        try:
            response = self.ask(self.owner, prefix=mission, suffix=suffix, addl_bindings= {}, tag = 'DriveSignal.check_drive_signal', max_tokens=120)
        except Exception as e:
            traceback.print_exc()
            print(f"Error checking drive signal: {e}")
            return []
        if not response:
            return []
        signals = []
        for signal_hash in hash_utils.findall_forms(response):
            signal = self.construct_signal(signal_hash, [drive], self.owner.context.simulation_time)

            if signal:
                raw_signal_importance = signal.importance
                signal.importance = drive.activation*signal.importance
                drive.activation = .5*drive.activation + .5*max(raw_signal_importance, signal.urgency)
                print(f'    {"opportunity" if signal.is_opportunity else "issue"} {signal.text} ({signal.importance:.2f}, {signal.urgency:.2f})')
                signals.append(signal)
                   
        self.process_signals(signals)
        # print(f"Found {len(signals)} signals")
        return signals
            
            
    def check_drive_signals(self):
        """Check all drives for signals"""
        for drive in self.owner.drives:
            self.check_drive_signal(drive)
            
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
                # below not needed - singleton clusters already have a name
                #if not any(new_cluster is cluster for cluster in updated_clusters):
                #    updated_clusters.append(new_cluster)

                
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
                urgency = 10*cluster.get_urgency(current_time=current_time, min_age=min_age, age_range=age_range)
                importance = 10*cluster.get_importance(current_time=current_time, min_age=min_age, age_range=age_range)
                score = math.pow(urgency * importance * signal_ratio * recency, 0.25)
                cluster.score = score
                cluster.signals.sort(key=lambda x: x.timestamp, reverse=True)
                cluster.signals = cluster.signals[:25]
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
                    text=signals[0].text.lstrip(':')
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
            age_minutes = max(15, (current_time - outlier_signal.timestamp).total_seconds() / 60)
            recency = 15/age_minutes
            signal_strength = outlier_signal.importance * outlier_signal.urgency
            if (recency * signal_strength)**.33 >= 0.5:
                # Create singleton cluster for important recent outlier
                outlier_cluster = SignalCluster(
                    manager=self,
                    centroid=outlier_signal.embedding,
                    signals=[outlier_signal],
                    drives=outlier_signal.drives,
                    is_opportunity=outlier_signal.is_opportunity,
                    text='outlier_'+outlier_signal.text.lstrip(':')
                )
                self.clusters.append(outlier_cluster)
            
        self.get_scored_clusters() # rank new clusters
        print(f"Reclustered into {len(self.clusters)} clusters")
        
    def get_signals_for_drive(self, drive: Drive=None, drive_id=None, n: int = 3) -> List[Tuple[SignalCluster, float]]:
        """Get the n highest scoring SignalClusters most similar to the given Drive.
        
        Args:
            drive: The Drive to find similar clusters for
            n: Maximum number of clusters to return (default 3)
            
        Returns:
            List of tuples (cluster, similarity_score) sorted by descending similarity * cluster.score
        """
        if not self.clusters:
            return []
        
        # Get scored clusters first
        scored_clusters = self.get_scored_clusters()
        if not scored_clusters:
            return []
        
        # Calculate similarity scores
        if drive is None and drive_id is not None:
            drive = Drive.get_by_id(drive_id)
        if drive is None:
            return []
        similar_clusters = []
        for cluster, cluster_score in scored_clusters:
            similarity = self._cosine_similarity(drive.embedding, cluster.centroid)
            # Combine similarity with cluster's existing score
            combined_score = similarity * cluster_score
            similar_clusters.append((cluster, combined_score))
        
        # Sort by combined score and return top n
        similar_clusters.sort(key=lambda x: x[1], reverse=True)
        return [cluster for cluster, _ in similar_clusters[:n]]
        
