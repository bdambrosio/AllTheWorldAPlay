"""Character drives and drive-awareness checks.

Drive: a standing motivation named in a play (Drive("find food and water")); shown in prompts
with its id, text and activation.

DriveSignalManager.check_drive_signals (called when a scene goal is set up,
Character.instantiate_narrative_goal) asks the LLM, per drive, what issues or opportunities the
character is aware of for that drive, and folds the strongest into drive.activation.

Removed 2026-11-09 (recoverable from git): analyze_text, an LLM pass over percepts for drive
signals, and the SignalCluster machinery that clustered them (DBSCAN, LLM cluster naming,
scoring) for goal generation and emotional stance. analyze_text was switched off in 867ea34
("speedups", 2025-07-19) and check_drive_signals cleared the clusters, so the clusters were
always empty and every consumer saw nothing.
"""
from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datetime import datetime
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional
from weakref import WeakValueDictionary

import numpy as np

from utils import hash_utils
from utils.llm_api import LLM
from utils.Messages import SystemMessage, UserMessage

if TYPE_CHECKING:
    from sim.agh import Character, Goal  # Only imported during type checking

from utils.embedding_model import get_embedding_model
_embedding_model = get_embedding_model()


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

@dataclass
class DriveSignal:
    """Represents a detected issue or opportunity related to a drive"""
    text: str                # The text that triggered this signal
    drives: List[Drive]      # Reference to the related drives
    type: str    # Issue, Opportunity, or Surprise
    importance: float       # 0-1.0 scale of importance
    urgency: float         # 0-1.0 scale of urgency
    surprise: float        # 0-1.0 scale of surprise
    timestamp: datetime    # When this was last detected
    embedding: np.ndarray  # Vector embedding of the text

    def to_string(self):
        return f'{self.text}: {self.type}. {[d.text for d in self.drives]} {self.importance:.2f} {self.urgency:.2f} {self.surprise:.2f} {self.timestamp}'
    
    def to_full_string(self):
        return f"""{self.text}. {self.type} Importance: {self.importance:.2f} Urgency: {self.urgency:.2f} Surprise: {self.surprise:.2f}
{'\n    '.join(['Drive: ' + d.text for d in self.drives])}"""


class DriveSignalManager:
    """Per-character drive-awareness checks (see module docstring)."""
    def __init__(self, owner: Character, llm: LLM, context=None, ask=None, embedding_dim=384):
        self.llm = llm
        self.context = context
        self.owner = owner
        self.ask = ask
        self.last_signals: Dict[str, List[DriveSignal]] = {}  # drive id -> signals from its last check

    def set_llm(self, llm: LLM):
        self.llm = llm

    def set_context(self, context):
        self.context = context

    def _get_embedding(self, text: str) -> np.ndarray:
        return _embedding_model.encode(text)

    # echoed template text, seen as a 'signal' with importance/urgency 1.0
    _TEMPLATE_ECHOES = ('words briefly naming', 'words explicitly identifying', '0.0-1.0')

    @staticmethod
    def _unit(value, default=None):
        """A float from a tag value, clamped to [0, 1] (unclamped values pushed activation past 1,
        and noisy_add then raised)."""
        try:
            return min(1.0, max(0.0, float(str(value).strip().split()[0])))
        except (ValueError, IndexError, TypeError):
            return default

    def construct_signal(self, signal_hash, drives, current_time):
        """A DriveSignal from one hash form, or None. drives: the drive(s) being checked, used when
        the reply's drive_ids name none we know (the model often put goal ids there)."""
        find = lambda tag: (hash_utils.find(tag, signal_hash) or '').strip()
        signal_type, desc, signal_text = find('type'), find('description'), find('signal')
        if not desc or any(e in desc or e in signal_text for e in self._TEMPLATE_ECHOES):
            return None
        importance, urgency = self._unit(find('importance')), self._unit(find('urgency'))
        if importance is None or urgency is None:
            return None
        surprise = self._unit(find('surprise'), default=0.2)

        signal_drives = [Drive._instances[i] for i in (d.strip() for d in find('drive_ids').split('@'))
                         if i in Drive._instances]
        if not signal_drives:
            signal_drives = list(drives or [])
        activation = 0.0
        for drive in signal_drives:
            activation = noisy_add(activation, min(1.0, max(0.0, drive.activation)))

        return DriveSignal(
            text=f'{signal_text}: {desc}',
            drives=signal_drives,
            type=signal_type.lower() or 'issue',
            importance=activation*importance,  # raw importance weighted by the drives' activation
            urgency=urgency,
            surprise=surprise,
            timestamp=current_time,
            embedding=self._get_embedding(desc)
        )

    def check_drive_signal(self, drive: Drive) -> List[DriveSignal]:
        """Analyze drive for self-awareness signals"""

        mission = f"""Given the information following, evaluate the awareness of the owner at the current time with respect to the following need: 
        
{drive.id}: {drive.text}

"""

        suffix = f"""

Report any issues or opportunities you expect the owner is aware of with respect to this need and only this need:

{drive.id}: {drive.text}

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
The valid tags in this response are signal, type, description, drive_ids, importance, urgency, and surprise.
The type tag takes a single work as its content, either issue or opportunity.
be careful to insert line breaks only where shown, separating a value from the next tag, as in the following example:

#signal 3-4 words briefly naming the key theme or essence (e.g., "Food Source Discovered")
#type issue / opportunity / surprise
#description 4-7 words explicitly identifying or elaborating the specific detail or actionable aspect of the signal (e.g., "Apple trees nearby provide food").
#drive_ids {drive.id}
#importance 0.0-1.0
#urgency 0.0-1.0
#surprise 0.0-1.0
##

Only respond if you find a clear and strong signal. Multiple signals can be on separate lines.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</end>
"""
            
        try:
            response = self.ask(self.owner, prefix=mission, suffix=suffix, addl_bindings= {}, tag = 'DriveSignal.check_drive_signal', max_tokens=300)
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
                # values are clamped to [0, 1] in construct_signal, so activation stays in [0, 1]
                drive.activation = .5*drive.activation + .5*max(raw_signal_importance, signal.urgency)
                print(f'    {signal.type} {signal.text} ({signal.importance:.2f}, {signal.urgency:.2f})')
                signals.append(signal)
        self.last_signals[drive.id] = signals
        return signals

    def check_drive_signals(self):
        """Check every drive (updates drive.activation). One failing check no longer aborts the
        scene setup that calls this."""
        import sim.ablation as ablation
        if not ablation.enabled('drive_checks'):
            return
        for drive in self.owner.drives:
            try:
                self.check_drive_signal(drive)
            except Exception as e:
                print(f'{self.owner.name} drive check failed for {drive.id}: {e}')
                traceback.print_exc()
        from sim.trace import trace
        trace.event('drives', character=self.owner.name,
                    activations={d.text: round(d.activation, 3) for d in self.owner.drives},
                    signals=[{'drive': d.text if d else None, 'text': sg.text, 'type': sg.type,
                              'importance': round(sg.importance, 3), 'urgency': round(sg.urgency, 3)}
                             for d, sg in self.recent_signals()])

    def recent_signals(self):
        """(drive, signal) pairs from each drive's latest check, for the Explore panel."""
        return [(Drive._instances.get(drive_id), s) for drive_id, sigs in self.last_signals.items() for s in sigs]
