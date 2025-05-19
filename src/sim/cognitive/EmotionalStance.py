from __future__ import annotations
import os, sys, re, traceback, requests, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ast import List
from enum import Enum
from typing import Any
from utils import hash_utils
from utils.Messages import UserMessage
import utils.xml_utils as xml
from sim.cognitive.driveSignal import Drive, SignalCluster
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking

class Arousal(Enum):
    Vigilant = "Vigilant" #vigilant, ready, focused"
    Anticipatory = "Anticipatory" #expectant, preparing"
    Agitated = "Agitated" #restless, unsettled"
    Relaxed = "Relaxed" #calm, at ease"
    Exhausted = "Exhausted" #depleted, drained"
    Compelled = "Compelled" #biologically / unconsciously motivated - includes sexual arousal, hunger, etc."
    Neutral = "Neutral" #neutral, indifferent"
    
class Tone(Enum):
    Angry = "Angry" #hostile, enraged"
    Fearful = "Fearful" #threatened, scared"
    Anxious = "Anxious" #worried, uneasy"
    Sad = "Sad" #sorrowful, grieving"
    Disgusted = "Disgusted" #revolted, repulsed, contemptuous"
    Confident = "Confident" #confident, assured"
    Surprised = "Surprised" #astonished, startled"
    Curious = "Curious" #curious, engaged"
    Joyful = "Joyful" #happy, elated"
    Content = "Content" #satisfied, peaceful"
    Neutral = "Neutral" #neutral, indifferent"

class Orientation(Enum):
    Controlling = "Controlling" #directing, managing others"
    Challenging = "Challenging" #testing, confronting others"
    Appeasing = "Appeasing" #placating, avoiding conflict"
    Avoiding = "Avoiding" #minimizing interaction"
    Supportive = "Supportive" #assisting others' goals"
    Seekingsupport = "Seekingsupport" #requesting assistance"
    Connecting = "Connecting" #building/strengthening relationships"
    Performing = "Performing" #seeking attention/approval"
    Observing = "Observing" #gathering social information"
    Defending = "Defending" #protecting position/resources"
    Neutral = "Neutral" #neutral, indifferent"

stance_definitions = {
    "Vigilant": "vigilant, ready, focused",
    "Anticipatory": "expectant, preparing",
    "Agitated": "restless, unsettled",
    "Relaxed": "calm, at ease",
    "Exhausted": "depleted, drained",
    "Compelled": "biologically / unconsciously motivated - includes sexual arousal, hunger, etc.",
    "Angry": "hostile, enraged",
    "Fearful": "threatened, scared",
    "Anxious": "worried, uneasy",
    "Sad": "sorrowful, grieving",
    "Disgusted": "revolted, repulsed, contemptuous",
    "Confident": "confident, assured",
    "Surprised": "astonished, startled",
    "Curious": "curious, engaged",
    "Joyful": "happy, elated",
    "Content": "satisfied, peaceful",
    "Controlling": "directing, managing others",
    "Challenging": "testing, confronting others",   
    "Appeasing": "placating, avoiding conflict",    
    "Avoiding": "minimizing interaction",
    "Supportive": "assisting others' goals",
    "Seekingsupport": "requesting assistance",
    "Connecting": "building/strengthening relationships",
    "Performing": "seeking attention/approval",
    "Observing": "gathering social information",
    "Defending": "protecting position/resources",
    "Neutral": "neutral, indifferent",
}
class EmotionalStance:
    """Represents an emotional stance of a character"""
    def __init__(self, arousal: Arousal=Arousal.Relaxed, tone: Tone=Tone.Content, orientation: Orientation=Orientation.Connecting):
        self.arousal = arousal
        self.tone = tone
        self.orientation = orientation

    def to_definition(self):
        return f'{str(self.arousal.value)}, {str(self.tone.value)}, {str(self.orientation.value)}'

    def to_definition(self):
        return f'{str(self.arousal.value)} ({stance_definitions[self.arousal.value]}), {str(self.tone.value)} ({stance_definitions[self.tone.value]}), {str(self.orientation.value)} ({stance_definitions[self.orientation.value]})'

    @staticmethod
    def from_signalCluster(signalCluster: SignalCluster, character: Character):
        prompt = [UserMessage(content="""You are an expert in emotional tone and orientation.
You are given a signal cluster that represents the interaction between the basic drives of a character and recent events.
Your task is to extract the arousal, tone and orientation of the character resulting from the signal cluster and the character's awareness of it.
 
Signal Cluster:
{{$signalCluster}}

Character:
{{$character}} 

There are three dimensions to your response:

1. Arousal: The arousal of the character. This takes values from the following list:
    Vigilant
    Anticipatory
    Agitated
    Relaxed
    Exhausted
    Compelled
    Neutral

2. Tone: The tone of the character resulting from the signal cluster and the character's awareness of it.
    Angry
    Fearful
    Anxious
    Sad
    Disgusted
    Surprised
    Curious
    Joyful
    Content
    Neutral

3. Orientation: The orientation of the character resulting from the signal cluster and the character's awareness of it.
    Controlling
    Challenging
    Appeasing
    Avoiding
    Supportive
    Seekingsupport
    Connecting
    Performing
    Observing
    Defending
    Neutral

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#arousal arousal
#tone tone
#orientation orientation
##

Respond only with the hash-formatted text, nothing else.
End your response with the:
<end/>
""")]
        
        response = character.llm.ask({"signalCluster": signalCluster.to_string(), 
                                      "character": character.character
                                      }, prompt, tag='EmotionalStance.from_signalCluster', stops=["<end/>"], max_tokens = 100)
        response = hash_utils.clean(response)

        tone = Tone.Neutral
        orientation = Orientation.Neutral
        response_arousal = hash_utils.find('arousal', response)
        response_arousal = response_arousal.strip().capitalize()
        if response_arousal in Arousal.__members__:
            arousal = Arousal(response_arousal)
        else:
            arousal = Arousal.Neutral
        response_tone = hash_utils.find('tone', response)
        response_tone = response_tone.strip().capitalize()
        if response_tone in Tone.__members__:
            tone = Tone(response_tone)
        else:
            tone = Tone.Neutral

        response_orientation = hash_utils.find('orientation', response)
        response_orientation = response_orientation.strip().capitalize()
        response_orientation = response_orientation.replace(' ', '') # Seeking support -> Seekingsupport
        if response_orientation in Orientation.__members__:
            orientation = Orientation(response_orientation)
        else:
            orientation = Orientation.Neutral
        return EmotionalStance(arousal, tone, orientation)

    def to_string(self):
        return "Stance:{"+str(self.arousal)+'+, '+str(self.tone)+', '+str(self.orientation)+'}'

    def from_signalClusters(signalClusters: List[SignalCluster], character: Character):
        prompt = [UserMessage(content="""You are an expert in emotional tone and orientation.
You are given a set of signalClusters that represents the interaction between the basic drives of a character and recent events.
Your task is to extract the arousal, tone and orientation of the character resulting from the signalClusters and the character's awareness of it.
Note that each signalCluster has a score. The score is a measure of the importance of the signalCluster to the character.
You should seek to identify the dominant arousal, tone, and orientation of the character. Do not merely average the values implied by various signalClusters.
Rather, imagine a quasi-stable emotional equilibrium that is being perturbed by the signalClusters, and is more or less labile and volatile according to the character.
 
Signal Clusters:
{{$signalClusters}}

Character:
{{$character}} 

There are three dimensions to your response:

1. Arousal: The arousal of the character. This takes values from the following list:
    Vigilant
    Anticipatory
    Agitated
    Relaxed
    Exhausted
    Compelled
    Neutral
                              
2. Tone: The tone of the character resulting from the signal cluster and the character's awareness of it.
    Angry
    Fearful
    Anxious
    Sad
    Disgusted
    Surprised
    Curious
    Joyful
    Content

3. Orientation: The orientation of the character resulting from the signal cluster and the character's awareness of it.
    Controlling
    Challenging
    Appeasing
    Avoiding
    Supportive
    Seekingsupport
    Connecting
    Performing
    Observing
    Defending

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
be careful to insert line breaks only where shown, separating a value from the next tag:

#arousal arousal
#tone tone
#orientation orientation
##

Respond only with the hash-formatted text, nothing else.
End your response with the:
<end/>
""")]
        
        response = character.llm.ask({"signalClusters": '\n'.join([signalCluster.to_string() for signalCluster in signalClusters]), 
                                      "character": character.character
                                      }, prompt, tag='EmotionalStance.from_signalClusters', stops=["<end/>"], max_tokens = 100)
        response = hash_utils.clean(response)

        response_arousal = hash_utils.find('arousal', response)
        response_arousal = response_arousal.strip().capitalize()
        if response_arousal in Arousal.__members__:
            arousal = Arousal(response_arousal)
        else:
            arousal = Arousal.Neutral
        response_tone = hash_utils.find('tone', response)
        response_tone = response_tone.strip().capitalize()
        if response_tone in Tone.__members__:
            tone = Tone(response_tone)
        else:
            tone = Tone.Neutral
        response_orientation = hash_utils.find('orientation', response)
        response_orientation = response_orientation.strip().capitalize()
        response_orientation = response_orientation.replace(' ', '') # Seeking support -> Seekingsupport
        if response_orientation in Orientation.__members__:
            orientation = Orientation(response_orientation)
        else:
            orientation = Orientation.Neutral
        return EmotionalStance(arousal, tone, orientation)
