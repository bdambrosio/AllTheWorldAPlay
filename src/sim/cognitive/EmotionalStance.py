from __future__ import annotations
from enum import Enum
from typing import Any
from src.utils.Messages import UserMessage
import utils.xml_utils as xml
from src.sim.cognitive.driveSignal import Drive, SignalCluster
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking

class Arousal(Enum):
    Alert = "Alert" #vigilant, ready, focused"
    Anticipatory = "Anticipatory" #expectant, preparing"
    Agitated = "Agitated" #restless, unsettled"
    Relaxed = "Relaxed" #calm, at ease"
    Exhausted = "Exhausted" #depleted, drained"
    Compelled = "Compelled" #biologically / unconsciously motivated - includes sexual arousal, hunger, etc."
    
    
class Tone(Enum):
    Angry = "Angry" #hostile, enraged"
    Fearful = "Fearful" #threatened, scared"
    Anxious = "Anxious" #worried, uneasy"
    Sad = "Sad" #sorrowful, grieving"
    Disgusted = "Disgusted" #revolted, repulsed, contemptuous"
    Surprised = "Surprised" #astonished, startled"
    Curious = "Curious" #curious, engaged"
    Joyful = "Joyful" #happy, elated"
    Content = "Content" #satisfied, peaceful"

class Orientation(Enum):
    Controlling = "Controlling" #directing, managing others"
    Challenging = "Challenging" #testing, confronting others"
    Appeasing = "Appeasing" #placating, avoiding conflict"
    Avoiding = "Avoiding" #minimizing interaction"
    Supportive = "Supportive" #assisting others' goals"
    SeekingSupport = "SeekingSupport" #requesting assistance"
    Connecting = "Connecting" #building/strengthening relationships"
    Performing = "Performing" #seeking attention/approval"
    Observing = "Observing" #gathering social information"
    Defending = "Defending" #protecting position/resources"

stance_definitions = {
    "Alert": "vigilant, ready, focused",
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
    "Surprised": "astonished, startled",
    "Curious": "curious, engaged",
    "Joyful": "happy, elated",
    "Content": "satisfied, peaceful",
    "Controlling": "directing, managing others",
    "Challenging": "testing, confronting others",   
    "Appeasing": "placating, avoiding conflict",    
    "Avoiding": "minimizing interaction",
    "Supportive": "assisting others' goals",
    "SeekingSupport": "requesting assistance",
    "Connecting": "building/strengthening relationships",
    "Performing": "seeking attention/approval",
    "Observing": "gathering social information",
    "Defending": "protecting position/resources",
}
class EmotionalStance:
    """Represents an emotional stance of a character"""
    def __init__(self, arousal: Arousal, tone: Tone, orientation: Orientation):
        self.arousal = arousal
        self.tone = tone
        self.orientation = orientation

    def to_definition(self):
        return f'{str(self.arousal.value)} ({stance_definitions[self.arousal.value]}), {str(self.tone.value)} ({stance_definitions[self.tone.value]}), {str(self.orientation.value)} ({stance_definitions[self.orientation.value]})'

    @staticmethod
    def from_signalCluster(signalCluster: SignalCluster, character: Character):
        prompt = [UserMessage(content=f"""You are an expert in emotional tone and orientation.
You are given a signal cluster that represents the interaction between the basic drives of a character and recent events.
Your task is to extract the arousal, tone and orientation of the character resulting from the signal cluster and the character's awareness of it.
 
Signal Cluster:
{{$signalCluster}}

Character:
{{$character}} 

There are three dimensions to your response:

1. Arousal: The arousal of the character. This takes values from the following list:
    Alert = "vigilant, ready, focused"
    Anticipatory = "expectant, preparing"
    Agitated = "restless, unsettled"
    Relaxed = "calm, at ease"
    Exhausted = "depleted, drained"
    Compelled = "biologically / unconsciously motivated - includes sexual arousal, hunger, etc."

2. Tone: The tone of the character resulting from the signal cluster and the character's awareness of it.
    Angry = "hostile, enraged"
    Fearful = "threatened, scared"
    Anxious = "worried, uneasy"
    Sad = "sorrowful, grieving"
    Disgusted = "revolted, repulsed, contemptuous"
    Surprised = "astonished, startled"
    Curious = "curious, engaged"
    Joyful = "happy, elated"
    Content = "satisfied, peaceful"

3. Orientation: The orientation of the character resulting from the signal cluster and the character's awareness of it.
    Controlling = "directing, managing others"
    Challenging = "testing, confronting others"
    Appeasing = "placating, avoiding conflict"
    Avoiding = "minimizing interaction"
    Supportive = "assisting others' goals"
    SeekingSupport = "requesting assistance"
    Connecting = "building/strengthening relationships"
    Performing = "seeking attention/approval"
    Observing = "gathering social information"
    Defending = "protecting position/resources"

Your response should be in XML format as follows:

<EmotionalStance>
    <Arousal> Alert / Anticipatory / Agitated =/ Relaxed / Exhausted / Compelled </Arousal>
    <Tone>Angry / Fearful / Anxious / Sad / Disgusted / Surprised / Curious / Joyful / Content</Tone>
    <Orientation>Controlling / Challenging / Appeasing / Avoiding / Supportive / SeekingSupport / Connecting / Performing / Observing / Defending</Orientation>
</EmotionalStance>

Respond only with the XML format, nothing else.
End your response with the:
<end/>
""")]
        
        response = character.llm.ask({"signalCluster": signalCluster.to_string(), 
                                      "character": character.character
                                      }, prompt, stops=["<end/>"], max_tokens = 100)
        if xml.find('<Arousal>', response):
            try:
                arousal = Arousal(xml.find('<Arousal>', response).strip().capitalize())
            except Exception as e:
                arousal = Arousal.Relaxed
        if xml.find('<Tone>', response):
            try:
                tone = Tone(xml.find('<Tone>', response).strip().capitalize())   
            except Exception as e:
                tone = Tone.Content
        if xml.find('<Orientation>', response):
            try:
                orientation = Orientation(xml.find('<Orientation>', response).strip().capitalize())
            except Exception as e:
                orientation = Orientation.Connecting
        return EmotionalStance(arousal, tone, orientation)

    def to_string(self):
        return "Stance:{"+str(self.arousal)+'+, '+str(self.tone)+', '+str(self.orientation)+'}'
