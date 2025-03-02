from __future__ import annotations
from enum import Enum
from typing import Any
from src.utils.Messages import UserMessage
import utils.xml_utils as xml
from src.sim.cognitive.driveSignal import Drive, SignalCluster
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.context import Context  # Only imported during type checking
    from sim.agh import Character
    from sim.memory.core import NarrativeSummary

class ScreenWriter:
    def __init__(self, context: Context):
        self.context = context

    def set_next_scene(self, message: str) -> str:
        prompt=[UserMessage(content="""
        You are the director and screenwriter for an improve play. Your task is to set the next scene to move the story forward.

       The current scene is:
        {{$current_scene}}

        The previous scenes have been:
        {{$previous_scenes}} 
                            

        """), message]
        return message  

