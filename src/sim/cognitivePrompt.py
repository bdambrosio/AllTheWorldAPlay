from __future__ import annotations
import json
import re
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Act, Character, Goal, Task
    from sim.context import Context  # Only imported during type checking
    from sim.narrativeCharacter import NarrativeCharacter

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Messages import UserMessage, SystemMessage
from sim.cognitive.EmotionalStance import EmotionalStance
from utils import llm_api
import utils.xml_utils as xml
import sim.map as map
import utils.hash_utils as hash_utils
import utils.choice as choice   
#from sim.cognitive.DialogManager import Dialog
from openai import OpenAI

class CognitiveToolInterface:
    def __init__(self, character):
        self.character: NarrativeCharacter = character
        self.pattern = r'\{\$[^}]*\}'
        self.client = None
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        except Exception as e:
            print(f"Error getting OpenAI API key: {e}")
            openai_api_key = None

        self.DefaultEndpoint = 'https://api.openai.com'
        self.openai_api_base = self.DefaultEndpoint
        self.UserAgent = 'Owl'
        self.model_name = 'gpt-4.1-mini'
        self.client = OpenAI(
            api_key=openai_api_key,
            timeout=45.0,  # 60 second timeout
            max_retries=2   # Retry up to 3 times
        )

        self.tool_registry = {
            'characterModel': self.get_characterModel,
            'driveSignals': self.get_driveSignals,
            'recentMemories': self.get_recentMemories,
            #'locationMemories': self.get_location_memories,
            'emotionalContext': self.get_emotionalContext,
            'narrativeContext': self.get_narrativeContext,
            'goalHistory': self.get_goalHistory,
            'situationalSignals': self.get_situationalSignals,
            #'resourceAvailability': self.get_resource_availability,    
            #'actorPresence': self.get_actor_presence,
        }

    
    def call_tool(self, tool_name: str, **kwargs):
        """Route tool calls to appropriate methods without caching since most tool calls are simple internal retrievals"""
        result = self.tool_registry[tool_name](**kwargs)
        return result    
    
    def make_tool(self, name, description, parameters):
        return {"type": "function",
            "function": {"name": name, "description": description, "parameters": parameters}
        }

    def get_tool_definitions(self):
        return [
            self.make_tool(
                "characterModel",
                "Get information about a specific character...",
                {
                    "type": "object",
                    "properties": {
                        "character_name": {"type": "string", "description": "Name of the character"}
                    },
                    "required": ["character_name"]
                }
            ),
            self.make_tool(
                "driveSignals",
                "Get drive-related information triggered by recent perceptions, optionally filtered by drive_id",
                {
                    "type": "object",
                    "properties": {
                        "drive_id": {"type": "string", "description": "The id of the drive to get signals for"}
                    },
                    "required": ["drive_id"]
                }
            ),
            self.make_tool(
                "recentMemories",
                "Get <count> most recent memories (records of perceptions, actions, and interactions)",
                {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "description": "The number of memories to get"}
                    },
                    "required": ["count"]
                }
            ),
            #self.make_tool(
            #    "locationMemories",
            #    "Get self memories and knowledge related to a specific location",
            #    {
            #        "type": "object",
            #        "properties": {
            #            "location": {"type": "string", "description": "The location to get memories for"}
            #        },
            #        "required": ["location"]
            #    }
            #),
            self.make_tool(
                "emotionalContext",
                "Get current emotional stance of the character",
                {
                    "type": "object",
                    "properties": {},
                }
            ),
            self.make_tool(
                "narrativeContext",
                "Get a text summary of the character's current situation and recent activities",
                {
                    "type": "object",
                    "properties": {},
                }
            ),
            self.make_tool(
                "goalHistory",
                "Get the character's recent goal achievements and failures",
                {
                    "type": "object",
                    "properties": {
                        "include_failed": {"type": "boolean", "description": "Whether to include failed goals"}
                    },
                    "required": ["include_failed"]
                }
            ),
            self.make_tool(
                "situationalSignals",
                "Get environmental/social signals relevant to current situation",
                {
                    "type": "object",
                    "properties": {}
                }
            )
            #self.make_tool(
            #    "resourceAvailability",
            #    "Check available resources, tools, or opportunities",
            #    {
            #        "type": "object",
            #        "properties": {
            #            "resource_type": {"type": "string", "description": "The type of resource to check availability for"}
            #        },
            #        "required": ["resource_type"]
            #    }
            #),
            #self.make_tool(
            #    "actorPresence",
            #    "Get information about who is present and their apparent states",
            #    {
            #        "type": "object",
            #        "properties": {
            #            "location": {"type": "string", "description": "The location to get actor presence for"}
            #        },
            #        "required": ["location"]
            #    }
            #)
        ]

    def get_characterModel(self, character_name: str):
        """Get information about a specific character including relationship, recent interactions, and dialog history"""
        return self.character.actor_models.get_actor_model(character_name, create_if_missing=True).get_relationship()
    
    def get_driveSignals(self, drive_id: str = None):
        """Get drive-related information triggered by recent perceptions, optionally filtered by drive_id"""
        primary_drive_signals = self.character.driveSignalManager.get_signals_for_drive(drive_id=drive_id)
        return "\n".join([f'{sc.id}: {sc.text}' for sc in primary_drive_signals])
    
    def get_recentMemories(self,count: int = 8, topic_filter: str = None):
        """Get recent memories, optionally filtered by topic/relevance"""
        recent_memories = self.character.structured_memory.get_recent(count)
        return '\n'.join(memory.text for memory in recent_memories) if recent_memories else "No recent memories"

    def get_locationMemories(self,location: str):
        """Get memories and knowledge about specific location"""
        return self.character.structured_memory.get_location_memories(location)

    def get_emotionalContext(self):
        """Get current emotional stance and recent emotional changes"""
        return self.character.emotionalStance.to_string() if self.character.emotionalStance else "No emotional context"

    def get_narrativeContext(self):
        """Get narrative context: 'current', 'recent', or 'full'"""
        return self.character.narrative.get_summary()

    def get_goalHistory(self,include_failed: bool = False):
        """Get recent goal achievements and failures"""
        return '\n'.join([g.short_string() for g in self.character.goal_history]) if self.character.goal_history else "No goal history"

    def get_situationalSignals(self):
        """Get environmental/social signals relevant to current situation"""    
        ranked_signalClusters = self.character.driveSignalManager.get_scored_clusters()
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order
        return "\n".join([sc.to_string() for sc in focus_signalClusters])

    def get_resourceAvailability(self,resource_type: str = None):
        """Check available resources, tools, or opportunities"""
        return self.character.structured_memory.get_resource_availability(resource_type)

    def get_actorPresence(self,location: str = None):
        """Get information about who is present and their apparent states"""
        return self.character.structured_memory.get_actor_presence(location)


    def substitute_bindings(self, prompt, bindings):
        """Substitute bindings in a prompt"""
        if bindings is None or len(bindings) == 0:
           return prompt
        else:
            substituted_prompt = ''
            matches = re.findall(self.pattern, prompt)
            new_content = prompt
            for match in matches:
                var = match[2:-1]
                if var in bindings.keys():
                    new_content = new_content.replace('{{$'+var+'}}', str(bindings[var]))
                else:
                    print(f' var not in bindings {var} {bindings}')
                    raise ValueError(f'unbound prompt variable {var}')
                substituted_prompt = new_content
            return substituted_prompt


    def cognitive_ask(self, character, system_prompt=None, prefix=None, suffix=None, 
                 addl_bindings={}, max_tokens=100, tag='', max_reasoning_turns=3, log=False):
        """
        Replacement for default_ask - uses tool-driven reasoning instead of front-loading all information in prompts
        """
        # Build the task from the original prompt components
        task_parts = []
        if prefix:
            task_parts.append(prefix)
        if suffix:
            task_parts.append(suffix)
    
        task = "\n".join(task_parts)
        task = self.substitute_bindings(task, addl_bindings)
        print(f"Task: {task}")
    
        # Use cognitive reasoning loop instead of massive prompt
        return self.cognitive_reasoning_loop(
            character=self.character,
            system_prompt=system_prompt,
            task=task,
            addl_bindings=addl_bindings,
            max_turns=max_reasoning_turns,
            max_tokens=max_tokens,
            tag=tag
        )

    def cognitive_reasoning_loop(self, character, system_prompt, task, addl_bindings, 
                           max_turns=3, max_tokens=100, tag=''):
        """
        Internal implementation that handles multi-turn tool-based reasoning
        """    
        # Minimal base context instead of the massive prompt from original default_ask
        base_context = f"""
        Character: {self.character.name + ' ' + self.character.character}
        Current Situation: {self.character.context.current_state}
        Current Location: {self.character.look_percept}
        """
    
        conversation = [
            {"role": "system", "content": system_prompt or "You are a cognitive actor with access to information tools."},
            {"role": "user", "content": f"{base_context}\n\nTask: {task}"}
        ]
    
        for turn in range(max_turns):
            response = self.client.chat.completions.create(
                model = 'gpt-4.1-mini-2025-04-14',
                messages = conversation, 
                tools=self.get_tool_definitions(),
                tool_choice="auto",
                max_tokens=max_tokens,
            )
        
            choice = response.choices[0]
            # ‼️ Always stash the assistant message itself first
            conversation.append(choice.message.model_dump())   # or dict(choice.message)

            # Did the model ask to call one or more tools?
            if choice.finish_reason == "tool_calls":
                for call in choice.message.tool_calls:
                    name = call.function.name                       # e.g. "driveSignals"
                    call_id = call.id
                    args = json.loads(call.function.arguments)      # JSON string → dict
                    print(f"Tool call: {name} with args: {args}")
                    result = self.call_tool(name, **args)

                    conversation.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result
                    })
                continue
            else:
                # Final answer reached
                return response.choices[0].message.content
        else:
            # Final answer reached
            return response.choices[0].message.content
    
        # Fallback if max turns reached
        return "Unable to complete reasoning within turn limit"
    
    def list_available_tools(self,reasoning_tools):
        if reasoning_tools:
            return "\n".join([f"- {tool.name}: {tool.description}" for tool in reasoning_tools])
        return "No tools available"

    def execute_tool(self,tool_name, arguments):
        # Implement tool execution logic here
        return f"Tool {tool_name} executed with arguments: {arguments}"

    def system_prompt(self):
        return """
        You are a cognitive actor with access to information tools.

        Current Situation: {self.character.context.current_state}
        Immediate Surroundings: {self.character.look_percept}
        Active Goal: {self.character.focus_goal}

        Available tools: {self.list_available_tools(reasoning_tools)}
        """
