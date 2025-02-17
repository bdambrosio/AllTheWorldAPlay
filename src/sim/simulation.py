import random
from typing import Optional

import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# Add parent directory to path to access existing simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.agh import Agh
from sim.context import Context
from utils.llm_api import LLM, generate_image
import base64
from sim.human import Human  # Add this import
import asyncio
from collections import deque
from typing import Dict, Any
import logging
import traceback  # Add at top if not already imported
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SimulationWrapper:
    """Wrapper for existing simulation engine"""
    
    def __init__(self, play_path=None):
        """Initialize new simulation instance"""
        if play_path is None:
            raise ValueError("Play path is required")
        import importlib.util
        spec = importlib.util.spec_from_file_location("play_module", play_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
            
        if not hasattr(module, 'W'):
            raise ValueError("Play file must define a 'W' variable")
        if hasattr(module, 'server'):
            server = module.server
        else:
            server = 'deepseek'
        if hasattr(module, 'world_name'):
            world_name = module.world_name
        else:
            world_name = 'Lost'
            
        self.context = module.W
        self.simulation = Simulation(self.context, server, world_name)
        self.simulation.initialized = True
        self.simulation.running = False
        self.simulation.paused = False
        
    def process_command(self, command: str) -> str:
        """Process command and return response"""
        return self.simulation.process_command(command)

class Simulation:
    """Wrapper for existing simulation engine"""
    
    def __init__(self, context, server='local', world_name='Lost'):
        """Initialize with existing context from scenario"""
        self.context = context
        self.server = server
        self.world_name = world_name
        self.initialized = False
        self.steps_since_last_update = 0
        self.running = False
        self.paused = False
        self.watcher = None  # Add watcher storage
        self.llm = LLM(server)
        self.context.set_llm(self.llm)
        for actor in self.context.actors:
           actor.set_llm(self.llm)
        for actor in self.context.actors:
           print(f'calling {actor.name} initialize')
           actor.initialize(self)
        for actor in self.context.actors:
            print(f'calling {actor.name} greet')
            #actor.greet()
            actor.see()
        self.inject_queue = asyncio.Queue()  # Queue for injects
        self.is_running = False
        self.is_paused = False
        self.websocket = None
        self.current_step = 0
        self.connection_active = False

    def process_command(self, command: str) -> str:
        """Process command and return response"""
        try:
            response = self.context.process_command(command)
            return response if response else "Command processed"
        except Exception as e:
            return f"Error: {str(e)}"
            
    async def step(self, char_update_callback=None, world_update_callback=None, log_callback=None):
        """Perform one simulation step with optional character update callback"""
        try:
            # Process any queued injects before the step
            while not self.inject_queue.empty():
                inject = await self.inject_queue.get()
                await self._process_inject(inject)

            if self.initialized:
                task, chars = self.context.next_act()
                # Process character step
                for char in chars:
                    if char:
                        print(f'{char.name} cognitive cycle')   
                        char.cognitive_cycle()
                        break # only execute for first available actor

                for actor in self.context.actors:
                    actor_data = actor.to_json()
                    try:
                        description = actor.generate_image_description()
                        if description:
                            image_path = generate_image(self.context.llm, description, filepath=actor.name+'.png')
                            with open(image_path, 'rb') as f:
                                image_data = base64.b64encode(f.read()).decode()
                                actor_data['image'] = image_data
                                if char_update_callback:
                                    await char_update_callback(self.context, actor.name, actor_data)
                                    await asyncio.sleep(0.1)
                    except Exception as e:
                        print(f"Error generating image for {actor.name}: {e}")
                        
                #now handle context
                if self.steps_since_last_update > random.randint(3, 5):    
                    if log_callback:
                        await log_callback('*** updating world ***')
                        await asyncio.sleep(0.1)
                    self.context.senses('')
                    if world_update_callback:
                        context_data = self.context.to_json()
                        await world_update_callback('World', context_data)
                        await asyncio.sleep(0.1)
                    self.steps_since_last_update = 0
                else:
                    self.steps_since_last_update += 1

                # Yield control briefly to allow other tasks
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.is_running = False
            raise  # Re-raise to maintain existing error handling
            
    def run(self):
        """Start continuous simulation"""
        self.running = True
        self.paused = False
        
    def pause(self):
        """Pause running simulation"""
        self.paused = True
        
    def stop(self):
        """Stop simulation completely"""
        self.running = False
        self.paused = False
        
    def save_world(self, filename=None):
        """Save current world state"""
        return self.context.save_world(filename)
        
    def load_world(self, filename):
        """Load saved world state"""
        try:
            return self.context.load_world(filename)
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.is_running = False
            raise  # Re-raise to maintain existing error handling
         
    async def inject(self, target_name, text, update_callback=None):
        """Route inject command using watcher pattern from worldsim"""
        try:
            if self.watcher is None:
                self.watcher = Human('Watcher', "Human user representative", None)
                self.watcher.set_context(self.context)
                self.context.actors.append(self.watcher)
            if target_name == 'World':
                # Stub for now
                return
            # Format message as "character-name, message" like worldsim
            formatted_message = f"{target_name}, {text}"
            self.watcher.inject(formatted_message)
            target = self.context.get_actor_by_name(target_name)
            if update_callback and target:
                await update_callback(target_name, target.to_json())
                await asyncio.sleep(0.1)
                target.show = ''
        except Exception as e:
            logger.error(f"Error in simulation step: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            self.is_running = False
            raise  # Re-raise to maintain existing error handling
        
    def get_character_status(self):
        """Get current character states"""
        return {
            'characters': self.characters,
            'running': self.running,
            'paused': self.paused
        }
        
    def get_history(self):
        """Get simulation history"""
        return self.context.history if hasattr(self.context, 'history') else []
        
    @property
    def characters(self):
        """Access actors through context"""
        return self.context.actors if self.initialized else []

    async def set_websocket(self, websocket):
        self.websocket = websocket
        self.connection_active = True
        
    async def handle_inject(self, target: str, text: str):
        await self.inject_queue.put({
            'target': target,
            'text': text
        })
        
            
    async def handle_disconnect(self):
        logger.info("Client disconnected, cleaning up...")
        self.connection_active = False
        self.websocket = None
        # Optionally pause simulation
        self.is_paused = True

    def get_character_details(self, char_name):
        """Get detailed character state"""
        char = self.context.get_actor_by_name(char_name)
        return char.get_explorer_state()

# Add a simple test character class
class TestCharacter:
    def __init__(self, name):
        self.name = name
        self.description = f"A person named {name} in a forest setting"  # Add description
        self.priorities = [
            Priority("Exploring", "high"),
            Priority("Survival", "medium")
        ]

class Priority:
    def __init__(self, name, value):
        self.name = name
        self.value = value
