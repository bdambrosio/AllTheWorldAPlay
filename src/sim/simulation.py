from typing import Optional

import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# Add parent directory to path to access existing simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.agh import Agh
from sim.context import Context
from utils.llm_api import LLM, generate_image
import base64

class SimulationWrapper:
    """Wrapper for existing simulation engine"""
    
    def __init__(self):
        """Initialize new simulation instance"""
        # Create scenario like worldsim does
        S = Agh("Samantha", """You are a pretty young Sicilian woman...""")
        S.set_drives([
            "solving the mystery of how they ended up in the forest with no memory.",
            "love and belonging, including home, acceptance, friendship, trust, intimacy.",
            "immediate physiological needs: survival, shelter, water, food, rest."
        ])
        S.add_to_history("You think This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?")

        J = Agh("Joe", """You are a young Sicilian male...""")
        J.set_drives([
            "communication and coordination with Samantha, gaining Samantha's trust.",
            "solving the mystery of how they ended up in the forest with no memory.",
            "immediate physiological needs: survival, shelter, water, food, rest."
        ])
        J.add_to_history("You think Ugh. Where am I?. How did I get here? Why can't I remember anything? Who is this woman?")
        J.add_to_history("You think Whoever she is, she is pretty!")

        W = Context([S, J], """A temperate, mixed forest-open landscape with no buildings, roads, or other signs of humananity...""")
        
        self.simulation = Simulation(W, server='deepseek', world_name='Lost')
        
    def process_command(self, command: str) -> str:
        """Process command and return response"""
        return self.simulation.process_command(command)

class Simulation:
    """Wrapper for existing simulation engine"""
    
    def __init__(self, context, server='deepseek', world_name='Lost'):
        """Initialize with existing context from scenario"""
        self.context = context
        self.server = server
        self.world_name = world_name
        self.initialized = True
        self.steps_since_last_update = 0
        self.running = False
        self.paused = False
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


    def process_command(self, command: str) -> str:
        """Process command and return response"""
        try:
            response = self.context.process_command(command)
            return response if response else "Command processed"
        except Exception as e:
            return f"Error: {str(e)}"
            
    async def step(self, char_update_callback=None):
        """Perform one simulation step with optional character update callback"""
        if self.initialized:
            for char in self.characters:
                # Process character step
                char.senses()
                
                # Generate and send image update
                try:
                    description = char.generate_image_description()
                    if description:
                        image_path = generate_image(self.context.llm, description)
                        if image_path:
                            with open(image_path, 'rb') as f:
                                image_data = base64.b64encode(f.read()).decode()
                                char_data = char.to_json()
                                char_data['image'] = image_data
                                if char_update_callback:
                                    await char_update_callback(char.name, char_data)
                except Exception as e:
                    print(f"Error generating image for {char.name}: {e}")
                    
                # Send update even if image fails
                if char_update_callback and not char_data.get('image'):
                    char_update_callback(char.name, char.to_json())
            
            #now handle context
            if self.steps_since_last_update > 5:    
                self.context.senses('')
                self.context.image(filepath='worldsim.png')
                if char_update_callback:
                    context_data = self.context.to_json()
                    await char_update_callback('World', context_data)
                self.steps_since_last_update = 0
            else:
                self.steps_since_last_update += 1
            
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
        return self.context.load_world(filename)
        
    def inject(self, text):
        """Inject text into simulation"""
        return self.context.inject(text)
        
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