import os
import json
import importlib.util
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
import random
import sys

@dataclass
class Character:
    """Represents a character in the story"""
    name: str
    description: str
    personality: str
    drives: List[str]
    current_thoughts: List[str]
    location: tuple = (0, 0)

@dataclass
class StoryEvent:
    """Represents an event in the story"""
    timestamp: str
    character: str
    action: str
    dialogue: Optional[str] = None
    location: Optional[tuple] = None

class StoryStage:
    """Main orchestrator for generating and running stories"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.characters = []
        self.story_events = []
        self.scenario_config = None
        self.map_config = None
        self.world_state = {}
        
    def load_scenario(self, scenario_file: str):
        """Load scenario configuration from a Python file"""
        try:
            # Load the scenario file as a module
            spec = importlib.util.spec_from_file_location(scenario_file)
            scenario_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scenario_module)
            
            # Extract scenario information
            self.scenario_config = {
                'terrain_types': getattr(scenario_module, 'terrain_types', None),
                'resource_types': getattr(scenario_module, 'resource_types', None),
                'terrain_rules': getattr(scenario_module, 'terrain_rules', {}),
                'resource_rules': getattr(scenario_module, 'resource_rules', {}),
                'infrastructure_rules': getattr(scenario_module, 'infrastructure_rules', {})
            }
            
            print(f"✓ Loaded scenario from {scenario_file}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading scenario: {e}")
            return False
    
    def extract_story_context(self, story_file: str) -> Dict[str, Any]:
        """Extract story context and character setups from the story file"""
        try:
            with open(story_file, 'r') as f:
                content = f.read()
            
            # Parse the story file to extract context
            context = {
                'setting': '',
                'initial_situation': '',
                'character_hints': []
            }
            
            # Look for context creation
            if 'Context(' in content:
                start = content.find('Context(')
                end = content.find(')', start)
                if end != -1:
                    context_str = content[start:end+1]
                    # Extract the description (third parameter)
                    parts = context_str.split(',')
                    if len(parts) >= 3:
                        desc_start = context_str.find('"""')
                        if desc_start != -1:
                            desc_end = context_str.find('"""', desc_start + 3)
                            if desc_end != -1:
                                context['setting'] = context_str[desc_start+3:desc_end].strip()
            
            # Look for character thoughts/internal states
            lines = content.split('\n')
            for line in lines:
                if 'add_perceptual_input' in line and 'internal' in line:
                    # Extract the thought
                    start = line.find('"')
                    end = line.rfind('"', 0, line.rfind('internal'))
                    if start != -1 and end != -1 and start < end:
                        thought = line[start+1:end]
                        context['character_hints'].append(thought)
            
            return context
            
        except Exception as e:
            print(f"Error extracting story context: {e}")
            return {'setting': '', 'initial_situation': '', 'character_hints': []}
    
    def generate_characters(self, story_file: str, num_characters: int = 2) -> List[Character]:
        """Use LLM to generate characters based on scenario and story context"""
        
        story_context = self.extract_story_context(story_file)
        
        # Create terrain description from scenario
        terrain_desc = "unknown terrain"
        if self.scenario_config and 'terrain_rules' in self.scenario_config:
            terrain_rules = self.scenario_config['terrain_rules']
            if 'lowland_distribution' in terrain_rules:
                terrains = list(terrain_rules['lowland_distribution'].keys())
                terrain_desc = f"a landscape with {', '.join(terrains).lower()}"
        
        prompt = f"""
        Create {num_characters} interesting characters for a story set in {terrain_desc}.
        
        Setting: {story_context.get('setting', 'A mysterious forest location')}
        
        Context clues from the story:
        {chr(10).join('- ' + hint for hint in story_context.get('character_hints', []))}
        
        For each character, provide:
        1. Name
        2. Physical description and clothing
        3. Personality traits (3-4 key traits)
        4. Primary motivations/drives (2-3 main goals)
        5. Initial thoughts/mental state
        
        Make the characters complementary but distinct. They should have reasons to interact and potential for both cooperation and conflict.
        
        Format as JSON:
        {{
            "characters": [
                {{
                    "name": "Character Name",
                    "description": "Physical description and appearance",
                    "personality": "Personality traits and characteristics",
                    "drives": ["Drive 1", "Drive 2", "Drive 3"],
                    "initial_thoughts": ["Thought 1", "Thought 2"]
                }}
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            
            result = json.loads(response.choices[0].message.content)
            characters = []
            
            for char_data in result['characters']:
                char = Character(
                    name=char_data['name'],
                    description=char_data['description'],
                    personality=char_data['personality'],
                    drives=char_data['drives'],
                    current_thoughts=char_data['initial_thoughts']
                )
                characters.append(char)
            
            self.characters = characters
            print(f"✓ Generated {len(characters)} characters")
            for char in characters:
                print(f"  - {char.name}: {char.personality[:50]}...")
            
            return characters
            
        except Exception as e:
            print(f"✗ Error generating characters: {e}")
            return []
    
    def generate_story_outline(self) -> str:
        """Generate a story outline based on characters and scenario"""
        
        char_descriptions = []
        for char in self.characters:
            char_desc = f"{char.name}: {char.description}\nPersonality: {char.personality}\nDrives: {', '.join(char.drives)}"
            char_descriptions.append(char_desc)
        
        # Get available resources from scenario
        resources = []
        if self.scenario_config and 'resource_rules' in self.scenario_config:
            for allocation in self.scenario_config['resource_rules'].get('allocations', []):
                resources.append(allocation['description'])
        
        prompt = f"""
        Create a story outline for an interactive narrative with these characters:
        
        {chr(10).join(char_descriptions)}
        
        Setting: A forest environment with these available resources and locations:
        {chr(10).join('- ' + resource for resource in resources[:10])}  # Limit to first 10
        
        Create a 5-scene story outline that:
        1. Starts with the characters meeting/discovering their situation
        2. Builds tension through exploration and discovery
        3. Includes character development and relationship dynamics
        4. Has a climactic challenge or revelation
        5. Provides a satisfying resolution
        
        Each scene should have:
        - Setting/location
        - Key events
        - Character interactions
        - Potential discoveries or challenges
        
        Format as a clear outline with Scene 1, Scene 2, etc.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            outline = response.choices[0].message.content
            print("✓ Generated story outline")
            return outline
            
        except Exception as e:
            print(f"✗ Error generating story outline: {e}")
            return ""
    
    def generate_scene(self, scene_number: int, scene_description: str, previous_events: List[StoryEvent]) -> List[StoryEvent]:
        """Generate a single scene with character actions and dialogue"""
        
        # Create context from previous events
        recent_context = ""
        if previous_events:
            recent_context = "Previous events:\n"
            for event in previous_events[-5:]:  # Last 5 events for context
                if event.dialogue:
                    recent_context += f"{event.character}: {event.dialogue}\n"
                else:
                    recent_context += f"{event.character} {event.action}\n"
        
        char_states = []
        for char in self.characters:
            char_states.append(f"{char.name}: {char.personality}")
        
        prompt = f"""
        Write Scene {scene_number} as a screenplay-style narrative.
        
        Scene Description: {scene_description}
        
        Characters:
        {chr(10).join(char_states)}
        
        {recent_context}
        
        Write 8-12 exchanges showing:
        - Character actions and reactions
        - Realistic dialogue
        - Environmental interactions
        - Character development
        - Plot advancement
        
        Format each line as either:
        ACTION: [Character] [does something]
        DIALOGUE: [Character]: "[what they say]"
        NARRATION: [Environmental or scene description]
        
        Make it feel like a real conversation/interaction, not just exposition.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            
            scene_text = response.choices[0].message.content
            events = self._parse_scene_to_events(scene_text, scene_number)
            
            print(f"✓ Generated Scene {scene_number} with {len(events)} events")
            return events
            
        except Exception as e:
            print(f"✗ Error generating scene {scene_number}: {e}")
            return []
    
    def _parse_scene_to_events(self, scene_text: str, scene_number: int) -> List[StoryEvent]:
        """Parse screenplay-style scene text into story events"""
        events = []
        lines = scene_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            timestamp = f"Scene_{scene_number}_{i:02d}"
            
            if line.startswith('ACTION:'):
                # Parse action line
                action_text = line[7:].strip()
                # Try to extract character name
                char_name = None
                for char in self.characters:
                    if char.name in action_text:
                        char_name = char.name
                        break
                
                if char_name:
                    event = StoryEvent(
                        timestamp=timestamp,
                        character=char_name,
                        action=action_text
                    )
                    events.append(event)
            
            elif line.startswith('DIALOGUE:'):
                # Parse dialogue line
                dialogue_text = line[9:].strip()
                if ':' in dialogue_text:
                    char_name, speech = dialogue_text.split(':', 1)
                    char_name = char_name.strip()
                    speech = speech.strip(' "')
                    
                    event = StoryEvent(
                        timestamp=timestamp,
                        character=char_name,
                        action="speaks",
                        dialogue=speech
                    )
                    events.append(event)
            
            elif line.startswith('NARRATION:'):
                # Environmental description
                narration = line[10:].strip()
                event = StoryEvent(
                    timestamp=timestamp,
                    character="NARRATOR",
                    action=narration
                )
                events.append(event)
        
        return events
    
    def run_full_story(self, scenario_file: str, story_file: str, output_file: str = "generated_story.txt"):
        """Run the complete story generation process"""
        
        print("🎭 Starting Story Generation...")
        print("=" * 50)
        
        # Load scenario
        if not self.load_scenario(scenario_file):
            return False
        
        # Generate characters
        characters = self.generate_characters(story_file)
        if not characters:
            return False
        
        # Generate story outline
        outline = self.generate_story_outline()
        if not outline:
            return False
        
        print("\n📖 Story Outline:")
        print("-" * 30)
        print(outline)
        
        # Extract scenes from outline
        scenes = self._extract_scenes_from_outline(outline)
        
        print(f"\n🎬 Generating {len(scenes)} scenes...")
        print("-" * 30)
        
        # Generate each scene
        all_events = []
        for i, scene_desc in enumerate(scenes, 1):
            print(f"\n🎬 Scene {i}: {scene_desc[:50]}...")
            scene_events = self.generate_scene(i, scene_desc, all_events)
            all_events.extend(scene_events)
        
        # Save the complete story
        self._save_story(all_events, output_file, outline)
        
        print(f"\n✅ Story complete! Saved to {output_file}")
        print(f"📊 Total events: {len(all_events)}")
        
        return True
    
    def _extract_scenes_from_outline(self, outline: str) -> List[str]:
        """Extract scene descriptions from the story outline"""
        scenes = []
        lines = outline.split('\n')
        current_scene = ""
        
        for line in lines:
            if line.strip().startswith('Scene ') and ':' in line:
                if current_scene:
                    scenes.append(current_scene.strip())
                current_scene = line.strip()
            elif current_scene and line.strip():
                current_scene += " " + line.strip()
        
        if current_scene:
            scenes.append(current_scene.strip())
        
        # If no scenes found, create generic ones
        if not scenes:
            scenes = [
                "Scene 1: The characters meet and discover their situation",
                "Scene 2: Exploration and discovery of the environment", 
                "Scene 3: A challenge or conflict arises",
                "Scene 4: Characters work together to overcome obstacles",
                "Scene 5: Resolution and character growth"
            ]
        
        return scenes
    
    def _save_story(self, events: List[StoryEvent], filename: str, outline: str):
        """Save the complete story to a file"""
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("GENERATED STORY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("STORY OUTLINE:\n")
            f.write("-" * 30 + "\n")
            f.write(outline + "\n\n")
            
            f.write("CHARACTERS:\n")
            f.write("-" * 30 + "\n")
            for char in self.characters:
                f.write(f"{char.name}: {char.description}\n")
                f.write(f"Personality: {char.personality}\n")
                f.write(f"Drives: {', '.join(char.drives)}\n\n")
            
            f.write("STORY:\n")
            f.write("-" * 30 + "\n\n")
            
            current_scene = None
            for event in events:
                # Check if we're starting a new scene
                scene_num = event.timestamp.split('_')[1]
                if scene_num != current_scene:
                    current_scene = scene_num
                    f.write(f"\n--- SCENE {scene_num} ---\n\n")
                
                if event.character == "NARRATOR":
                    f.write(f"[{event.action}]\n\n")
                elif event.dialogue:
                    f.write(f"{event.character}: \"{event.dialogue}\"\n\n")
                else:
                    f.write(f"{event.action}\n\n")


# Example usage
def main():
    """Example of how to use the StoryStage system"""
    
    try:
        # You'll need to set your OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        # Initialize the stage
        stage = StoryStage(api_key)
        
        # Run the complete story generation
        success = stage.run_full_story(
            scenario_file="/home/bruce/Downloads/AllTheWorldAPlay/src/plays/scenarios/forest.py",  # Your scenario file
            story_file="/home/bruce/Downloads/AllTheWorldAPlay/src/plays/lost.py",       # Your story setup file
            output_file="forest_adventure.txt"
        )
        
        if success:
            print("🎉 Story generation completed successfully!")
        else:
            print("❌ Story generation failed.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()