from __future__ import annotations
import json
import traceback
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import random
from src.utils import hash_utils
from utils.Messages import UserMessage, SystemMessage
import re
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    from sim.context import Context
    from sim.agh import Character
    from sim.narrativeCharacter import NarrativeCharacter
    from sim.character_dataclasses import Goal

from sim.character_dataclasses import datetime_handler

@dataclass
class TurnSnapshot:
    """Snapshot of character state at a specific turn"""
    turn_number: int
    timestamp: datetime
    drive_activations: Dict[str, float]  # drive_text -> activation
    last_action: Optional[Dict[str, Any]]  # mode, action, target
    goal_progress: Dict[str, float]  # goal_id -> progress
    relationship_changes: Dict[str, str]  # actor_name -> relationship_summary

@dataclass
class DiscoveryEvent:
    """Record of a world discovery event"""
    timestamp: datetime
    event_type: str  # 'resource', 'location', 'environmental'
    description: str
    discoverer: Optional[str]  # character name

@dataclass
class StalenessMetrics:
    """Computed staleness indicators"""
    drive_flatlines: int
    action_loops: int  
    relationship_stagnation: int
    discovery_drought_turns: int
    goal_paralysis: int
    conversational_circles: int
    environmental_stasis: int
    staleness_score: int  # 0-10
    primary_factors: List[str]
    # Detail lists to provide actor/goal context for each indicator
    drive_flatlines_details: List[Tuple[str, str]] = field(default_factory=list)  # (character, drive)
    action_loops_details: List[Tuple[str, str]] = field(default_factory=list)    # (character, action)
    relationship_stagnation_details: List[str] = field(default_factory=list)     # character names
    goal_paralysis_details: List[Tuple[str, str]] = field(default_factory=list)  # (character, goal_id)
    conversational_circles_details: List[Tuple[str, str]] = field(default_factory=list)  # (charA, charB)

class NarrativeStalnessDetector:
    """Detects narrative staleness using 5 sliding window trackers"""
    
    def __init__(self, context: 'Context', window_size: int = 5):
        self.context = context
        self.window_size = window_size
        self.turn_counter = 0
        self.last_check_turn = 0
        self.check_frequency = 3  # Check every N turns
        
        # Sliding Window Trackers
        self.character_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.discovery_events: deque = deque(maxlen=window_size * 3)
        self.dialog_topics: deque = deque(maxlen=window_size * 2)
        self.environmental_changes: deque = deque(maxlen=window_size)
        
        # State tracking
        self.last_world_state = ""
        # Track known resources per character
        self.last_character_resource_counts: Dict[str, int] = {}
        # Track known actors per character  
        self.last_character_actor_counts: Dict[str, int] = {}
        self.known_locations = set()
        
        # Cache for LLM-based action similarity checks
        self.action_similarity_cache: Dict[str, bool] = {}
        
        # Initialize baseline resource counts
        self._initialize_baseline_counts()
        
    def _initialize_baseline_counts(self):
        """Initialize baseline counts for character knowledge"""
        for char in self.context.actors + self.context.npcs:
            if hasattr(char, 'resource_models') and char.resource_models:
                self.last_character_resource_counts[char.name] = len(char.resource_models.known_resources)
            else:
                self.last_character_resource_counts[char.name] = 0
                
            if hasattr(char, 'actor_models') and char.actor_models:
                self.last_character_actor_counts[char.name] = len(char.actor_models.known_actors)
            else:
                self.last_character_actor_counts[char.name] = 0
    
    def record_turn(self, character: 'Character'):
        """Record character state for this turn"""
        self.turn_counter += 1
        
        # Extract drive activations
        drive_activations = {}
        if hasattr(character, 'drives'):
            drive_activations = {
                drive.text: drive.activation 
                for drive in character.drives
            }
        
        # Extract last action
        last_action = None
        if hasattr(character, 'last_act') and character.last_act:
            last_action = {
                'mode': character.last_act.mode,
                'action': character.last_act.action,
                'target': str(getattr(character.last_act, 'target', None))
            }
        
        # Extract goal progress
        goal_progress = {}
        if hasattr(character, 'goals'):
            for goal in character.goals:
                goal_progress[goal.id] = (getattr(goal, 'progress', 0),f'{getattr(goal, "name", "")}: {getattr(goal, "description", "")}')
        
        # Create snapshot
        snapshot = TurnSnapshot(
            turn_number=self.turn_counter,
            timestamp=self.context.simulation_time,
            drive_activations=drive_activations,
            last_action=last_action,
            goal_progress=goal_progress,
            relationship_changes={}
        )
        
        self.character_snapshots[character.name].append(snapshot)
    
    def detect_discoveries(self, changes: str):
        """Detect new discoveries since last check"""
        total_discoveries = 0
        
        # Check for character resource discoveries
        for char in self.context.actors + self.context.extras:
            if not hasattr(char, 'resource_models') or not char.resource_models:
                continue
                
            current_count = len(char.resource_models.known_resources)
            last_count = self.last_character_resource_counts.get(char.name, 0)
            
            if current_count > last_count:
                discoveries = current_count - last_count
                total_discoveries += discoveries
                
                event = DiscoveryEvent(
                    timestamp=self.context.simulation_time,
                    event_type='resource',
                    description=f'{char.name} discovered {discoveries} new resources',
                    discoverer=char.name
                )
                self.discovery_events.append(event)
                self.last_character_resource_counts[char.name] = current_count
        
        # Check for character discoveries
        for char in self.context.actors + self.context.extras:
            if not hasattr(char, 'actor_models') or not char.actor_models:
                continue
                
            current_count = len(char.actor_models.known_actors)
            last_count = self.last_character_actor_counts.get(char.name, 0)
            
            if current_count > last_count:
                discoveries = current_count - last_count
                total_discoveries += discoveries
                
                event = DiscoveryEvent(
                    timestamp=self.context.simulation_time,
                    event_type='character',
                    description=f'{char.name} met {discoveries} new characters',
                    discoverer=char.name
                )
                self.discovery_events.append(event)
                self.last_character_actor_counts[char.name] = current_count
        
        # Check for environmental changes
        if changes:
            prompt = [UserMessage(content="""Rate the significance of these changes in terms of the central dramatic question:
#Changes:
{{$changes}}

#Central Dramatic Question:
{{$central_narrative}}

#Respond with an integer in the range of 0 - 5, where 0 is no change and 5 is a major change, and a very terse (3-5 words) description of the change.
Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Close the hash-formatted text with ##  on a separate line, as shown below.
be careful to insert line breaks only where shown, separating a value from the next tag:

#significance [0-5 integer]
#description [3-5 words]
##

End your response with:
</end>
""")]
            response = self.context.llm.ask({"changes":changes, "central_narrative":self.context.central_narrative}, prompt, 
                                            tag='NarrativeStaleness.detect_discoveries', stops=['</end>'], max_tokens=20)
            try:
                significance = hash_utils.find('significance', response)
                description = hash_utils.find('description', response)
                significance = int(significance.strip())
                if significance >= 2:
                    # Create event with both significance and description
                    event = DiscoveryEvent(
                        timestamp=self.context.simulation_time,
                        event_type='environmental',
                        description=f'Environmental change: {description}',
                        discoverer=None
                    )
                    self.discovery_events.append(event)
            except Exception as e:
                traceback.print_exc()
                pass# Fallback behavior
        self.last_world_state = self.context.current_state
    
    def extract_dialog_topics(self):
        """Extract recent dialog topics and detect repetitive conversations"""
        for actor in self.context.actors + self.context.extras:
            if hasattr(actor, 'actor_models'):
                for known_actor in actor.actor_models.actor_models():
                    if hasattr(known_actor, 'dialog') and known_actor.dialog and hasattr(known_actor.dialog, 'transcript'):
                        # Get recent dialog turns
                        recent_turns = known_actor.dialog.transcript[-10:] if known_actor.dialog.transcript else []
                        if len(recent_turns) < 5:  # Need at least 2 turns to detect repetition
                            continue
                            
                        # Format turns for LLM analysis
                        dialog_text = '\n'.join(recent_turns)
                        
                        try:
                            prompt = [UserMessage(content="""Are these recent dialog turns repetitive, covering familiar ground, or showing conversational loops?

#Recent Dialog:
{{$dialog}}

#Central Dramatic Question:
{{$central_narrative}}

Consider whether the conversation is:
- Retreading previous topics without advancement
- Going in circles
- Lacking new information or development
                                                 
Respond with: repeat / fresh

Respond ONLY with one word. Do not include any reasoning, introductory, discursive, explanatory, or formatting text.
End your response with:
</end>
""")]
                            response = self.context.llm.ask(
                                {"dialog": dialog_text, "central_narrative": self.context.central_narrative or "None established"}, 
                                prompt,                               
                                tag='NarrativeStaleness.extract_dialog_topics', 
                                stops=['</end>'], 
                                max_tokens=10
                            )
                            
                            assessment = response.strip().split('</end>')[0].lower()
                            
                            if assessment == 'repeat':
                                self.dialog_topics.append({
                                    'timestamp': self.context.simulation_time,
                                    'participants': [actor.name, known_actor.canonical_name],
                                    'type': 'repetitive',
                                    'content': f'Repetitive conversation between {actor.name} and {known_actor.canonical_name}'
                                })
                                
                        except Exception as e:
                            # Fallback: skip this dialog analysis
                            continue
    
    def _are_actions_semantically_similar(self, action1: tuple, action2: tuple) -> bool:
        """Check if two actions are semantically similar using LLM with caching"""
        # Quick literal equality check first
        if action1 == action2:
            return True
            
        # Format actions for comparison
        action1_str = f"{action1[0]}: {action1[1]}"  # mode: action
        action2_str = f"{action2[0]}: {action2[1]}"  # mode: action
        
        # Create cache key (ensure consistent ordering)
        actions_sorted = sorted([action1_str, action2_str])
        cache_key = f"{actions_sorted[0]}||{actions_sorted[1]}"
        
        # Check cache first
        if cache_key in self.action_similarity_cache:
            return self.action_similarity_cache[cache_key]
        
        # Ask LLM for semantic similarity
        try:
            prompt = [UserMessage(content="""Are these two actions semantically equivalent in terms of character behavior and narrative purpose?

Action 1: {{$action1}}
Action 2: {{$action2}}

Consider:
- Are they trying to accomplish the same goal?
- Do they represent the same type of behavior?
- Would they advance the narrative in similar ways?

Respond with: similar/different

Respond ONLY with one word. Do not include any reasoning or explanation.
End your response with:
</end>
""")]
            response = self.context.llm.ask(
                {"action1": action1_str, "action2": action2_str}, 
                prompt, 
                tag='NarrativeStaleness.action_similarity', 
                stops=['</end>'], 
                max_tokens=5
            )
            
            assessment = response.strip().split('</end>')[0].lower()
            is_similar = assessment == 'similar'
            
            # Cache the result
            self.action_similarity_cache[cache_key] = is_similar
            return is_similar
            
        except Exception as e:
            # Fallback to literal comparison on LLM error
            return action1 == action2

    def compute_staleness_metrics(self) -> StalenessMetrics:
        """Compute staleness indicators from sliding window data"""
        metrics = StalenessMetrics(
            drive_flatlines=0,
            action_loops=0,
            relationship_stagnation=0,
            discovery_drought_turns=0,
            goal_paralysis=0,
            conversational_circles=0,
            environmental_stasis=0,
            staleness_score=0,
            primary_factors=[]
        )
        
        # count # characters
        chars = set()
        for char_name, snapshots in self.character_snapshots.items():
            chars.add(char_name)

        # 1. Drive Flatlines - Check for unchanging drive activations
        for char_name, snapshots in self.character_snapshots.items():
            if len(snapshots) >= 3:
                for drive_text in snapshots[-1].drive_activations:
                    recent_values = [
                        s.drive_activations.get(drive_text, 0) 
                        for s in list(snapshots)[-3:]
                    ]
                    if len(set(recent_values)) <= 1:  # All same value
                        metrics.drive_flatlines += 1
                        metrics.drive_flatlines_details.append((char_name, drive_text))
        
        # 2. Action Loops - Check for repetitive action patterns  
        for char_name, snapshots in self.character_snapshots.items():
            if len(snapshots) >= 3:
                recent_actions = []
                for s in list(snapshots)[-3:]:
                    if s.last_action:
                        recent_actions.append((s.last_action.get('mode', ''), s.last_action.get('action', '')))
                
                if len(recent_actions) >= 2:
                    # First check: same action instance = not a real loop
                    if recent_actions[-1] is recent_actions[-2]:
                        continue  # Skip, same instance
                    
                    # Then check semantic similarity for different instances
                    if self._are_actions_semantically_similar(recent_actions[-1], recent_actions[-2]):
                        metrics.action_loops += 1
                        metrics.action_loops_details.append((char_name, recent_actions[-1][1]))
        
        # 3. Relationship Stagnation
        for char_name, snapshots in self.character_snapshots.items():
            character: NarrativeCharacter = self.context.get_actor_or_npc_by_name(char_name)
            if character and len(snapshots) >= 1:
                relationship_changes = 0
                for model in character.actor_models.known_actors.values():
                    if model.recent_relationship_update:
                        relationship_changes += 1
                        model.recent_relationship_update = False
                if relationship_changes == 0:
                    metrics.relationship_stagnation += 1
                    metrics.relationship_stagnation_details.append(char_name)
        
        # 4. Discovery Drought
        recent_discoveries = [
            e for e in self.discovery_events 
            if (self.context.simulation_time - e.timestamp).total_seconds() < 600  # Last 10 minutes
        ]
        if len(recent_discoveries) <= 2 and self.turn_counter > 5:
            metrics.discovery_drought_turns = min(self.turn_counter, 5)
        
        # 5. Goal Paralysis
        goal_count = 0
        for char_name, snapshots in self.character_snapshots.items():
            if len(snapshots) >= 2:  # Need at least two points to compute a delta
                # Use as many recent snapshots as available, up to the configured window_size
                window_snapshots = list(snapshots)[-self.window_size:]
                for goal_id in window_snapshots[-1].goal_progress:
                    goal_count += 1
                    # Gather progress history for this goal (only snapshots where goal exists)
                    progress_values = [s.goal_progress[goal_id] for s in window_snapshots if goal_id in s.goal_progress]
                    if len(progress_values) < 3:
                        continue  # Not enough data points for this goal
                    deltas = [abs(progress_values[i][0] - progress_values[i - 1][0]) for i in range(1, len(progress_values))]
                    avg_delta = sum(deltas) / len(deltas)
                    if avg_delta < 0.1:
                        metrics.goal_paralysis += 1
                        metrics.goal_paralysis_details.append((char_name, str(goal_id), window_snapshots[-1].goal_progress[goal_id][1]))
        
        # 6. Conversational Circles
        # Count recent repetitive dialog events
        recent_repetitive_dialogs = [
            dt for dt in list(self.dialog_topics)[-6:]  # Last 6 dialog events
            if dt.get('type') == 'repetitive'
        ]
        metrics.conversational_circles = len(recent_repetitive_dialogs)
        
        # Collect conversational circle participants for details
        for dt in recent_repetitive_dialogs:
            participants = dt.get('participants', [])
            if isinstance(participants, list) and len(participants) == 2:
                metrics.conversational_circles_details.append(tuple(participants))
        
        # 7. Environmental Stasis
        if len(self.environmental_changes) < 2 and self.turn_counter > 5:
            metrics.environmental_stasis = 1
        
        # Calculate overall staleness score
        total_indicators = (
            min(float(metrics.drive_flatlines)/len(chars), 3) +
            min(float(metrics.action_loops*2)/len(chars), 2) +
            min(float(metrics.relationship_stagnation)/len(chars), 2) +
            min(metrics.discovery_drought_turns, 2) +
            min(float(metrics.goal_paralysis)/goal_count, 3) +
            min(metrics.conversational_circles, 2) +
            metrics.environmental_stasis
        )
        
        metrics.staleness_score = min(10, total_indicators)
        
        # Identify primary factors
        if metrics.drive_flatlines > 0:
            metrics.primary_factors.append("drive_flatlines")
        if metrics.action_loops > 0:
            metrics.primary_factors.append("action_loops")
        if metrics.relationship_stagnation > 1:
            metrics.primary_factors.append("relationship_stagnation")
        if metrics.discovery_drought_turns > 2:
            metrics.primary_factors.append("discovery_drought")
        if metrics.goal_paralysis > 1:
            metrics.primary_factors.append("goal_paralysis")
        if metrics.conversational_circles > 1:
            metrics.primary_factors.append("conversational_circles")
        if metrics.environmental_stasis > 0:
            metrics.primary_factors.append("environmental_stasis")
        
        return metrics
    
    def generate_staleness_analysis_prompt(self, metrics: StalenessMetrics, scene: Dict[str, Any]) -> List[Any]:
        """Generate the LLM prompt for staleness analysis"""
                
        system_prompt = SystemMessage(content="""You are a narrative tension analyzer for an interactive drama system. 
Your job is to detect when dramatic momentum has stalled and determine if an intervation is needed to reinvigorate the story.
You will analyze patterns across multiple data streams to identify narrative staleness indicators.""")
        
        # Quick scoreboard with computed metrics
        scoreboard = f"""## Computed Staleness Metrics\n"""
        scoreboard += f"Staleness Score: {metrics.staleness_score}/10\n"
        scoreboard += f"Primary Factors: {', '.join(metrics.primary_factors) if metrics.primary_factors else 'None'}\n"
        scoreboard += f"Drive Flatlines: {metrics.drive_flatlines} ({self._format_list(metrics.drive_flatlines_details)})\n"
        scoreboard += f"Action Loops: {metrics.action_loops} ({self._format_list(metrics.action_loops_details)})\n"
        scoreboard += f"Relationship Stagnation: {metrics.relationship_stagnation} ({self._format_list(metrics.relationship_stagnation_details)})\n"
        scoreboard += f"Discovery Drought Turns: {metrics.discovery_drought_turns}\n"
        scoreboard += f"Goal Paralysis: {metrics.goal_paralysis} ({self._format_list(metrics.goal_paralysis_details)})\n"
        scoreboard += f"Conversational Circles: {metrics.conversational_circles} ({self._format_list(metrics.conversational_circles_details)})\n"
        scoreboard += f"Environmental Stasis: {metrics.environmental_stasis}\n"
        
        analysis_prompt = UserMessage(content=f"""# NARRATIVE STALENESS ANALYSIS

## Central Dramatic Question
{self.context.central_narrative if hasattr(self.context, 'central_narrative') else 'Character-driven improvisation'}

## Current Situation
{self.context.current_state}

## Scoreboard analysis of important metrics
{scoreboard}

---

# STALENESS INDICATORS TO CHECK:

**Drive Flatlines**: Character_name - drive_id, Drive_text : drive with flatline activation
**Action Loops**: Character_name - Action_text : action with repetitive execution
**Relationship Stagnation**: Character_name : trust/tension levels static despite interaction?
**Discovery Drought**:turns since last discovery : Have 3+ turns passed with no new world elements?
**Goal Paralysis**: Character_name - Goal_id - Goal_name : Are characters making no measurable progress toward objectives?
**Conversational Circles**: Character_name - Character_name : Are dialog topics recycling without deepening?
**Environmental Stasis**:turns since last environmental change : Has the physical world remained unchanged?

---

###
In designing an intervention, consider that the current plan for the act in progress is:

{json.dumps(self.context.current_act, indent=2, default=datetime_handler)}

###

and the current scene within that act, in which the intervention will be implemented, is:
{json.dumps(scene, indent=2, default=datetime_handler)}

###

Tthe following interventions have already been implemented. Do not repeat any, literally, in style, or in substance.

{'\n'.join(self.context.previous_interventions)}

###

# ANALYSIS REQUIRED:

1. **Staleness Score** (0-10): Rate current narrative momentum
   - 0-3: High energy, no intervention needed
   - 4-6: Moderate concern, monitor closely  
   - 7-10: Stalled, intervention required

2. **Primary Staleness Factors**: Which indicators are triggering concern?

3. **Intervention Type** (if score ≥ 7):
   - **Environmental**: Natural disaster, structural failure, weather
   - **Character**: New character arrival, injury/illness, revelation
   - **Resource**: Discovery of key object, loss of critical item
   - **Psychological**: Change in a character's drive, increasing or descreasing its activation (urgency)
   - **Temporal**: Deadline pressure, time limit imposed
   - **Moral**: Ethical dilemma forcing character choice

4. **Entity Spawn Requirements** (if intervention needed):
   - What new Characters, sprites, or resources would serve the intervention?
   - Provide XML specifications for each

---

# RESPONSE FORMAT:

#staleness_score [0-10 integer]
#primary_factors [comma-separated list]
#intervention_needed [yes/no]
#intervention_type [environmental/character/resource/temporal/moral]
#intervention_description [1-2 sentence description of the event]

## Entity Spawning or drive state change (if needed), in XML format - name is required, and is the name of the entity or state change. A state_change can only be used to change the activation of a drive:
```xml
<new_character   name="..." description="..." motivation="..." drives="..."/>
<new_sprite id="..." description="..." behavior="..." lifecycle="..."/>
<new_resource name="..." description="..." properties="..."/>
<state_change name="character_name" drive="drive_id" value="+ / -" />
```


#justification [1 sentence explaining why this intervention serves the central dramatic question]

End your response with:
</end>
""")
        
        return [system_prompt, analysis_prompt]
    
    def _format_dict(self, data: Dict) -> str:
        """Format dictionary data for prompt"""
        if not data:
            return "No data available"
        
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for subkey, subvalue in value.items():
                    lines.append(f"  {subkey}: {subvalue}")
            else:
                lines.append(f"{key}: {value}")
        return '\n'.join(lines)
    
    def _format_list(self, data_list: List[Any]) -> str:
        """Helper to format list/tuple data into comma separated string"""
        if not data_list:
            return 'None'
        formatted_items = []
        for item in data_list:
            if isinstance(item, (list, tuple)):
                formatted_items.append(' - '.join(map(str, item)))
            else:
                formatted_items.append(str(item))
        return '\n\t'.join(formatted_items)
    
    def extract_entity_spawning(self, response: str) -> Dict[str, List[Dict]]:
        """Extract entities, with LLM repair if parsing fails"""
        
        result = {
            'new_character': [],
            'new_sprite': [],
            'new_resource': [],
            'state_change': []
        }
        
        # Find XML code blocks using regex
        xml_pattern = r'```xml\s*\n(.*?)\n```'
        xml_matches = re.findall(xml_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if not xml_matches:
            return result
        
        # Try parsing each block
        for xml_content in xml_matches:
            try:
                # Attempt direct parsing
                wrapped_xml = f"<root>{xml_content.strip()}</root>"
                root = ET.fromstring(wrapped_xml)
                
                # Parse each entity type
                for entity_type in ['new_character', 'new_sprite', 'new_resource', 'state_change']:
                    for elem in root.findall(entity_type):
                        entity_dict = dict(elem.attrib)
                        result[entity_type].append(entity_dict)
                        
            except ET.ParseError as e:
                # Use LLM repair instead of regex fallback
                print(f"XML parsing failed, attempting repair: {e}")
                repaired_xml = self.context.repair_xml(xml_content, str(e))
                
                if repaired_xml:
                    try:
                        wrapped_xml = f"<root>{repaired_xml.strip()}</root>"
                        root = ET.fromstring(wrapped_xml)
                        
                        # Parse each entity type
                        for entity_type in ['new_character', 'new_sprite', 'new_resource', 'state_change']:
                            for elem in root.findall(entity_type):
                                entity_dict = dict(elem.attrib)
                                result[entity_type].append(entity_dict)
                                
                    except ET.ParseError as repair_error:
                        print(f"Could not repair XML: {repair_error}")
                        continue
                else:
                    print(f"XML repair failed for: {xml_content[:100]}...")
                    continue
        
        return result

    def parse_staleness_response(self, response: str) -> Dict[str, Any]:
        """
        Parse complete staleness analysis response including entities.
        """
        import utils.hash_utils as hash_utils
        
        # Parse hash fields as before
        staleness_score = hash_utils.find('staleness_score', response)
        primary_factors = hash_utils.find('primary_factors', response)
        intervention_needed = hash_utils.find('intervention_needed', response)
        intervention_type = hash_utils.find('intervention_type', response)
        intervention_description = hash_utils.find('intervention_description', response)
        justification = hash_utils.find('justification', response)
        
        # Parse entity spawning
        entities = self.extract_entity_spawning(response)
        
        result = {
            'staleness_score': int(staleness_score.strip()) if staleness_score else 0,
            'primary_factors': primary_factors.split(',') if primary_factors else [],
            'intervention_needed': intervention_needed.strip().lower() == 'yes' if intervention_needed else False,
            'intervention_type': intervention_type.strip() if intervention_type else None,
            'intervention_description': intervention_description.strip() if intervention_description else None,
            'justification': justification.strip() if justification else None,
            'entities': entities,
            'raw_response': response
        }
        
        return result
    
    async def analyze_staleness(self, changes: str, scene: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run full staleness analysis and return intervention recommendations"""
        # Update turn counter and record discoveries
        self.detect_discoveries(changes)
        self.extract_dialog_topics()
        
        # Check if it's time for analysis
        if self.turn_counter - self.last_check_turn < self.check_frequency:
            return None
        
        self.last_check_turn = self.turn_counter
        
        # Compute metrics
        metrics = self.compute_staleness_metrics()
        
        # Generate LLM analysis
        prompt = self.generate_staleness_analysis_prompt(metrics, scene)
        
        try:
            response = self.context.llm.ask(
                {},
                prompt,
                tag='NarrativeStaleness.analyze',
                temp=0.7,
                stops=['</end>'],
                max_tokens=500
            )
            
            # Parse response
            result = self.parse_staleness_response(response)
            
            # Access parsed entities
            for character in result['entities']['new_character']:
                print(f"New character: {character['name']} - {character['description']}")
            
            for sprite in result['entities']['new_sprite']:
                print(f"New sprite: {sprite['id']} - {sprite['description']}")
            
            for resource in result['entities']['new_resource']:
                print(f"New resource: {resource['name']} - {resource['description']}")

            for state_change in result['entities']['state_change']:
                print(f"State change: {state_change['name']} - {state_change['drive']} - {state_change['value']}")
            
            return result
            
        except Exception as e:
            print(f"Error in staleness analysis: {e}")
            traceback.print_exc()
            return None
    
    def should_trigger_intervention(self, analysis: Dict[str, Any]) -> bool:
        """Determine if intervention should be triggered"""
        if not analysis:
            return False
        
        staleness_score = analysis.get('staleness_score', 0)
        intervention_needed = analysis.get('intervention_needed', False)
        
        # Trigger if high staleness score OR explicit LLM recommendation
        return staleness_score >= 7 or intervention_needed 