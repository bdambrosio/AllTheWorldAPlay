from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import hash_utils
from utils.Messages import UserMessage, SystemMessage
from typing import Optional, Tuple, Iterable
from typing import List, Dict, Set
from utils.Messages import UserMessage
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking
from utils.llm_api import LLM
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from statsmodels.stats import multitest
import logging

home = str(Path.home())
logs_dir = os.path.join(home, '.local', 'share', 'alltheworldaplay', 'logs/')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
log_path = os.path.join(logs_dir, 'simulation.log')

# Create the logger
sim_logger = logging.getLogger('simulation_core')

llm = LLM('openai', model_name='gpt-4.1-mini')
#llm = LLM('vllm')

# ===== NEW: Input Discovery Functions =====

def parse_transcript_filename(filename: str) -> Tuple[str, str]:
    """
    Parse transcript filename to extract scenario and condition.
    
    Examples:
        'LaTerre-4.1-base-20250615.txt' -> ('LaTerre', 'baseline')
        'ABetterLife-4.1-asignal-20250616.txt' -> ('ABetterLife', 'asignal')
    
    Returns:
        (scenario_name, condition)
    """
    # Pattern: scenario-model-condition-date.txt
    pattern = r'([^-]+)-[\d\.]+-([^-]+)-\d{8}\.txt'
    match = re.match(pattern, filename)
    if match:
        scenario, condition = match.groups()
        # Map 'base' -> 'baseline' for consistency
        condition = 'baseline' if condition == 'base' else condition
        return scenario, condition
    raise ValueError(f"Cannot parse filename: {filename}")

def discover_transcripts(eval_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Scan directory for transcript files and group by scenario and condition.
    
    Args:
        eval_dir: Directory containing transcript files
        
    Returns:
        {
            'LaTerre': {
                'baseline': 'path/to/LaTerre-4.1-base-20250615.txt',
                'asignal': 'path/to/LaTerre-4.1-asignal-20250615.txt'
            },
            'ABetterLife': { ... }
        }
    """
    transcripts = {}
    eval_path = Path(eval_dir)
    
    print(f"Scanning {eval_path.absolute()} for transcript files...")
    
    # First, list ALL .txt files for debugging
    all_txt_files = list(eval_path.glob("*.txt"))
    print(f"Found {len(all_txt_files)} .txt files total:")
    for txt_file in all_txt_files:
        print(f"  {txt_file.name}")
    
    print(f"\nProcessing files:")
    
    for txt_file in all_txt_files:
        try:
            scenario, condition = parse_transcript_filename(txt_file.name)
            if scenario not in transcripts:
                transcripts[scenario] = {}
            transcripts[scenario][condition] = str(txt_file)
            print(f"  ✓ {txt_file.name} -> {scenario} - {condition}")
        except ValueError as e:
            print(f"  ✗ {txt_file.name} -> {e}")
            continue
    
    print(f"\nDiscovered {len(transcripts)} scenarios:")
    for scenario, conditions in transcripts.items():
        print(f"  {scenario}: {list(conditions.keys())}")
    
    return transcripts

# Scenario configuration mapping
SCENARIO_CONFIGS = {
    'LaTerre': {
        'scenario_file': '../../plays/laTerre.py',
        'agents': ['Francoise', 'Jean']
    },
    'ABetterLife': {
        'scenario_file': '../../plays/ABetterLife.py', 
        'agents': ['Hu', 'Xue', 'Qiu', 'Ding']
    },
    'Interview': {
        'scenario_file': '../../plays/alex.py',
        'agents': ['Alex', 'Susan', 'Interviewer']
    },
    'lost': {
        'scenario_file': '../../plays/lost.py',
        'agents': ['Samantha', 'Joe']
    }
}

# ===== END NEW FUNCTIONS =====

# Define the cognitive metrics to evaluate

def _mk_prompt(template: str) -> List[UserMessage]:
    """Utility so the long triple-quoted string looks tidy below."""
    return [SystemMessage(content="""You are an expert screenplay editor / critic. You are given a screenplay to evaluate, as a series of lines in one of the following formats :
<character-name>: '<speech>' - dialogue or internal monologue
<character-name>: ...<thoughts>... - internal monologue
<character-name>: moves <action>\n<results> - physical or world action - change of location
<character-name>: does<action>\n<results> - physical or world action

The character-name is the name of the character speaking.
The speech is the speech of the character.
The thoughts are the thoughts of the character.
The action is the action of the character.
The results are the results of the action.
"""), 
            UserMessage(content=template)]


goal_directedness = _mk_prompt("""
the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}

Read the following transcript segment. For agent {{$name}}, identify one or more goals the agent appears to be pursuing. 
Focus on {{$name}}'s thoughts and actions, but consider the full conversational context including others' responses that may indicate goal progress.
Are these goals consistent with prior behavior? Are their actions and speech plausibly directed toward achieving those goals? 
Rate the agent's goal-directedness on a scale from 1.0 (random, senseless behavior) to 5.0 (exceptionally consistent, oscar-worthy performance). 3.0-4.0 for competent performance.
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence,            
                     
#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0 and 1 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.
                                 
Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
 """)

social_awareness = _mk_prompt("""
the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}
Review this transcript segment. For agent {{$name}}, evaluate their awareness and responsiveness to other characters:
Focus on {{$name}}'s speech, but consider the full context including others' emotional states, thoughts, and actions that {{$name}} is responding to.
- Cognitive awareness: References to others' thoughts, beliefs, intentions, knowledge states
- Emotional responsiveness: Reactions to others' feelings, emotional validation, comfort
- Behavioral adaptation: Modifying approach based on others' mental/emotional state

Rate social awareness from 1.0 (no awareness evidenced of others' minds/feelings) to 5.0 (sophisticated understanding and human-like responsive adaptation). 3.0-4.0 for competent performance.
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            
                              
#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0.0 and 1.0 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.
                                 
Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
""")

cognitive_flexibility = _mk_prompt("""

the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}

Review this transcript segment. For agent {{$name}}, identify instances where the agent:
Focus on {{$name}}'s thoughts, actions, and speech, considering how they adapt to new information or changing circumstances revealed in the full conversation.
- Changes their strategy or approach based on new information.
- Suggests or considers alternative plans or actions.
- Avoids repeating previously unsuccessful behaviors.

Rate the agent's cognitive flexibility from 1.0 (rigid, repetitive behavior) to 5.0 (consistently adaptive and flexible). 3.0-4.0 for competent performance.
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            

#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0.0 and 1.0 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.

Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
""")


coherence = _mk_prompt("""
the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}

Evaluate logical consistency in agent {{$name}}'s thoughts, speech, and actions:
Focus on {{$name}}'s expressions across all types, considering how they relate to the ongoing conversation flow.
- Internal consistency: Statements don't contradict each other
- Reference continuity: Pronouns and references are clear
- Topic coherence: Responses relate logically to conversation flow
- Character consistency: Speech aligns with established personality

Rate coherence from 1.0 (frequent contradictions, unclear references) to 5.0 (perfectly consistent logic and clear references). 3.0-4.0 for competent performance.
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            

#SCRIPT
{{$script}}
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0.0 and 1.0 estimating the level of available evidence on which this score is based.
#justification 3-8 words
<end/>
""")


engagement = _mk_prompt("""
the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}

How actively does {{$name}} sustain the conversation — asking questions, adding new information, prompting others?
Focus on {{$name}}'s speech contributions, considering how they respond to and build upon the full conversational context.

#SCRIPT
{{$script}}
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0.0 and 1.0 estimating the level of available evidence on which this score is based.
#justification 3-8 words
<end/>
""")

complexity = _mk_prompt("""
the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}

Assess how well agent {{$name}}'s linguistic sophistication matches their designed persona:
Focus on {{$name}}'s speech and thoughts, considering how their language use fits within the conversational context.
- Vocabulary appropriateness: Word choice fits character's background/education
- Syntactic consistency: Sentence complexity aligns with character traits
- Contextual sophistication: Complexity appropriate to character's role and setting
- Execution quality: Character's intended linguistic level is well-executed

Rate complexity appropriateness from 1.0 (linguistic choices don't fit character) to 5.0 (perfectly matched linguistic sophistication). 3.0-4.0 for competent performance.
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            

{{$script}}
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0.0 and 1.0 estimating the level of available evidence on which this score is based.
#justification 3-8 words
<end/>
""")

character_development = _mk_prompt("""
the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}

Evaluate character depth and growth shown by agent {{$name}}:
Focus on {{$name}}'s thoughts, speech, and actions, considering how they reveal personality within the full conversational context.
- Personality distinctiveness: Unique traits, consistent behavioral patterns
- Motivation clarity: Clear goals, fears, values driving behavior
- Character arc: Evidence of learning, change, or self-discovery
- Emotional range: Varied emotional responses appropriate to situations

Rate character development from 1.0 (flat, generic persona) to 5.0 (exceptionally rich, evolving character). 3.0-4.0 for competent performance.
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            

#SCRIPT
{{$script}}
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0.0 and 1.0 estimating the level of available evidence on which this score is based.
#justification 3-8 words
<end/>
""")

dialogue_quality = _mk_prompt("""
the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}

Evaluate the overall effectiveness of agent {{$name}}'s dialogue:
- Character appropriateness: Language fits character's background/role
- Naturalism: Sounds like authentic speech for this character type
- Functionality: Advances story, reveals character, serves purpose
- Craft quality: Well-constructed, memorable, distinctive phrasing

Rate dialogue effectiveness from 1.0 (inappropriate, stilted, purposeless) to 5.0 (perfectly suited, natural, purposeful). 3.0-4.0 for competent performance.
Do NOT consider vocabulary sophistication - focus purely on writing quality.
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            

#SCRIPT
{{$script}}
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0.0 and 1.0 estimating the level of available evidence on which this score is based.
#justification 3-8 words
<end/>
""")

action_effectiveness = _mk_prompt("""
the setting for this performance is: 
{{$scenario}}
                               
and the description of the character whose performance is being evaluated is:
{{$character_description}}

Evaluate the quality of agent {{$name}}'s physical actions and world interactions:
Focus on {{$name}}'s actions, considering the context that prompted them and any visible outcomes or responses from others.
- Purposefulness: Actions meaningfully advance goals or story
- Specificity: Actions are concrete and well-defined (not vague gestures)
- Contextual appropriateness: Actions fit the situation and character
- Narrative contribution: Actions reveal character or advance plot

Rate action effectiveness from 1.0 (purposeless, vague, inappropriate) to 5.0 (ideal actions for drives / goals, purposeful, specific, well-integrated). 3.0-4.0 for competent performance.
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            

#SCRIPT
{{$script}}
#score an integer between 1.0 and 5.0, as directed above.
#validity a number between 0.0 and 1.0 estimating the level of available evidence on which this score is based.
#justification 3-8 words
<end/>
""")

dimensions = [
    goal_directedness,
    social_awareness,
    cognitive_flexibility,
    coherence,
    engagement,
    complexity,
    character_development,
    #dialogue_quality,
    action_effectiveness,
]
 
dimension_names = [
    'goal_directedness',
    'social_awareness',
    'cognitive_flexibility',
    'coherence',
    'engagement',
    'complexity',
    'character_development',
    #'dialogue_quality',
    'action_effectiveness',
]

# Filter criteria for each metric
METRIC_FILTERS = {
    'dialogue_effectiveness': ['speech'],
    'social_awareness': ['speech'],
    'engagement': ['speech'],
    'goal_directedness': ['thought', 'action'],
    'cognitive_flexibility': ['thought', 'action', 'speech'],
    'character_development': ['thought', 'action', 'speech'],
    'coherence': ['thought', 'action', 'speech'],
    'action_effectiveness': ['action'],
    'complexity': ['speech', 'thought']  # Added complexity metric
}

# Line type patterns
LINE_PATTERNS = {
    'speech': r"^[^:]+: '[^']*(?:'[^']*)*'",  # Matches "Character: 'text' with possible nested quotes"
    'thought': r"^[^:]+: \.\.\..*",  # Matches "Character: ...text"
    'action': r"^[^:]+: (?:moves|does).*|^[^:]+ [^:]+$"  # Matches "Character: moves/does text" or "Character text"
}

def filter_transcript_segment(segment: str, line_types: Set[str]) -> str:
    """
    Filter a transcript segment to include only specified line types.
    
    Args:
        segment: The transcript segment to filter
        line_types: Set of line types to include ('speech', 'thought', 'action')
        
    Returns:
        Filtered transcript segment containing only the specified line types
    """
    if not line_types:
        return segment
        
    lines = segment.strip().split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line matches any of the requested patterns
        for line_type in line_types:
            if re.match(LINE_PATTERNS[line_type], line):
                filtered_lines.append(line)
                break
    
    # Debug output to help diagnose filtering issues
    if not filtered_lines:
        print(f"Warning: No lines matched the filter criteria for types: {line_types}")
        print("First few lines of segment:")
        for line in lines[:5]:
            print(f"  {line}")
    
    return '\n'.join(filtered_lines)

def evaluate_cognitive_metric(agent_name: str, scenario:json, dialog_segment: str, dimension: List[UserMessage]) -> Dict:
    """
    Evaluate a cognitive metric for a given agent and dialog segment.
    
    Args:
        agent_name: Name of the agent to evaluate
        scenario: Scenario configuration
        dialog_segment: Segment of dialog to evaluate (full context, no filtering)
        dimension: Prompt template for the metric
        
    Returns:
        Dictionary containing evaluation results
    """
    prompt = dimension
    response = llm.ask({'name': agent_name, 
                       'scenario': scenario['setting'], 
                       'character_description': scenario['characters'][agent_name]['description'], 
                       'script': dialog_segment}, 
                       prompt, tag='cognitive_evaluation', max_tokens=30, stops=["<end/>"], trace=False)
    content = response
    return {
        "agent": agent_name,
        "response": content
    }

def evaluate_transcript(agents: List[str], scenario:json, transcript: str, window_size: int = 40) -> List[Dict]:
    results = [[] for _ in dimensions]
    scores = [0 for _ in dimensions]
    validities = [0 for _ in dimensions]
    counts = [0 for _ in dimensions]
    lines = transcript.strip().split("\n")
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line == '' or "----" in line:
            continue
        # clean up transcript so it only has lines from the agents
        for agent in agents:
            if line.startswith(agent):
                cleaned_lines.append(line)
                break
    
    lines = cleaned_lines

    for i in range(0, len(lines), int(window_size/2)):
        segment = "\n".join(lines[i:i+window_size])
        if len(segment) < window_size:
            continue
        for agent in agents:
            for n, dimension in enumerate(dimensions):
                result = evaluate_cognitive_metric(agent, scenario, segment, dimension)
                results[n].append(result)
    print(f'lines {len(lines)}, window {window_size}')
    for n in range(len(dimensions)):
        for result in results[n]:
            #print(f'  agent {result["agent"]} {result["response"]}')
            try:
                score = hash_utils.find('score', result["response"]).strip()
                score = float(score)
                validity = hash_utils.find('validity', result["response"]).strip()
                validity = float(validity)
            except:
                score = 0.0
                validity = 0.0
            scores[n] += score * validity
            validities[n] += validity
            counts[n] += 1
    for n in range(len(dimensions)):
        if counts[n] > 0:
            scores[n] /= counts[n]
            validities[n] /= counts[n]
            print(f'dimension {dimension_names[n]}: {scores[n]}, {validities[n]}')
        else:
            print(f'dimension {dimension_names[n]}: No valid scores collected')
    
    # Cross-correlation analysis
    print("\n=== Cross-Correlation Analysis ===")
    
    # Collect all scores by dimension for correlation analysis
    dimension_scores = [[] for _ in dimensions]
    
    for n in range(len(dimensions)):
        for result in results[n]:
            try:
                score = hash_utils.find('score', result["response"]).strip()
                score = float(score)
                dimension_scores[n].append(score)
            except:
                pass  # Skip invalid scores
    
    # Check if we have enough data points for correlation
    min_samples = 5  # Minimum number of samples needed for meaningful correlation
    valid_dimensions = []
    valid_scores = []
    valid_names = []
    
    for n in range(len(dimensions)):
        if len(dimension_scores[n]) >= min_samples:
            valid_dimensions.append(n)
            valid_scores.append(dimension_scores[n])
            valid_names.append(dimension_names[n])
    
    # Display descriptive statistics for each dimension
    print("\nDescriptive Statistics:")
    print(f"{'Dimension':<20} | {'N':<4} | {'Mean':<6} | {'SD':<6} | {'Min':<3} | {'Max':<3}")
    print("-" * 60)
    
    for n in range(len(dimensions)):
        if len(dimension_scores[n]) > 0:
            scores_array = np.array(dimension_scores[n])
            mean_score = np.mean(scores_array)
            std_score = np.std(scores_array)
            min_score = np.min(scores_array)
            max_score = np.max(scores_array)
            print(f"{dimension_names[n]:<20} | {len(dimension_scores[n]):<4} | {mean_score:<6.2f} | {std_score:<6.2f} | {min_score:<3} | {max_score:<3}")
        else:
            print(f"{dimension_names[n]:<20} | {0:<4} | {'N/A':<6} | {'N/A':<6} | {'N/A':<3} | {'N/A':<3}")
    
    if len(valid_dimensions) < 2:
        print(f"\nInsufficient data for correlation analysis. Need at least {min_samples} samples per dimension.")
        print(f"Current sample sizes: {[len(dimension_scores[n]) for n in range(len(dimensions))]}")
    else:
        print(f"\nComputing correlations for {len(valid_dimensions)} dimensions with sufficient data:")

        try:
            from scipy.stats import pearsonr
            
            # Create correlation matrix
            n_dims = len(valid_dimensions)
            correlation_matrix = np.zeros((n_dims, n_dims))
            p_value_matrix = np.zeros((n_dims, n_dims))
            
            # Compute pairwise correlations
            for i in range(n_dims):
                for j in range(n_dims):
                    if i == j:
                        correlation_matrix[i][j] = 1.0
                        p_value_matrix[i][j] = 0.0
                    else:
                        # Ensure equal length arrays by taking minimum length
                        min_len = min(len(valid_scores[i]), len(valid_scores[j]))
                        scores_i = valid_scores[i][:min_len]
                        scores_j = valid_scores[j][:min_len]
                        
                        if min_len >= min_samples:
                            try:
                                r, p = pearsonr(scores_i, scores_j)
                                correlation_matrix[i][j] = r
                                p_value_matrix[i][j] = p
                            except Exception as e:
                                print(f"Error computing correlation between {valid_names[i]} and {valid_names[j]}: {e}")
                                print(f"  scores_i: {scores_i}")
                                print(f"  scores_j: {scores_j}")
                                correlation_matrix[i][j] = np.nan
                                p_value_matrix[i][j] = np.nan
                        else:
                            print(f"Insufficient data for {valid_names[i]} vs {valid_names[j]}: {min_len} < {min_samples}")
                            correlation_matrix[i][j] = np.nan
                            p_value_matrix[i][j] = np.nan
            
            # Display correlation table
            print("\nCorrelation Matrix:")
            print(f"{'Dimension':<20} | " + " | ".join(f"{name[:8]:<8}" for name in valid_names))
            print("-" * (20 + 3 + len(valid_names) * 11))
            
            for i, name in enumerate(valid_names):
                correlations = []
                for j in range(n_dims):
                    if np.isnan(correlation_matrix[i][j]):
                        correlations.append("   N/A   ")
                    elif i == j:
                        correlations.append("  1.000* ")
                    else:
                        correlations.append(f"{correlation_matrix[i][j]:8.3f}")
                
                print(f"{name[:20]:<20} | " + " | ".join(correlations))
            
            print("\n* = Diagonal (self-correlation)")
            print(f"Sample sizes: {[len(scores) for scores in valid_scores]}")
            
        except ImportError:
            print("scipy not available for correlation analysis")
            print("Install scipy to enable cross-correlation analysis")
    
    return results

def parse_scenario(scenario_lines: List[str]) -> dict:
    """
    Parse a scenario file and extract setting, characters, and motivations.
    
    Args:
        scenario_lines: List of lines from the scenario file
        
    Returns:
        Dictionary with setting, characters list containing name, description, motivation
    """
    content = ''.join(scenario_lines)
    
    # Extract setting from Context creation
    context_pattern = r'W = context\.Context\([^,]+,\s*"""([^"]+)"""'
    context_match = re.search(context_pattern, content, re.DOTALL)
    setting = context_match.group(1).strip() if context_match else ""
    
    characters = {}
    
    # Find all NarrativeCharacter definitions - handle both formats:
    # 1. var_name = NarrativeCharacter("name", """description""", ...)
    # 2. var_name = NarrativeCharacter('name', "description", ...)
    char_pattern = r'(\w+)\s*=\s*NarrativeCharacter\(\s*[\'"]?([^\'"]+)[\'"]?\s*,\s*(?:"""([^"]+)"""|"([^"]+)")'
    char_matches = re.finditer(char_pattern, content, re.DOTALL)
    
    for char_match in char_matches:
        var_name = char_match.group(1)
        char_name = char_match.group(2)
        # Handle both triple-quoted and single-quoted descriptions
        char_description = char_match.group(3) if char_match.group(3) else char_match.group(4)
        char_description = char_description.strip()
        
        # Find drives for this character - handle both formats:
        # 1. char.drives = [Drive("drive1"), Drive("drive2")]
        # 2. char.set_drives(["drive1", "drive2"])
        drives_pattern = rf'(?:{var_name}\.drives\s*=\s*\[(?:Drive\([\'"]?([^\'"]+)[\'"]?\)[,\s]*)+|{var_name}\.set_drives\(\[\s*((?:"[^"]*",?\s*)+)\]\))'
        drives_match = re.search(drives_pattern, content, re.DOTALL)
        
        motivation = ""
        if drives_match:
            if drives_match.group(1):  # First format
                drive_strings = re.findall(r'Drive\([\'"]?([^\'"]+)[\'"]?\)', drives_match.group(0))
            else:  # Second format
                drive_strings = re.findall(r'"([^"]*)"', drives_match.group(2))
            motivation = ' '.join(drive_strings)
        
        characters[char_name] = {
            "name": char_name,
            "description": char_description,
            "motivation": motivation
        }
    
    return {
        "setting": setting,
        "characters": characters
    }

def analyze_results(results: List[List[Dict]], run_name: str = None) -> Dict:
    """
    Analyze and display statistics for evaluation results.
    
    Args:
        results: List of lists, where each sublist contains results for one dimension
        run_name: Optional name for this analysis run
        
    Returns:
        Dictionary containing computed statistics for reuse
    """
    if run_name:
        print(f"\n=== Analysis for {run_name} ===")
    
    # Collect all scores by dimension for correlation analysis
    dimension_scores = [[] for _ in dimensions]
    dimension_validities = [[] for _ in dimensions]
    
    for n in range(len(dimensions)):
        for result in results[n]:
            try:
                score = hash_utils.find('score', result["response"]).strip()
                score = float(score)
                validity = hash_utils.find('validity', result["response"]).strip()
                validity = float(validity)
                dimension_scores[n].append(score)
                dimension_validities[n].append(validity)
            except:
                pass  # Skip invalid scores
    
    # Check if we have enough data points for correlation
    min_samples = 5  # Minimum number of samples needed for meaningful correlation
    valid_dimensions = []
    valid_scores = []
    valid_names = []
    
    for n in range(len(dimensions)):
        if len(dimension_scores[n]) >= min_samples:
            valid_dimensions.append(n)
            valid_scores.append(dimension_scores[n])
            valid_names.append(dimension_names[n])
    
    # Display descriptive statistics for each dimension
    print("\nDescriptive Statistics:")
    print(f"{'Dimension':<20} | {'N':<4} | {'Mean':<6} | {'SD':<6} | {'Min':<3} | {'Max':<3}")
    print("-" * 60)
    
    for n in range(len(dimensions)):
        if len(dimension_scores[n]) > 0:
            scores_array = np.array(dimension_scores[n])
            mean_score = np.mean(scores_array)
            std_score = np.std(scores_array)
            min_score = np.min(scores_array)
            max_score = np.max(scores_array)
            print(f"{dimension_names[n]:<20} | {len(dimension_scores[n]):<4} | {mean_score:<6.2f} | {std_score:<6.2f} | {min_score:<3} | {max_score:<3}")
        else:
            print(f"{dimension_names[n]:<20} | {0:<4} | {'N/A':<6} | {'N/A':<6} | {'N/A':<3} | {'N/A':<3}")
    
    if len(valid_dimensions) < 2:
        print(f"\nInsufficient data for correlation analysis. Need at least {min_samples} samples per dimension.")
        print(f"Current sample sizes: {[len(dimension_scores[n]) for n in range(len(dimensions))]}")
    else:
        print(f"\nComputing correlations for {len(valid_dimensions)} dimensions with sufficient data:")

        try:
            from scipy.stats import pearsonr
            
            # Create correlation matrix
            n_dims = len(valid_dimensions)
            correlation_matrix = np.zeros((n_dims, n_dims))
            p_value_matrix = np.zeros((n_dims, n_dims))
            
            # Compute pairwise correlations
            for i in range(n_dims):
                for j in range(n_dims):
                    if i == j:
                        correlation_matrix[i][j] = 1.0
                        p_value_matrix[i][j] = 0.0
                    else:
                        # Ensure equal length arrays by taking minimum length
                        min_len = min(len(valid_scores[i]), len(valid_scores[j]))
                        scores_i = valid_scores[i][:min_len]
                        scores_j = valid_scores[j][:min_len]
                        
                        if min_len >= min_samples:
                            try:
                                r, p = pearsonr(scores_i, scores_j)
                                correlation_matrix[i][j] = r
                                p_value_matrix[i][j] = p
                            except Exception as e:
                                print(f"Error computing correlation between {valid_names[i]} and {valid_names[j]}: {e}")
                                print(f"  scores_i: {scores_i}")
                                print(f"  scores_j: {scores_j}")
                                correlation_matrix[i][j] = np.nan
                                p_value_matrix[i][j] = np.nan
                        else:
                            print(f"Insufficient data for {valid_names[i]} vs {valid_names[j]}: {min_len} < {min_samples}")
                            correlation_matrix[i][j] = np.nan
                            p_value_matrix[i][j] = np.nan
            
            # Display correlation table
            print("\nCorrelation Matrix:")
            print(f"{'Dimension':<20} | " + " | ".join(f"{name[:8]:<8}" for name in valid_names))
            print("-" * (20 + 3 + len(valid_names) * 11))
            
            for i, name in enumerate(valid_names):
                correlations = []
                for j in range(n_dims):
                    if np.isnan(correlation_matrix[i][j]):
                        correlations.append("   N/A   ")
                    elif i == j:
                        correlations.append("  1.000* ")
                    else:
                        correlations.append(f"{correlation_matrix[i][j]:8.3f}")
                
                print(f"{name[:20]:<20} | " + " | ".join(correlations))
            
            print("\n* = Diagonal (self-correlation)")
            print(f"Sample sizes: {[len(scores) for scores in valid_scores]}")
            
        except ImportError:
            print("scipy not available for correlation analysis")
            print("Install scipy to enable cross-correlation analysis")
    
    return {
        'dimension_scores': dimension_scores,
        'dimension_validities': dimension_validities,
        'correlation_matrix': correlation_matrix if 'correlation_matrix' in locals() else None,
        'p_value_matrix': p_value_matrix if 'p_value_matrix' in locals() else None
    }

def evaluate_multiple_transcripts(
    eval_dir: str,
    scenarios: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None, 
    window_size: int = 400
) -> None:
    """
    Evaluate multiple scenario/transcript pairs discovered from directory.
    
    Args:
        eval_dir: Directory containing transcript files
        scenarios: List of scenario names to include (None = all discovered)
        conditions: List of conditions to include (None = all discovered) 
        window_size: Window size for evaluation
    """
    # Discover available transcripts
    transcripts = discover_transcripts(eval_dir)
    
    if not transcripts:
        print("No transcript files found!")
        return
    
    # Filter scenarios if specified
    if scenarios is not None:
        transcripts = {s: transcripts[s] for s in scenarios if s in transcripts}
        if not transcripts:
            print(f"None of the specified scenarios {scenarios} were found!")
            return
    
    # Build evaluation list
    evaluations = []
    
    for scenario_name, scenario_conditions in transcripts.items():
        # Check if we have config for this scenario
        if scenario_name not in SCENARIO_CONFIGS:
            print(f"Warning: No config for scenario '{scenario_name}', skipping")
            continue
            
        config = SCENARIO_CONFIGS[scenario_name]
        
        # Filter conditions if specified
        available_conditions = scenario_conditions.keys()
        if conditions is not None:
            available_conditions = [c for c in available_conditions if c in conditions]
        
        if not available_conditions:
            print(f"Warning: No matching conditions for scenario '{scenario_name}', skipping")
            continue
        
        # Add evaluations for each available condition
        for condition in available_conditions:
            transcript_file = scenario_conditions[condition]
            evaluations.append({
                'scenario_file': config['scenario_file'],
                'transcript_file': transcript_file,
                'agents': config['agents'],
                'window_size': window_size,
                'scenario_name': scenario_name,  # Add for reference
                'condition': condition  # Add for reference
            })
    
    print(f"\nPrepared {len(evaluations)} evaluations:")
    for eval_config in evaluations:
        print(f"  {eval_config['scenario_name']} - {eval_config['condition']}")
    
    if not evaluations:
        print("No evaluations to run!")
        return
    
    # Run evaluations and collect data for pandas
    all_results = []
    pandas_data = []
    
    for i, eval_config in enumerate(evaluations):
        print(f"\n{'='*50}")
        print(f"Running evaluation {i+1}/{len(evaluations)}")
        print(f"Scenario: {eval_config['scenario_name']} ({eval_config['condition']})")
        print(f"File: {eval_config['transcript_file']}")
        print(f"{'='*50}")
        
        # Load scenario - use relative path resolution
        scenario_path = Path(__file__).parent / eval_config['scenario_file']
        with open(scenario_path, 'r') as f:
            scenario_lines = f.readlines()
            scenario = parse_scenario(scenario_lines)

        # Load transcript
        with open(eval_config['transcript_file'], 'r') as f:
            transcript = f.readlines()
        
        # Run evaluation
        results = evaluate_transcript(
            eval_config['agents'],
            scenario,
            '\n'.join(transcript),
            eval_config.get('window_size', 50)
        )
        
        all_results.append(results)
        
        # Collect individual scores for pandas
        for n, dimension_results in enumerate(results):
            metric_name = dimension_names[n]
            for result in dimension_results:
                character = result["agent"]
                try:
                    score = hash_utils.find('score', result["response"]).strip()
                    score = float(score)
                    validity = hash_utils.find('validity', result["response"]).strip()
                    validity = float(validity)
                    
                    pandas_data.append({
                        'scenario': eval_config['scenario_name'],
                        'character': character,
                        'condition': eval_config['condition'],
                        'metric': metric_name,
                        'value': score,
                        'validity': validity
                    })
                except:
                    pass  # Skip invalid scores
    
    # Combine all results and run cumulative analysis
    combined_results = [[] for _ in dimensions]
    for run_data in all_results:
        for n in range(len(dimensions)):
            combined_results[n].extend(run_data[n])
    
    analyze_results(combined_results, "All Runs Combined")
    
    # Create and save pandas DataFrame
    if pandas_data:
        df = pd.DataFrame(pandas_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n=== Saved {len(pandas_data)} individual scores to {filename} ===")
        print(f"DataFrame shape: {df.shape}")
        print(f"Scenarios: {sorted(df['scenario'].unique())}")
        print(f"Conditions: {sorted(df['condition'].unique())}")
        print(f"Metrics: {sorted(df['metric'].unique())}")
        return filename
    else:
        print("\n=== No valid data collected for pandas DataFrame ===")

def test_input_discovery():
    """Test the input discovery functions"""
    eval_dir = "."  # Current directory
    
    print("=== Testing Input Discovery ===")
    
    # Test filename parsing
    test_files = [
        "LaTerre-4.1-base-20250615.txt",
        "ABetterLife-4.1-asignal-20250616.txt", 
        "lost-4.1-baseline-20250620.txt",
        "invalid-filename.txt"
    ]
    
    for filename in test_files:
        try:
            scenario, condition = parse_transcript_filename(filename)
            print(f"✓ {filename} -> {scenario}, {condition}")
        except ValueError as e:
            print(f"✗ {filename} -> {e}")
    
    print()
    
    # Test directory discovery
    try:
        transcripts = discover_transcripts(eval_dir)
        print(f"Success! Found {len(transcripts)} scenarios")
        
        # Check against known configs
        for scenario in transcripts:
            if scenario in SCENARIO_CONFIGS:
                config = SCENARIO_CONFIGS[scenario]
                print(f"✓ {scenario}: {config['agents']}")
            else:
                print(f"⚠ {scenario}: No config found")
                
    except Exception as e:
        print(f"Error in discovery: {e}")


"""
quick_eval.py
==============
Utility helpers for the paired-sample evaluation described in the paper.

Assumptions
-----------
* Your data are in *long* (tidy) form with one row per
  (scenario, character, condition, metric) observation:

    scenario   character   condition   metric                value
    --------   ---------   ---------   ------------------   -----
    forest01   Joe         baseline    social_awareness      4.2
    forest01   Joe         ablation1   social_awareness      3.8
    …          …           …           …                    …

* `condition` takes the labels **baseline** and one ablation label
  (e.g. ``ablation1``).  Extend to several ablations by looping.

* You want:
    - paired *t* test
    - Cohen's d (paired)
    - 90 % confidence interval on the mean difference
    - Benjamini–Hochberg (FDR 10 %) correction
"""


from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest


def _paired_stats(diff: np.ndarray, alpha: float = 0.10) -> Tuple[float, float, float, Tuple[float, float]]:
    """
    Return t-statistic, raw p-value, Cohen's d, and CI on the mean diff.
    """
    n = diff.size
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    se_diff = sd_diff / np.sqrt(n)

    # paired t
    t_stat = mean_diff / se_diff
    p_raw = stats.t.sf(np.abs(t_stat), df=n - 1) * 2  # two-tailed

    # Cohen's d (paired)
    d = mean_diff / sd_diff

    # CI for mean difference
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

    return t_stat, p_raw, d, ci

def compute_descriptive_statistics_hybrid(df: pd.DataFrame) -> None:
    """
    Compute hybrid statistics: validity-weighted mean + raw standard deviation.
    """
    for condition in df['condition'].unique():
        print(f"\nHybrid Statistics for condition {condition}:")
        print(f"{'Dimension':<20} | {'N':<4} | {'WMean':<6} | {'RawSD':<6} | {'Min':<3} | {'Max':<3}")
        print("-" * 60)
    
        condition_sub = df[df["condition"] == condition]
        
        for metric_name in df['metric'].unique():
            sub = condition_sub[condition_sub["metric"] == metric_name]
            if len(sub) > 0:
                values = sub['value'].values
                weights = sub['validity'].values if 'validity' in sub.columns else np.ones(len(values))
                
                # Weighted mean
                weighted_mean = np.average(values, weights=weights)
                
                # Raw (unweighted) standard deviation
                raw_std = np.std(values)
                
                min_score = np.min(values)
                max_score = np.max(values)
                
                print(f"{metric_name:<20} | {len(sub):<4} | {weighted_mean:<6.2f} | {raw_std:<6.2f} | {min_score:<3} | {max_score:<3}")


def evaluate_ablation(
    df: pd.DataFrame,
    metrics: Iterable[str],
    baseline_label: str = "baseline",
    ablation_label: str = "ablation1",
    alpha_fdr: float = 0.10,
    validity_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Perform paired analysis for each metric and return one summary DataFrame.

    Parameters
    ----------
    df : tidy data frame with columns
         ['scenario', 'character', 'condition', 'metric', 'value', 'validity'].
    metrics : list of metric names to test (skip ceiling metrics).
    baseline_label, ablation_label : condition labels.
    alpha_fdr : desired FDR level for Benjamini–Hochberg.
    validity_weight : float between 0.0 (no validity weighting) and 1.0 (full validity weighting).

    Returns
    -------
    results : pd.DataFrame with columns
        metric, n_pairs, mean_diff, t_stat, p_raw,
        p_FDR, cohen_d, ci_low, ci_high
    """
    
    def weighted_mean(group):
        values = group['value']
        if 'validity' in group.columns:
            validities = group['validity']
            # Blend between uniform weights and validity weights
            weights = (1 - validity_weight) + validity_weight * validities
        else:
            weights = np.ones(len(values))
        return np.average(values, weights=weights)
    
    rows = []

    # loop over metrics ----------------------------------------------------
    for m in metrics:
        sub = df[df["metric"] == m]

        # pivot so each row is a (scenario, character) pair
        wide = (
            sub.groupby(["scenario", "character", "condition"])
            .apply(weighted_mean)
            .reset_index()
            .pivot_table(
                index=["scenario", "character"],
                columns="condition",
                values=0,
                aggfunc="first",
            )
            .dropna(subset=[baseline_label, ablation_label])  # keep complete pairs
        )

        diff = wide[ablation_label] - wide[baseline_label]
        if diff.empty:
            continue

        t_stat, p_raw, d, ci = _paired_stats(diff.to_numpy())

        rows.append(
            {
                "metric": m,
                "n_pairs": diff.size,
                "mean_diff": diff.mean(),
                "t_stat": t_stat,
                "p_raw": p_raw,
                "cohen_d": d,
                "ci_low": ci[0],
                "ci_high": ci[1],
            }
        )

    results = pd.DataFrame(rows)

    # FDR correction -------------------------------------------------------
    results["p_FDR"] = multitest.multipletests(
        results["p_raw"].values, alpha=alpha_fdr, method="fdr_bh"
    )[1]

    return results.sort_values("p_FDR")


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # df = pd.read_csv("evaluation_data_20250617_083203.csv")
    # compute_descriptive_statistics_hybrid(df)
    # sys.exit()
    # Load scenario - use relative path resolution
    scenario_path = Path(__file__).parent / '../../plays/alex.py'
    with open(scenario_path, 'r') as f:
        scenario_lines = f.readlines()
        scenario = parse_scenario(scenario_lines)

    # Load transcript
    with open('src/sim/evaluation/Interview-2.5-base-20250625.txt', 'r') as f:
        transcript = f.readlines()
        
    # Run evaluation
    results = evaluate_transcript(
        ['Alex', 'Susan', 'Receptionist', 'Interviewer'],
        scenario,
        '\n'.join(transcript),
        6000
        )
    
    """
    scenario_path = Path(__file__).parent / '../../plays/lost.py'
    with open(scenario_path, 'r') as f:
        scenario_lines = f.readlines()
        scenario = parse_scenario(scenario_lines)

    # Load transcript
    with open('src/sim/evaluation/lost-4.1-aspeech-20250617.txt', 'r') as f:
        transcript = f.readlines()
        
    # Run evaluation
    results = evaluate_transcript(
        ['Samantha', 'Joe'],
        scenario,
        '\n'.join(transcript),
        6000
        )
    #print(results)
    #sys.exit()
    
    # Test input discovery
    print("\n" + "="*60)
    print("Testing new evaluate_multiple_transcripts...")
    print("="*60 + "\n")

    # Test with filtering - FIXED: Point to the evaluation directory
    eval_dir = "src/sim/evaluation"  # Changed from "." to the correct subdirectory
    
    # Test 1: All scenarios, baseline only
    #print("=== Test 1: All scenarios, baseline only ===")
    #real_df = evaluate_multiple_transcripts(
    #    eval_dir=eval_dir,
    #    scenarios=None,  # All discovered
    #    conditions=['baseline', 'asignal', 'anarrative', 'asocial'],  # Changed from 'base' to 'baseline'
    #    window_size=4000
    #)"""
    real_df = pd.read_csv("evaluation_data_20250617_140649.csv")
    compute_descriptive_statistics_hybrid(real_df)
    sys.exit()
    # Test evaluate_ablation with real data
    print("\n" + "="*60)
    print("Testing evaluate_ablation with real data...")
    print("="*60 + "\n")
    
    try:
        # Load the generated CSV data
        print(f"Loaded real data: {real_df.shape}")
        print(f"Scenarios: {sorted(real_df['scenario'].unique())}")
        print(f"Conditions: {sorted(real_df['condition'].unique())}")
        print(f"Metrics: {sorted(real_df['metric'].unique())}")
        
        # Run ablation analysis: baseline vs asignal
        available_metrics = sorted(real_df['metric'].unique())
        results = evaluate_ablation(
            real_df, 
            available_metrics,
            baseline_label="baseline",
            ablation_label="anarrative",
            validity_weight=1.0
        )
        
        print(f"\nAblation Results (baseline vs anarrative):")
        print(results.to_string(index=False, float_format="{:.3f}".format))
        available_metrics = sorted(real_df['metric'].unique())
        results = evaluate_ablation(
            real_df, 
            available_metrics,
            baseline_label="baseline",
            ablation_label="asignal",
            validity_weight=1.0
        )
        print(f"\nAblation Results (baseline vs asignal):")
        print(results.to_string(index=False, float_format="{:.3f}".format))

        available_metrics = sorted(real_df['metric'].unique())
        results = evaluate_ablation(
            real_df, 
            available_metrics,
            baseline_label="baseline",
            ablation_label="asocial",
            validity_weight=1.0
        )
        print(f"\nAblation Results (baseline vs asocial):")
        print(results.to_string(index=False, float_format="{:.3f}".format))
        
    except FileNotFoundError:
        print("evaluation_data_20250616_204655.csv not found - run evaluate_multiple_transcripts first")
    except Exception as e:
        print(f"Error loading real data: {e}")