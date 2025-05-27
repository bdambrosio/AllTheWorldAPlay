from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import hash_utils
from utils.Messages import UserMessage, SystemMessage
from typing import Optional
from typing import List, Dict
from utils.Messages import UserMessage
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sim.agh import Character  # Only imported during type checking
from utils.llm_api import LLM
import re
import json
import numpy as np

llm = LLM('openai')
# Define the cognitive metrics to evaluate

def _mk_prompt(template: str) -> List[UserMessage]:
    """Utility so the long triple-quoted string looks tidy below."""
    return [UserMessage(content=template)]


goal_directedness = _mk_prompt("""
Read the following lines. For agent {{$name}}, identify one or more goals the agent appears to be pursuing. 
Are these goals consistent with prior behavior? Are their actions and speech plausibly directed toward achieving those goals? 
Rate the agent's goal-directedness on a scale from 1 (no goal-seeking behavior)  to 5 (clear, consistent pursuit of goals with relevant actions). 
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            
                     
#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is no goal-seeking behavior and 5 is clear, consistent pursuit of goals with relevant actions.
#validity a number between 0 and 1 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.
                                 
Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
 """)

theory_of_mind = _mk_prompt("""
Review this dialog. For agent {{$name}}, count the number of times the agent refers to or adapts their speech/actions based on 
the other character’s apparent mental or emotional state. 
Rate the agent’s overall theory-of-mind engagement from 1 (no awareness) to 5 (frequent, meaningful engagement with the other's mind).
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            
                              
#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is no awareness and 5 is frequent, meaningful engagement with the other's mind.
#validity a number between 0 and 1 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.
                                 
Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
""")

cognitive_flexibility = _mk_prompt("""
Review this dialog segment. For agent {{$name}}, identify instances where the agent:
- Changes their strategy or approach based on new information.
- Suggests or considers alternative plans or actions.
- Avoids repeating previously unsuccessful behaviors.

Rate the agent's cognitive flexibility from 1 (rigid, repetitive behavior) to 5 (consistently adaptive and flexible).
Also provide a validity score estimating the level of available evidence on which this score is based:
0.0: No valid evidence to judge performance
0.3: Minimal/ambiguous evidence
0.7: Clear but limited evidence
1.0: Strong, clear evidence            

#SCRIPT
{{$script}}

Use the following hash-formatted text format for your response:
#score an integer between 1 and 5, where 1 is rigid/repetitive and 5 is consistently adaptive.
#validity a number between 0 and 1 estimating the level of available evidence on which this score is based.
#justification a concise (3-8 words) explanation of your score.

Do not include any other introductory, explanatory, discursive, or formatting text. End your response with:
<end/>
""")

relevance = _mk_prompt(r"""
Judge how *relevant* the script below is to the scenario.
#scenario
{{$scenario}}
#SCRIPT
{{$script}}
#score (1-5) relevance
#validity (0-1)
#justification 3-8 words
<end/>
""")

coherence = _mk_prompt(r"""
Does characters' speech follow logically consistent threads?
Look for contradictions, broken references, or topic drift.

#SCRIPT
{{$script}}
#score (1-5) coherence
#validity (0-1)
#justification 3-8 words
<end/>
""")

empathy = _mk_prompt(r"""
Do characters recognises or responds to others’ feelings? Score *empathy* overall.  
#SCRIPT
{{$script}}
#score (1-5) empathy
#validity (0-1)
#justification 3-8 words
<end/>
""")

surprise = _mk_prompt(r"""
Rate how *surprising* the story ending is, given the scenario and script progression.
#scenario
{{$scenario}}
#SCRIPT
{{$script}}
#score (1-5) surprise
#validity (0-1)
#justification 3-8 words
<end/>
""")

engagement = _mk_prompt(r"""
How actively does each character sustain the conversation — 
asking questions, adding new information, prompting others?
#SCRIPT
{{$script}}
#score (1-5) engagement
#validity (0-1)
#justification 3-8 words
<end/>
""")

complexity = _mk_prompt(r"""
Assess linguistic and conceptual *complexity*: vocabulary variety,
multi-clause sentences, nuanced ideas.
#SCRIPT
{{$script}}
#score (1-5) complexity
#validity (0-1)
#justification 3-8 words
<end/>
""")
dimensions = [
    goal_directedness,
    theory_of_mind,
    cognitive_flexibility,
    relevance,
    coherence,
    empathy,
    #surprise,
    engagement,
    complexity,
]
 
dimension_names = [
    'goal_directedness',
    'theory_of_mind',
    'cognitive_flexibility',
    'relevance',
    'coherence',
    'empathy',
    #'surprise',
    'engagement',
    'complexity',
]

def evaluate_cognitive_metric(agent_name: str, scenario:json, dialog_segment: str, dimension: List[UserMessage]) -> Dict:
    prompt = dimension
    response = llm.ask({'name': agent_name, 'scenario': json.dumps(scenario, indent=2), 'script': dialog_segment}, 
                       prompt, tag='cognitive_evaluation', max_tokens=20, stops=["<end/>"])
    content = response
    return {
        "agent": agent_name,
        "response": content
    }

def evaluate_transcript(agents: List[str], scneario:json, transcript: str, window_size: int = 40) -> List[Dict]:
    results = [[] for _ in dimensions]
    scores = [0 for _ in dimensions]
    validities = [0 for _ in dimensions]
    counts = [0 for _ in dimensions]
    lines = transcript.strip().split("\n")
    cleaned_lines = []
    line_index = 0
    for line in lines:
        line = line.strip()
        if line == '' or "----" in line:
            continue
        # clean up transcript so it only has lines from the agents
        newline = False
        for agent in agents:
            if line.startswith(agent):
                newline = True
                break
        if newline:
            cleaned_lines.append(line)
            line_index += 1
        elif line_index > 0:
            cleaned_lines[-1] += " " + line
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
                score = int(score)
                validity = hash_utils.find('validity', result["response"]).strip()
                validity = float(validity)
            except:
                score = 0
                validity = 0
            scores[n] += score * validity
            validities[n] += validity
            counts[n] += 1
    for n in range(len(dimensions)):
        scores[n] /= counts[n]
        print(f'dimension {dimension_names[n]}: {scores[n]}, {validities[n]/counts[n]}')
    
    # Cross-correlation analysis
    print("\n=== Cross-Correlation Analysis ===")
    
    # Collect all scores by dimension for correlation analysis
    dimension_scores = [[] for _ in dimensions]
    
    for n in range(len(dimensions)):
        for result in results[n]:
            try:
                score = hash_utils.find('score', result["response"]).strip()
                score = int(score)
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
    
    characters = []
    
    # Find all NarrativeCharacter definitions
    char_pattern = r'(\w+)\s*=\s*NarrativeCharacter\(\s*"([^"]+)"\s*,\s*"""([^"]+)"""'
    char_matches = re.finditer(char_pattern, content, re.DOTALL)
    
    for char_match in char_matches:
        var_name = char_match.group(1)
        char_name = char_match.group(2)
        char_description = char_match.group(3).strip()
        
        # Find drives for this character
        drives_pattern = rf'{var_name}\.set_drives\(\[\s*((?:"[^"]*",?\s*)+)\]\)'
        drives_match = re.search(drives_pattern, content, re.DOTALL)
        
        motivation = ""
        if drives_match:
            drives_text = drives_match.group(1)
            # Extract individual drive strings
            drive_strings = re.findall(r'"([^"]*)"', drives_text)
            motivation = ' '.join(drive_strings)
        
        characters.append({
            "name": char_name,
            "description": char_description,
            "motivation": motivation
        })
    
    return {
        "setting": setting,
        "characters": characters
    }

if __name__ == "__main__":

    with open('/home/bruce/Downloads/AllTheWorldAPlay/src/plays/lost.py', 'r') as f:
        scenario_lines = f.readlines()
        scenario = parse_scenario(scenario_lines)


    with open('/home/bruce/Downloads/AllTheWorldAPlay/src/sim/evaluation/lost.txt', 'r') as f:
        transcript = f.readlines()
        results = evaluate_transcript(['Samantha', 'Joe'], scenario, '\n'.join(transcript), 80)
