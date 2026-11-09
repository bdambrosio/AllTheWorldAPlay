"""June 2025 per-character rubric prompts, verbatim from the deleted
src/sim/evaluation/cognitiveEvaluation.py (git tag pre-cleanup), kept for continuity with the
ablation transcripts in docs/research/. Used by tools/judges.py.

Bindings: scenario, character_description, name, script. Each returns #score (1-5),
#validity (0-1) and #justification, ending with <end/>.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from typing import List
from utils.Messages import UserMessage, SystemMessage


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


RUBRICS_2025 = {
    'goal_directedness': goal_directedness,
    'social_awareness': social_awareness,
    'cognitive_flexibility': cognitive_flexibility,
    'coherence': coherence,
    'engagement': engagement,
    'complexity': complexity,
    'character_development': character_development,
    'action_effectiveness': action_effectiveness,
    # dialogue_quality was commented out of the 2025 runs; kept available
    'dialogue_quality': dialogue_quality,
}
