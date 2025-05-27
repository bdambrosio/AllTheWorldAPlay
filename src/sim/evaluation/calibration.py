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
from utils.llm_api import LLM
import sys, csv, os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr            # you already have scipy for NINA, right?

from utils import hash_utils
from utils.llm_api import LLM
from utils.Messages import UserMessage

# ----------------------------------------------------------------------
# =====  PROMPT TEMPLATES  (identical to the ones we added earlier) ====
# ----------------------------------------------------------------------
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
Judge how *relevant* the script below is to the story prompt.
#PROMPT
{{$prompt}}
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
Rate how *surprising* the story ending is, given the story prompt and script progression.
#PROMPT
{{$prompt}}
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
    surprise,
    engagement,
    complexity,
]
 


DIMENSIONS = {
    "Relevance":   relevance,
    "Coherence":   coherence,
    "Empathy":     empathy,
    "Surprise":    surprise,
    "Engagement":  engagement,
    "Complexity":  complexity,
}

# ----------------------------------------------------------------------
# =====  CORE EVALUATION ROUTINES  =====================================
# ----------------------------------------------------------------------
llm = LLM("openai")        # or whatever backend string you use

def _ask_llm(story: str, prompt: List[UserMessage]) -> int:
    """Send the story to the model and pull the #score integer back out."""
    resp = llm.ask({"name": "Narrator", "prompt":prompt, "script": story},
                   prompt,
                   tag="hanna_metric",
                   max_tokens=20,
                   stops=["<end/>"])
    # try/except keeps the loop alive if the LLM glitches
    try:
        return int(hash_utils.find("score", resp).strip())
    except Exception:
        return np.nan       # easy to drop later

def evaluate_story(row: pd.Series) -> Dict[str, float]:
    story_text = row["Story"]
    scores: Dict[str, float] = {}
    for dim_name, prompt in DIMENSIONS.items():
        scores[dim_name] = _ask_llm(story_text, prompt)
    return scores

# ----------------------------------------------------------------------
# =====  DRIVER ========================================================
# ----------------------------------------------------------------------
def main(path: str) -> None:
    # First, let's examine the raw file structure
    print("=== EXAMINING RAW FILE ===")
    with open(path, 'r') as f:
        first_few_lines = [f.readline().strip() for _ in range(2)]
    for i, line in enumerate(first_few_lines):
        print(f"Line {i}: {line[:200]}...")  # First 200 chars
    
    df = pd.read_csv(
        path,
        sep=",",
        quoting=csv.QUOTE_MINIMAL,  # Respect quotes around fields with commas
        dtype=str,                  # let pandas keep everything as string first
        on_bad_lines="skip",
    )
    
    print("\n=== PARSED DATAFRAME ===")
    print("DataFrame columns:", list(df.columns))
    print("DataFrame shape:", df.shape)
    print("DIMENSIONS keys:", list(DIMENSIONS.keys()))

    # Filter to human stories only (Story ID 0-95)
    df['Story ID'] = pd.to_numeric(df['Story ID'], errors='coerce')
    human_stories = df[df['Story ID'] <= 95].copy()
    print(f"Filtered to human stories: {len(human_stories)} rows")

    # Temporary: reduce to subset for testing
    max_story_id = min(50, human_stories['Story ID'].max())  # Test with first 10 stories
    human_stories = human_stories[human_stories['Story ID'] <= max_story_id]
    print(f"Testing subset: Story IDs 0-{max_story_id}")

    # Debug: Check raw data before conversion
    print("Raw data sample (first few rows, score columns only):")
    score_cols = [col for col in DIMENSIONS.keys() if col in human_stories.columns]
    #print(human_stories[['Story ID'] + score_cols].head(10))
    
    print("\nRaw string values (before any conversion):")
    for i in range(min(2, len(human_stories))):
        row = human_stories.iloc[i]
        print(f"Row {i}: Story ID = '{row['Story ID']}'")
        for col in score_cols:
            print(f"  {col} = '{row[col]}' (type: {type(row[col])})")
        print()
    
    # Convert crowd score columns to numeric
    for col in DIMENSIONS.keys():
        if col in human_stories.columns:
            print(f"Converting {col}: sample raw values = {human_stories[col].head(3).tolist()}")
            human_stories[col] = pd.to_numeric(human_stories[col], errors="coerce")
            print(f"After conversion: sample values = {human_stories[col].head(3).tolist()}")
        else:
            print(f"Warning: Column '{col}' not found in dataset")

    # Debug: Check data before grouping
    print("Data before grouping (sample):")
    sample_story = human_stories[human_stories['Story ID'] == human_stories['Story ID'].iloc[0]]
    print(f"Story ID {sample_story['Story ID'].iloc[0]} has {len(sample_story)} rows:")
    for col in DIMENSIONS.keys():
        if col in sample_story.columns:
            values = sample_story[col].tolist()
            print(f"  {col}: {values} -> mean = {sample_story[col].mean():.2f}")
    
    # Group by Story ID and average crowd scores
    crowd_scores = human_stories.groupby('Story ID')[list(DIMENSIONS.keys())].mean()
    print(f"Crowd scores shape: {crowd_scores.shape}")
    print("Sample crowd scores:")
    print(crowd_scores.head())
    print("Crowd score ranges:")
    for col in DIMENSIONS.keys():
        if col in crowd_scores.columns:
            print(f"  {col}: {crowd_scores[col].min():.2f} - {crowd_scores[col].max():.2f}")

    # Get unique stories for evaluation (one per Story ID)
    unique_stories = human_stories.drop_duplicates('Story ID')[['Story ID', 'Human']].set_index('Story ID')
    print(f"Unique stories for evaluation: {len(unique_stories)}")

    # Evaluate each unique story once
    def evaluate_unique_story(row):
        story_text = row['Human']
        scores = {}
        for dim_name, prompt in DIMENSIONS.items():
            scores[dim_name] = _ask_llm(story_text, prompt)
        return pd.Series(scores)

    computed = unique_stories.apply(evaluate_unique_story, axis=1)
    computed.columns = [f"our_{c}" for c in computed.columns]
    print(f"Computed scores shape: {computed.shape}")
    print("Sample computed scores:")
    print(computed.head())
    print("Computed score ranges:")
    for col in computed.columns:
        print(f"  {col}: {computed[col].min():.2f} - {computed[col].max():.2f}")

    # Combine crowd and computed scores
    df_combined = pd.concat([crowd_scores, computed], axis=1)
    print(f"Combined dataframe shape: {df_combined.shape}")

    # descriptive statistics -----------------------------------------------
    print("Descriptive Statistics:")
    for dim in DIMENSIONS.keys():
        if dim in df_combined.columns:
            crowd = df_combined[dim].astype(float)
            ours = df_combined[f"our_{dim}"].astype(float)
            crowd_clean = crowd.dropna()
            ours_clean = ours.dropna()
            
            print(f"  {dim:11}:")
            print(f"    Crowd: mean={crowd_clean.mean():.2f}, sd={crowd_clean.std():.2f}, n={len(crowd_clean)}")
            print(f"    Ours:  mean={ours_clean.mean():.2f}, sd={ours_clean.std():.2f}, n={len(ours_clean)}")

    # correlation matrix -----------------------------------------------
    print("\nCorrelation Matrix (Our dimensions vs All crowd dimensions):")
    
    # Get all crowd dimensions that exist in the data
    all_crowd_dims = [col for col in df_combined.columns if col in ['Relevance', 'Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity']]
    our_dims = [f"our_{dim}" for dim in DIMENSIONS.keys() if f"our_{dim}" in df_combined.columns]
    
    print(f"{'Our Dimension':<15} | " + " | ".join(f"{dim:<11}" for dim in all_crowd_dims))
    print("-" * (15 + 3 + len(all_crowd_dims) * 14))
    
    for our_dim in our_dims:
        our_name = our_dim.replace("our_", "")
        correlations = []
        
        for crowd_dim in all_crowd_dims:
            crowd = df_combined[crowd_dim].astype(float)
            ours = df_combined[our_dim].astype(float)
            mask = (~crowd.isna()) & (~ours.isna())
            
            if mask.sum() < 3:
                correlations.append("N/A")
            else:
                r, p = pearsonr(crowd[mask], ours[mask])
                # Highlight diagonal (matching dimensions) with asterisk
                if crowd_dim == our_name:
                    correlations.append(f"{r:6.3f}*")
                else:
                    correlations.append(f"{r:7.3f}")
        
        print(f"{our_name:<15} | " + " | ".join(f"{corr:<11}" for corr in correlations))
    
    print("\n* = Matching dimension (diagonal)")
    
    # Original simple correlation for reference
    print("\nSimple correlations (matching dimensions only):")
    for dim in DIMENSIONS.keys():
        if dim not in df_combined.columns:
            print(f"  {dim:11}:  N/A  (column not found)")
            continue
        crowd = df_combined[dim].astype(float)
        ours  = df_combined[f"our_{dim}"].astype(float)
        mask  = (~crowd.isna()) & (~ours.isna())
        if mask.sum() < 3:   # not enough data
            print(f"  {dim:11}:  N/A  (too few points)")
            continue
        r, p = pearsonr(crowd[mask], ours[mask])
        print(f"  {dim:11}:  r = {r:6.3f}   p = {p:.3g}")

    # optional: save the dataframe so you can inspect per-story deltas
    out = Path(path).with_suffix(".scored.tsv")
    df_combined.to_csv(out, sep="\t", index=True)  # index=True to save Story ID
    print(f"\nFull table with our scores written to {out}")

# ----------------------------------------------------------------------
main("/home/bruce/Downloads/hanna-benchmark-asg/hanna_stories_annotations.csv")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hanna_eval.py <hanna_dataset.tsv>")
        sys.exit(1)
    main(sys.argv[1])
