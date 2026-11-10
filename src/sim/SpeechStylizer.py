from __future__ import annotations
import asyncio
import importlib
import logging
from typing import TYPE_CHECKING
import os, sys, re, traceback, requests, json
import numpy as np
from sentence_transformers import SentenceTransformer

from sim.cognitive.EmotionalStance import Arousal, Orientation, Tone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from typing import Dict, List, Optional
from utils.Messages import SystemMessage, UserMessage
import utils.hash_utils as hash_utils
#from sim.cognitive.EmotionalStance import EmotionalStance

if TYPE_CHECKING:
    from sim.narrativeCharacter import NarrativeCharacter
    from sim.cognitive.knownActor import KnownActor
logger = logging.getLogger('simulation_core')
from utils.embedding_model import get_embedding_model
_embedding_model = get_embedding_model()
_embedding_model.max_seq_length = 384
_voice_embeddings = None
_voices = None

def get_voice_embedding(voice):
    try:
        description = voice.get('description', '')
        if description is None:
            description = ' '
        labels = voice.get('labels', {})
        description = description + ' ' + labels.get('description', '')
        return _embedding_model.encode(description.strip().lower())
    except Exception as e:
        logger.error(f"Error getting voice embedding for {voice}: {e}")
        return _embedding_model.encode(' none ')

class SpeechStylizer:
    """A character's speaking voice.

    voice_line(): one fixed line on how the character speaks (tone, register, habits), read
    once from the character description. It goes into the character's own description, so
    every prompt that writes its speech sees the same voice.

    voice_params(): text-to-speech settings for an utterance, when voices are on. They vary
    with emotion, relationship and scene; the words do not. (Each Say used to be rewritten by
    a second LLM call under a style whose formality, metaphor and 'oddity' were re-rolled per
    line: the same character swung between clipped slang and archaic formality.)
    """

    # ---------------------------------------------------------------------
    def __init__(self, character:NarrativeCharacter, global_variability: float = 0.5, style_rotation_period: int = 3, rng: Optional[random.Random] = None) -> None:
        global _voice_embeddings, _voices
        self.char: NarrativeCharacter = character
        self.global_var = max(0.0, min(1.0, global_variability))
        self.rotation_period = max(1, style_rotation_period)
        self.rng = rng or random.Random()

        self._base_style = self._extract_personality_style()
        if _voice_embeddings is None:
            _voices = self.char.context.voice_service.get_voices()
            _voice_embeddings = [get_voice_embedding(voice) for voice in _voices if get_voice_embedding(voice) is not None]
        self.voices = _voices
        self.voice_embeddings = _voice_embeddings
        self.voice_id = None
        if _voices:  # none without an ElevenLabs key (NO_TTS): not an error
            try:
                self.voice_id = self.pick_best_voice(_voices)
            except Exception as e:
                logger.error(f"Error picking best voice: {e}")
        self._call_counter = 0
        self.relationship_cache = {}
        self.relationships_last_refreshed = 0

    # =====================================================================
    # PUBLIC API
    # ---------------------------------------------------------------------
    def voice_line(self) -> str:
        """How this character speaks, in a line: the same for the whole play."""
        style = self._base_style
        formality = style.get("formality", 0.5)
        quirks = style.get("lexical_quirks", {})
        parts = [', '.join(style.get("tone", [])[:2]),
                 'formal register' if formality >= 0.7 else 'casual register' if formality < 0.35 else 'everyday register']
        if quirks.get("slang", 0.0) >= 0.5:
            parts.append('uses slang')
        parts.append('fond of a metaphor' if quirks.get("metaphor", 0.0) >= 0.6 else 'plain words, rarely a figure of speech')
        if quirks.get("profanity"):
            parts.append('swears')
        return '; '.join(p for p in parts if p)

    def voice_params(self, target_character=None) -> Dict:
        """Text-to-speech settings for the next utterance; {} when voices are off."""
        if self.voice_id is None:
            return {}
        return self.to_elevenlabs_params(self._compose_style_dict(target_character=target_character))

    # =====================================================================
    # INTERNAL ORCHESTRATION
    # ---------------------------------------------------------------------
    def _compose_style_dict(self, target_character=None, scene_context: Optional[dict] = None, dominant_theme: Optional[str] = None,) -> Dict:
        """Fuse baseline, dynamic factors, theme, and stochastic jitter."""

        self._call_counter += 1

        # ---------------- start with immutable baseline -------------------
        style = {
            "tone": list(self._base_style["tone"]),
            "formality": self._base_style["formality"],
            "lexical_quirks": dict(self._base_style["lexical_quirks"]),
            "syntactic_oddity": 0.0,
        }

        # ---------------- dynamic: emotion --------------------------------
        style = self._merge(style, self._from_emotion())

        # ---------------- dynamic: relationship ---------------------------
        if target_character is not None:
            style = self._merge(style, self._from_relationship(target_character))

        # ---------------- scene factors (tension etc.) --------------------
        scene = scene_context or getattr(self.char, "current_scene", None) or getattr(self.char.context, "current_scene", None) or {}
        style = self._merge(style, self._from_scene(scene))

        # ---------------- theme parsing (high‑level vibe) -----------------
        act = (
            getattr(self.char, "current_act", {}) or getattr(self.char.context, "current_act", {}) or {}
            or ""
        )
        style = self._merge(style, self._from_theme(self.char.context.current_state))

        # ---------------- stochastic spice --------------------------------
        spice_factor = self._scheduled_spice()
        style = self._apply_jitter(style, spice_factor)

        # ---------------- cleanup ----------------------------------------
        style["tone"] = list(set(style["tone"]))
        style["formality"] = float(min(1.0, max(0.0, style["formality"])))
        style["syntactic_oddity"] = float(
            min(1.0, max(0.0, style["syntactic_oddity"]))
        )
        style["lexical_quirks"]["slang"] = float(
            min(1.0, max(0.0, style["lexical_quirks"].get("slang", 0.0)))
        )
        return style

    # =====================================================================
    # LOW‑LEVEL INGESTION FUNCTIONS
    # ---------------------------------------------------------------------
    def _extract_personality_style(self) -> Dict:
        """Extract personality-based speech style using LLM."""

        prompt = [UserMessage(content="""You are analyzing the personality of {{$name}}.

#Character Description
{{$character_description}}

#Drives
{{$drives}}

Based on these descriptions, analyze the character's speech style and provide a JSON response with these exact keys:
- tone: most dominant, stable, tones (e.g. gruff, warm, polite, casual, thoughtful, comic, etc.) likely present in character's speech (limit to 1 -2 tones)
- formality: float between 0-1 (0 being very informal, 1 being very formal)
- lexical_quirks: dictionary containing:
  - slang: float 0-1 indicating likelihood of using slang
  - profanity: boolean indicating if they would use profanity
  - metaphor: float 0-1 indicating likelihood of using metaphorical language

Format your response as valid JSON only, no other text.
""")]

        response = self.char.llm.ask({"name": self.char.name, 
                                      "character_description": self.char.character, 
                                      "drives": '\n'.join([f'{d.id}: {d.text}; activation: {d.activation:.2f}' for d in self.char.drives])}, 
                                      prompt, tag='extract_style', max_tokens=150)
        
        try:
            style = json.loads(response.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            style = self.char.context.repair_json(response, e)
        if style is None:
            return {
                "tone": [],
                "formality": 0.5,
                "lexical_quirks": {"slang": 0.0, "profanity": False, "metaphor": 0.0}
            }
        # Ensure we have all required keys with defaults
        tone = style.get("tone") or []
        if isinstance(tone, str):   # "gruff, warm": list() of it made a tone of every letter
            tone = re.split(r'\s*[,;/]\s*|\s+and\s+', tone)
        style["tone"] = [t.strip() for t in tone if isinstance(t, str) and t.strip()]
        style.setdefault("formality", 0.5)
        style.setdefault("lexical_quirks", {})
        style["lexical_quirks"].setdefault("slang", 0.0)
        style["lexical_quirks"].setdefault("profanity", False)
        style["lexical_quirks"].setdefault("metaphor", 0.0)
        return style

    # ---------------------------------------------------------------------
    def _from_emotion(self) -> Dict:
        #ranked_signalClusters = self.char.driveSignalManager.get_scored_clusters()
        #focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order
        emotionalState = self.char.emotionalStance        
        style = {"tone": [], "syntactic_oddity": 0.0}
        
        # Map arousal to syntactic oddity
        if emotionalState.arousal == Arousal.Agitated:
            style["syntactic_oddity"] = 0.4
        elif emotionalState.arousal == Arousal.Vigilant:
            style["syntactic_oddity"] = 0.25
        elif emotionalState.arousal == Arousal.Exhausted:
            style["syntactic_oddity"] = 0.15
            
        # Map tone to emotional qualities
        tone_mapping = {
            Tone.Angry: ["angry", "sharp"],
            Tone.Fearful: ["nervous", "anxious"],
            Tone.Anxious: ["nervous", "worried"],
            Tone.Sad: ["melancholic", "subdued"],
            Tone.Disgusted: ["disdainful", "cold"],
            Tone.Surprised: ["excited", "animated"],
            Tone.Curious: ["interested", "engaged"],
            Tone.Joyful: ["buoyant", "cheerful"],
            Tone.Content: ["calm", "satisfied"]
        }
        
        if emotionalState.tone in tone_mapping:
            style["tone"].extend(tone_mapping[emotionalState.tone])
            
        # Map orientation to additional tone qualities
        orientation_mapping = {
            Orientation.Controlling: ["authoritative"],
            Orientation.Challenging: ["defiant"],
            Orientation.Appeasing: ["conciliatory"],
            Orientation.Avoiding: ["distant"],
            Orientation.Supportive: ["warm"],
            Orientation.Seekingsupport: ["vulnerable"],
            Orientation.Connecting: ["open"],
            Orientation.Performing: ["animated"],
            Orientation.Observing: ["measured"],
            Orientation.Defending: ["guarded"]
        }
        
        if emotionalState.orientation in orientation_mapping:
            style["tone"].extend(orientation_mapping[emotionalState.orientation])
            
        return style

    # ---------------------------------------------------------------------
    def _from_relationship(self, target: NarrativeCharacter) -> Dict:
        """Analyze relationship text to determine speech style adjustments."""
        if not target:
            return {"tone": [], "formality": 0.5}
        
        # Get the relationship text from the character's known actor model
        known_actor: KnownActor = self.char.actor_models.get_actor_model(target.name)
        if not known_actor:
            return {"tone": [], "formality": 0.5}
        
        if type(known_actor) == str:
            logger.error(f"KnownActor {target.name} is not a KnownActor: {known_actor}")
        else:
            relationship = known_actor.relationship

        if self.relationship_cache.get(target.name) is not None and self.relationships_last_refreshed < 3:
            self.relationships_last_refreshed += 1
            return self.relationship_cache[target.name]
        
        prompt = [
            SystemMessage(content="""You are a dialogue style analyzer.
Given a relationship description between two characters, determine how it should affect their speech style.
Focus on formality level and emotional tone.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Close the hash-formatted text with ##  on a separate line, as shown below.
be careful to insert line breaks only where shown, separating a value from the next tag:

#tone list of emotional qualities
#formality float 0-1 (0 being very informal, 1 being very formal)
##

End your response with:
</end>
"""),
            UserMessage(content=f"""Analyze this relationship description to determine speech style:

#Character {self.char.name}
#Target {target.name}
#Relationship {relationship.replace('\n', ' ')}


Again, provide your analysis as a hash-formatted text with these keys:

#tone list of emotional qualities
#formality float 0-1 (0 being very informal, 1 being very formal)
##

End your response with:
</end>
""")
        ]

        style = {"tone": [], "formality": 0.5}
        try:
            style = {"tone": [], "formality": 0.5}
            response = self.char.llm.ask({}, prompt, tag='relationship_style', max_tokens=150)
            if response is None:
                return style
            tone = hash_utils.findList('tone', response)
            if tone is not None and len(tone) > 0:
                style["tone"] = tone
            formality = hash_utils.find('formality', response)
            try:
                style["formality"] = float(formality.strip())
            except ValueError:
                pass
            self.relationship_cache[target.name] = style
            self.relationships_last_refreshed = 0
            return style
        except:
           logger.error(f"Error analyzing relationship style for {self.char.name} and {target.name}: {traceback.format_exc()}")
        return style

    # ---------------------------------------------------------------------
    def _from_scene(self, scene: Dict) -> Dict:

        prompt = [
            SystemMessage(content="""You are a dialogue style analyzer.
Given a scene description, determine how it should affect the character's speech style.
Focus on emotional tone and syntactic oddity.

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Close the hash-formatted text with ##  on a separate line, as shown below.
be careful to insert line breaks only where shown, separating a value from the next tag:
                        
#tone list of emotional qualities
#syntactic_oddity float 0-1 (0 being very normal, 1 being very odd)
##

"""),
            UserMessage(content="""Analyze this scene description to determine speech style:

Character: {{$character_name}}
Scene: {{$scene}}

Respond using the following hash-formatted text, where each tag is preceded by a # and followed by a single space, followed by its content.
Close the hash-formatted text with ##  on a separate line, as shown below.
be careful to insert line breaks only where shown, separating a value from the next tag:
                        
#tone list of emotional qualities
#syntactic_oddity float 0-1 (0 being very normal, 1 being very odd)
##

End your response with:
</end>
""")
        ]
        style = {"tone": [], "syntactic_oddity": 0.0}
        scene_str = f"Goal: {scene.get('goal', '')}\nPre-Narrative: {scene.get('pre_narrative', '')}\nPost-Narrative: {scene.get('post_narrative', '')}"
        response = self.char.llm.ask({"character_name": self.char.name, "scene": scene_str}, prompt, tag='scene_style', max_tokens=150)
        if response is None:
            return style
        tone = hash_utils.findList('tone', response)
        if tone is not None and len(tone) > 0:
            style["tone"] = tone
        syntactic_oddity = hash_utils.find('syntactic_oddity', response)
        if syntactic_oddity is not None and len(syntactic_oddity) > 0:
            try:
                style["syntactic_oddity"] = float(syntactic_oddity.strip())
            except ValueError:
                pass
        return style

    # ---------------------------------------------------------------------
    def _from_theme(self, theme_str: str) -> Dict:
        theme = theme_str.lower()
        style = {"tone": []}
        keywords = {
            "inspired": "enthusiastic",
            "grounded": "calm",
            "hopeful": "hopeful",
            "connected": "warm",
            "positive": "upbeat",
        }
        for key, tone in keywords.items():
            if key in theme:
                style["tone"].append(tone)
        return style

    # ---------------------------------------------------------------------
    def _scheduled_spice(self) -> float:
        """Return a jitter multiplier that *occasionally* spikes."""
        base = self.global_var
        if self._call_counter % self.rotation_period == 0:
            base *= 1.5  # short burst of extra spice
        return min(1.0, base)

    # ---------------------------------------------------------------------
    def _apply_jitter(self, style: Dict, magnitude: float) -> Dict:
        j = magnitude
        if j == 0:
            return style

        def jit(x):
            return max(0.0, min(1.0, x * (1 + self.rng.uniform(-j, j))))

        style["formality"] = jit(style["formality"])
        style["syntactic_oddity"] = jit(style["syntactic_oddity"])
        slang = style["lexical_quirks"].get("slang", 0.0)
        style["lexical_quirks"]["slang"] = jit(slang)
        metaphor = style["lexical_quirks"].get("metaphor", 0.0)
        style["lexical_quirks"]["metaphor"] = jit(metaphor)
        return style

    # ---------------------------------------------------------------------
    @staticmethod
    def _merge(base: Dict, delta: Dict) -> Dict:
        """Merge helper that averages scalars & concatenates tone lists."""
        out = base.copy()
        if "tone" in delta:
            out.setdefault("tone", [])
            out["tone"].extend(delta["tone"])
        if "formality" in delta:
            out["formality"] = (out["formality"] + delta["formality"]) / 2
        if "syntactic_oddity" in delta:
            out["syntactic_oddity"] = max(
                out["syntactic_oddity"], delta["syntactic_oddity"]
            )
        if "lexical_quirks" in delta:
            for k, v in delta["lexical_quirks"].items():
                out["lexical_quirks"][k] = max(out["lexical_quirks"].get(k, 0.0), v)
        return out

    # ---------------------------------------------------------------------
    def to_elevenlabs_params(self, style: Dict) -> Dict:
        """Convert SpeechStylizer style to ElevenLabs speech parameters."""
        if self.voice_id is None:
            return None
        params = {
            'stability': 1.0 - (style['syntactic_oddity'] * 0.5),  # Reduce stability for odd syntax
            'similarity_boost': 1.0 - (style['lexical_quirks']['slang'] * 0.3),  # Reduce similarity for slang
            'style': 0.75,
            'use_speaker_boost': False,  # Always enable for character consistency
            'voice_id': self.voice_id
        }
        
        # Map formality to stability
        params['stability'] = max(0.1, min(1.0, params['stability'] + (style['formality'] * 0.3)))
        #params['style'] =0.5
         
        # Adjust for metaphor usage
        if style['lexical_quirks'].get('metaphor', 0.0) > 0.01:
            params['similarity_boost'] = max(0.1, params['similarity_boost'] - 0.2)
        
        return params

    def pick_best_voice(self, voices: list) -> dict:
        """
        Select the best ElevenLabs voice record for self.char using LLM-extracted traits.
        """
        global _voice_embeddings, _voices
        # 1. Construct prompt for LLM
        personality_embedding = _embedding_model.encode(' '.join(self.char.personality))

        # 3. Score each voice
        def score_voice(voice, gender, age, accent, personality, personality_embedding):
            score = 0
            labels = voice.get('labels', {})
            # Gender match
            label_gender = labels.get('gender', '').lower()
            if gender and label_gender == gender.lower():
                score += 5
            # Age match
            label_age = labels.get('age', '').lower()
            if age and label_age and label_age == age.lower():
                score += 1
            # Accent match
            label_accent = labels.get('accent', '').lower()
            if accent and label_accent and accent.lower() == label_accent:
                score += 1
            # Personality/description match (partial, case-insensitive)
            if personality and 'description' in labels:
                for word in personality:
                    if word.strip().lower() in labels['description'].lower():
                        score += .1
                dot_result = np.dot(personality_embedding, _voice_embeddings[voices.index(voice)])
                if isinstance(dot_result, np.ndarray):
                    dot_result = float(dot_result)  # fallback to scalar for now
                score += dot_result
            return score

        # 4. Pick the best voice
        gender = self.char.gender
        age = self.char.age
        accent = self.char.accent
        personality = self.char.personality
        scores = [score_voice(voice, gender, age, accent, personality, personality_embedding) for voice in voices]
        max_score = max(scores)
        best_voice = voices[scores.index(max_score)]
        return best_voice['voice_id']
