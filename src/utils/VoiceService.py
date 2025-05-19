# voice_service.py

import requests
import json
from typing import Dict, Optional, List, Any
import os
from playsound3 import playsound
import tempfile
import asyncio
import soundfile as sf

class CoquiTTSProvider:
    def __init__(self):
        self.voices = []
        self.voice_map = {}
        self.tts = None
        self._voices_loaded = False
        self.sample_rate = 24000  # Default for XTTS-v2

    def initialize(self) -> None:
        try:
            from TTS.api import TTS
            from TTS.utils.manage import ModelManager
            import builtins
            
            # Force all confirmations to "yes"
            ModelManager.ask_tos = lambda self, path: True
            
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
            self._voices_loaded = True
            print("Coqui-TTS initialized successfully")
        except Exception as e:
            print(f"Error initializing Coqui-TTS: {e}")
            self._voices_loaded = False
            raise

    async def speak(self, text: str, options: Dict = None) -> None:
        if not self.tts:
            self.initialize()

        options = options or {}
        
        try:
            # Get reference audio if provided
            speaker_wav = options.get('speaker_wav')
            language = options.get('language', 'en')
            
            # Generate audio
            audio = self.tts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language
            )

            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                sf.write(temp_file.name, audio, self.sample_rate)
                temp_file.flush()
                playsound(temp_file.name)
                os.unlink(temp_file.name)  # Clean up

        except Exception as e:
            raise Exception(f"Coqui-TTS error: {e}")

    async def synthesize(self, text: str, options: Dict = None) -> str:
        """Generate speech and return the path to the audio file without playing it."""
        if not self.tts:
            self.initialize()

        options = options or {}
        
        try:
            # Get reference audio if provided
            speaker_wav = options.get('speaker_wav')
            language = options.get('language', 'en')
            
            # Generate audio
            audio = self.tts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language
            )

            # Save to temporary file but don't play
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                sf.write(temp_file.name, audio, self.sample_rate)
                temp_file.flush()
                return temp_file.name  # Return the path instead of playing

        except Exception as e:
            raise Exception(f"Coqui-TTS error: {e}")

    def get_voices(self) -> List[Dict]:
        # Coqui-TTS doesn't have a concept of voices like ElevenLabs
        # Instead, it uses reference audio files for voice cloning
        return [{
            'name': 'XTTS-v2',
            'description': 'Multilingual voice cloning model',
            'requires_reference_audio': True
        }]

    def stop(self) -> None:
        # Note: playsound doesn't support stopping, this is a stub
        pass

    def pause(self) -> None:
        # Note: playsound doesn't support pausing, this is a stub
        pass

    def resume(self) -> None:
        # Note: playsound doesn't support resuming, this is a stub
        pass

    def map_character_to_voice(self, character: str, voice: str) -> None:
        # For Coqui-TTS, this would map characters to reference audio files
        self.voice_map[character] = voice

class ElevenLabsProvider:
    def __init__(self):
        self.api_key = None
        self.base_url = 'https://api.elevenlabs.io/v1'
        self.voices = None  # Change from [] to None to indicate "not loaded"
        self.voice_map = {}
        self.current_audio_file = None
        self._voices_loaded = False

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key
        self.fetch_voices()

    def fetch_voices(self) -> None:
        if self.voices is not None:
            # Already cached, do nothing
            return
        try:
            response = requests.get(
                f"{self.base_url}/voices",
                headers={
                    'xi-api-key': self.api_key,
                    'Content-Type': 'application/json'
                }
            )
            response.raise_for_status()
            data = response.json()
            self.voices = data['voices']
            self._voices_loaded = True
            print(f"ElevenLabs: Loaded {len(self.voices)} voices")
        except Exception as e:
            print(f"Error fetching ElevenLabs voices: {e}")
            self.voices = []
            self._voices_loaded = False
            raise

    async def speak(self, text: str, options: Dict = None) -> None:
        if not self.api_key:
            raise ValueError("ElevenLabs API key not set")

        options = options or {}
        
        # Determine voice ID
        voice_id = options.get('voice_id')
        if not voice_id and options.get('character'):
            voice_id = self.voice_map.get(options['character'])
        if not voice_id and self.voices:
            voice_id = self.voices[0]['voice_id']

        if not voice_id:
            raise ValueError("No voice available for ElevenLabs TTS")

        # Build request parameters
        request_data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": options.get('stability', 0.5),
                "similarity_boost": options.get('similarityBoost', 0.5)
            }
        }
        
        # Add optional parameters if provided
        if 'style' in options:
            request_data['voice_settings']['style'] = options['style']
        if 'speakerBoost' in options:
            request_data['voice_settings']['use_speaker_boost'] = options['speakerBoost']

        try:
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers={
                    'xi-api-key': self.api_key,
                    'Content-Type': 'application/json'
                },
                json=request_data
            )
            response.raise_for_status()

            # Save to temporary file and play
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                self.current_audio_file = temp_file.name
                playsound(temp_file.name)
                os.unlink(temp_file.name)  # Clean up

        except Exception as e:
            raise Exception(f"ElevenLabs TTS error: {e}")

    async def synthesize(self, text: str, options: Dict = None) -> str:
        """Generate speech and return the path to the audio file without playing it."""
        if not self.api_key:
            raise ValueError("ElevenLabs API key not set")

        options = options or {}
        
        # Determine voice ID
        voice_id = options.get('voice_id')
        if not voice_id and options.get('character'):
            voice_id = self.voice_map.get(options['character'])
        if not voice_id and self.voices:
            voice_id = self.voices[0]['voice_id']

        if not voice_id:
            raise ValueError("No voice available for ElevenLabs TTS")

        # Build request parameters
        request_data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": options.get('stability', 0.5),
                "similarity_boost": options.get('similarityBoost', 0.5)
            }
        }
        
        # Add optional parameters if provided
        if 'style' in options:
            request_data['voice_settings']['style'] = options['style']
        if 'speakerBoost' in options:
            request_data['voice_settings']['use_speaker_boost'] = options['speakerBoost']

        try:
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers={
                    'xi-api-key': self.api_key,
                    'Content-Type': 'application/json'
                },
                json=request_data
            )
            response.raise_for_status()

            # Save to temporary file but don't play
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                self.current_audio_file = temp_file.name
                return temp_file.name  # Return the path instead of playing

        except Exception as e:
            raise Exception(f"ElevenLabs TTS error: {e}")

    def get_voices(self) -> list:
        if self.voices is None:
            self.fetch_voices()
        return self.voices or []

    def stop(self) -> None:
        # Note: playsound doesn't support stopping, this is a stub
        pass

    def pause(self) -> None:
        # Note: playsound doesn't support pausing, this is a stub
        pass

    def resume(self) -> None:
        # Note: playsound doesn't support resuming, this is a stub
        pass

    def map_character_to_voice(self, character: str, voice: str) -> None:
        self.voice_map[character] = voice

class VoiceService:
    def __init__(self):
        self.providers = {
            'coqui': CoquiTTSProvider(),
            'elevenlabs': ElevenLabsProvider()
        }
        self.current_provider = 'coqui'  # Set Coqui as default

    def set_provider(self, provider_name: str) -> None:
        if provider_name in self.providers:
            self.current_provider = provider_name

    def set_api_key(self, provider_name: str, api_key: str) -> None:
        if provider_name in self.providers and hasattr(self.providers[provider_name], 'set_api_key'):
            self.providers[provider_name].set_api_key(api_key)

    async def speak(self, text: str, options: Dict = None) -> None:
        await self.providers[self.current_provider].speak(text, options or {})

    def get_voices(self) -> List[Dict]:
        try:
            provider = self.providers[self.current_provider]
            if hasattr(provider, 'get_voices'):
                #if asyncio.iscoroutinefunction(provider.get_voices):
                #    return await provider.get_voices()
                return provider.get_voices()
            return []
        except Exception as e:
            print(f"Error getting voices from {self.current_provider}: {e}")
            return []

    def stop(self) -> None:
        self.providers[self.current_provider].stop()

    def pause(self) -> None:
        self.providers[self.current_provider].pause()

    def resume(self) -> None:
        self.providers[self.current_provider].resume()

    def map_character_to_voice(self, character: str, voice: str) -> None:
        self.providers[self.current_provider].map_character_to_voice(character, voice)

    async def synthesize(self, text: str, options: Dict = None) -> str:
        """Generate speech and return the path to the audio file without playing it."""
        return await self.providers[self.current_provider].synthesize(text, options or {})

# Create singleton instance
voice_service = VoiceService()