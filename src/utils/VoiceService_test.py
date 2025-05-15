import asyncio
import os
from VoiceService import VoiceService

async def test_coqui_tts():
    print("\n=== Testing Coqui-TTS ===")
    voice_service = VoiceService()
    
    # Test get_voices
    voices = await voice_service.get_voices()
    print(f"Available voices: {voices}")
    
    # Test basic speech
    print("\nTesting basic speech...")
    await voice_service.speak(
        "Hello, this is a test of the Coqui TTS system.",
        {
            'speaker_wav': 'female.wav',  # Replace with actual path
            'language': 'en'
        }
    )
    
    # Test character mapping
    print("\nTesting character mapping...")
    voice_service.map_character_to_voice('narrator', 'path/to/narrator_voice.wav')
    await voice_service.speak(
        "This is the narrator speaking.",
        {
            'character': 'narrator',
            'language': 'en'
        }
    )

async def test_elevenlabs():
    print("\n=== Testing ElevenLabs ===")
    voice_service = VoiceService()
    voice_service.set_provider('elevenlabs')
    
    # Set your API key
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("Please set ELEVENLABS_API_KEY environment variable")
        return
        
    voice_service.set_api_key('elevenlabs', api_key)
    
    # Test get_voices
    voices = await voice_service.get_voices()
    print(f"Available voices: {voices}")
    
    # Test basic speech
    print("\nTesting basic speech...")
    await voice_service.speak(
        "Hello, this is a test of the ElevenLabs TTS system.",
        {
            'voiceId': voices[0]['voice_id'] if voices else None,
            'stability': 0.5,
            'similarityBoost': 0.5
        }
    )
    
    # Test character mapping
    print("\nTesting character mapping...")
    if voices:
        voice_service.map_character_to_voice('narrator', voices[0]['voice_id'])
        await voice_service.speak(
            "This is the narrator speaking.",
            {
                'character': 'narrator',
                'stability': 0.5,
                'similarityBoost': 0.5
            }
        )

async def main():
    try:
        # Test Coqui-TTS
        await test_coqui_tts()
    except Exception as e:
        print(f"Error during testing: {e}")
       
    try:    # Test ElevenLabs
        await test_elevenlabs()
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(main())