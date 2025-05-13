// src/sim/webworld/src/voiceService.js

// Provider: Web Speech API
const webSpeechProvider = {
  synth: window.speechSynthesis,
  voices: [],
  voiceMap: {},

  init() {
    // Load voices (may be async in some browsers)
    this.voices = this.synth.getVoices();
    if (this.voices.length === 0) {
      window.speechSynthesis.onvoiceschanged = () => {
        this.voices = this.synth.getVoices();
      };
    }
  },

  speak(text, options = {}) {
    return new Promise((resolve, reject) => {
      this.stop();
      const utter = new window.SpeechSynthesisUtterance(text);

      // Character-to-voice mapping
      let voiceName = options.voice;
      if (options.character && this.voiceMap[options.character]) {
        voiceName = this.voiceMap[options.character];
      }
      if (voiceName) {
        const match = this.voices.find(v => v.name === voiceName);
        if (match) utter.voice = match;
      }

      // Additional options
      if (options.rate) utter.rate = options.rate;
      if (options.pitch) utter.pitch = options.pitch;
      if (options.volume) utter.volume = options.volume;

      utter.onend = resolve;
      utter.onerror = reject;

      this.synth.speak(utter);
    });
  },

  getVoices() {
    return this.voices;
  },

  stop() {
    this.synth.cancel();
  },

  pause() {
    this.synth.pause();
  },

  resume() {
    this.synth.resume();
  },

  mapCharacterToVoice(character, voice) {
    this.voiceMap[character] = voice;
  }
};

// ElevenLabs Provider
const elevenLabsProvider = {
    apiKey: null,
    baseUrl: 'https://api.elevenlabs.io/v1',
    voices: [],
    voiceMap: {},
    audioElement: null,
    currentUtterance: null,
    isPlaying: false,
    
    init(apiKey = null) {
      if (apiKey) this.apiKey = apiKey;
      if (!this.apiKey) {
        console.warn('ElevenLabs provider initialized without API key');
        return;
      }
      
      // Create audio element for playback
      if (!this.audioElement) {
        this.audioElement = new Audio();
        this.audioElement.addEventListener('ended', () => {
          this.isPlaying = false;
          // If we have an onend callback for the current utterance, call it
          if (this.currentUtterance && this.currentUtterance.onend) {
            this.currentUtterance.onend();
          }
          this.currentUtterance = null;
        });
        
        this.audioElement.addEventListener('error', (e) => {
          console.error('Audio playback error:', e);
          this.isPlaying = false;
          // If we have an onerror callback for the current utterance, call it
          if (this.currentUtterance && this.currentUtterance.onerror) {
            this.currentUtterance.onerror(e);
          }
        });
      }
      
      // return the promise so callers can await readiness
      return this.fetchVoices();
    },
    
    async fetchVoices() {
      try {
        const response = await fetch(`${this.baseUrl}/voices`, {
          method: 'GET',
          headers: {
            'xi-api-key': this.apiKey,
            'Content-Type': 'application/json'
          }
        });
        
        if (!response.ok) {
          throw new Error(`Failed to fetch voices: ${response.status}`);
        }
        
        const data = await response.json();
        // Transform to similar format as Web Speech API
        this.voices = data.voices.map(voice => ({
          name: voice.name,
          voiceId: voice.voice_id,
          // Add additional properties to match Web Speech API structure if needed
          lang: 'en-US',
          localService: false,
          default: false
        }));
        
        console.log(`ElevenLabs: Loaded ${this.voices.length} voices`);
      } catch (error) {
        console.error('Error fetching ElevenLabs voices:', error);
        this.voices = [];
      }
    },
    
    async speak(text, options = {}) {
      return new Promise(async (resolve, reject) => {
        if (!this.apiKey) {
          reject(new Error('ElevenLabs API key not set'));
          return;
        }
        
        this.stop(); // Stop any current speech
        
        let voiceId = options.voiceId || this.voices[0].voiceId;
        
        // If character is specified, check the voice map
        if (options.character && this.voiceMap[options.character]) {
          // The voice map might store either a voice name or a voice ID
          const mappedVoice = this.voiceMap[options.character];
          
          if (typeof mappedVoice === 'string' && !mappedVoice.includes('-')) {
            const byName = this.voices.find(v => v.name === mappedVoice);
            voiceId = byName ? byName.voiceId : mappedVoice;  // UID if not a name            -   …       // name → look up id
          } else {
            voiceId = mappedVoice;
          }
          if (typeof mappedVoice === 'string') {
            // string could be name or UID
            const byName = this.voices.find(v => v.name === mappedVoice);
            voiceId = byName ? byName.voiceId : mappedVoice;  // UID if not a name
          } else if (typeof mappedVoice === 'object' && mappedVoice.voiceId) {
            // merge per‑character defaults
            voiceId = mappedVoice.voiceId;
            options = { ...mappedVoice, ...options };   // caller overrides defaults
          } else {
            return reject(new Error('Invalid voice mapping for character'));
          }        
        }
        
        // If we still don't have a voiceId, use the first available voice
        if (!voiceId && this.voices.length > 0) {
          voiceId = this.voices[0].voiceId;
        }
        
        if (!voiceId) {
          reject(new Error('No voice available for ElevenLabs TTS'));
          return;
        }
        
        this.currentUtterance = {
          text,
          voice: { voiceId },
          onend: resolve,
          onerror: reject,
          rate: options.rate || 1,
          pitch: options.pitch || 1,
          volume: options.volume || 1
        };
        
        // Build request parameters with expanded options
        const requestData = {
          text: text,
          model_id: "eleven_monolingual_v1",
          voice_settings: {
            stability: typeof options.stability === 'number' ? options.stability : 0.5,
            similarity_boost: typeof options.similarityBoost === 'number' ? options.similarityBoost : 0.5,
            style: typeof options.style === 'number' ? options.style : undefined,
            use_speaker_boost: typeof options.speakerBoost === 'boolean' ? options.speakerBoost : undefined
          }
        };
        Object.keys(requestData.voice_settings).forEach(
          k => requestData.voice_settings[k] === undefined && delete requestData.voice_settings[k]
        );
        
        try {
          const response = await fetch(`${this.baseUrl}/text-to-speech/${voiceId}`, {
            method: 'POST',
            headers: {
              'xi-api-key': this.apiKey,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
          });
          
          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`ElevenLabs API error (${response.status}): ${errorText}`);
          }
          
          // Get the audio blob and play it
          const audioBlob = await response.blob();
          const audioUrl = URL.createObjectURL(audioBlob);
          
          this.audioElement.src = audioUrl;
          this.audioElement.volume = this.currentUtterance.volume;
          
          // Clean up the previous URL
          const oldSrc = this.audioElement.src;
          this.audioElement.onloadeddata = () => {
            if (oldSrc.startsWith('blob:')) {
              URL.revokeObjectURL(oldSrc);
            }
          };
          
          // Play the audio
          this.isPlaying = true;
          await this.audioElement.play();
          
        } catch (error) {
          console.error('ElevenLabs TTS error:', error);
          reject(error);
          this.currentUtterance = null;
        }
      });
    },
    
    getVoices() {
      return this.voices;
    },
    
    stop() {
      if (this.audioElement) {
        this.audioElement.pause();
        this.audioElement.currentTime = 0;
        this.isPlaying = false;
      }
      this.currentUtterance = null;
    },
    
    pause() {
      if (this.audioElement && this.isPlaying) {
        this.audioElement.pause();
        this.isPlaying = false;
      }
    },
    
    resume() {
      if (this.audioElement && !this.isPlaying && this.currentUtterance) {
        this.audioElement.play();
        this.isPlaying = true;
      }
    },
    
    mapCharacterToVoice(character, voice) {
      this.voiceMap[character] = voice;
    },
    
    setApiKey(apiKey) {
      this.apiKey = apiKey;
      this.fetchVoices();
    }
  };
  

// Main voiceService wrapper
const providers = {
  webspeech: webSpeechProvider,
  elevenlabs: elevenLabsProvider, 
};

let currentProvider = 'webspeech';

const voiceService = {
  async setProvider(providerName) {
    if (providers[providerName]) {
      currentProvider = providerName;
      await providers[providerName].init?.();   // wait for voices
    }
  },

  speak(text, options) {
    return providers[currentProvider].speak(text, options);
  },

  getVoices() {
    return providers[currentProvider].getVoices();
  },

  stop() {
    providers[currentProvider].stop();
  },

  pause() {
    providers[currentProvider].pause();
  },

  resume() {
    providers[currentProvider].resume();
  },

  mapCharacterToVoice(character, voice) {
    providers[currentProvider].mapCharacterToVoice(character, voice);
  },
  
  async setApiKey(providerName, apiKey) {
    if (providers[providerName] && providers[providerName].setApiKey) {
        await providers[providerName].setApiKey(apiKey);
    }
  }
};

// Initialize on load
webSpeechProvider.init();

export default voiceService;