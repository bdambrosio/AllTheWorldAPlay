import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import CharacterPanel from './components/CharacterPanel';
import ShowPanel from './components/ShowPanel';
import WorldPanel from './components/WorldPanel';
import DirectorChoiceModal from './components/DirectorChoiceModal';
import ActChoiceModal from './components/ActChoiceModal';
import SceneChoiceModal from './components/SceneChoiceModal';
import DirectorChairModal from './components/DirectorChairModal';
import TabPanel from './components/TabPanel';
import './components/TabPanel.css';
import config from './config';
import { ReplayProvider, useReplay } from './contexts/ReplayContext';
import ExplorerModal from './components/ExplorerModal';

function App() {
  const { recordEvent } = useReplay();
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [characters, setCharacters] = useState({});
  const [showText, setShowText] = useState('');
  const [simStatus, setSimStatus] = useState({
    running: false,
    paused: false
  });
  const [logText, setLogText] = useState('');
  const [plays, setPlays] = useState([]);
  const [showPlayDialog, setShowPlayDialog] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [selectedPlay, setSelectedPlay] = useState(null);
  const [simState, setSimState] = useState({ running: false, paused: false });
  const [currentPlay, setCurrentPlay] = useState(null);
  const [worldState, setWorldState] = useState({});
  const websocket = useRef(null);
  const logRef = useRef(null);
  const initialized = useRef(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const messageQueue = useRef([]);
  const sendingLock = useRef(false);
  const [choiceRequest, setChoiceRequest] = useState(null);
  const [showDirectorChair, setShowDirectorChair] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [pendingTabEvent, setPendingTabEvent] = useState(null);
  const [pendingExplorerEvent, setPendingExplorerEvent] = useState(null);
  const [pendingExplorerTabEvent, setPendingExplorerTabEvent] = useState(null);
  const [appMode, setAppMode] = useState('simulation');
  const pendingPlayLoadRef = useRef(null);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const speechEnabledRef = useRef(speechEnabled);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    async function initSession() {
      try {
        const response = await fetch(`${config.httpUrl}/api/session/new`);
        const data = await response.json();
        setSessionId(data.session_id);
        
        websocket.current = new WebSocket(`${config.wsProtocol}://${config.wsHost}/ws/${data.session_id}`);
        
        websocket.current.onmessage = (event) => {
              const data = JSON.parse(event.data);
          console.log('Message received:', data);  // See full message
          if (data.text === 'goal_choice' || data.text === 'task_choice' || data.text === 'act_choice' || data.text === 'scene_choice') {
            setChoiceRequest({
              ...data,
              choice_type: data.text.split('_')[0]  // 'goal', 'task', 'act', or 'scene'
            });
          } else {
            switch(data.type) {
              case 'character_update':
                console.log('Character data:', data.name);  // Reduced logging
                setCharacters(prev => ({
                  ...prev,
                  [data.name]: {
                    ...data.data,
                    image: data.data.image ? `data:image/jpeg;base64,${data.data.image}` : null,
                    status: isProcessing ? 'processing' : 'idle',
                    explorer_state: data.data.explorer_state
                  }
                }));
                break;
              case 'character_details':
                console.log('Character details received:', data.name);
                setCharacters(prev => ({
                  ...prev,
                  [data.name]: {
                    ...prev[data.name],
                    explorer_state: data.details,
                    status: 'idle'
                  }
                }));
                break;
              case 'show_update':
                console.log('show_update:', data.text);
                setLogText(prev => {
                  const newEntry = (data.name && data.name.trim() !== '') ? `${data.name}: ${data.text}` : data.text;
                  return prev ? `${prev} \n${newEntry}` : newEntry;
                });
                break;
              case 'context_update':
                console.log('context_update:', data.text);
                setLogText(prev => {
                  const newEntry = data.text; 
                  return prev ? `${prev} \n\n ${newEntry} \n\n` : newEntry;
                });
                break;
              case 'status_update':
                console.log('status_update:', data.status);
                setSimState(data.status);
                setIsProcessing(data.status.running && !data.status.paused);
                break;
              case 'output':
                console.log('output:', data.text);
                setMessages(prev => [...prev, data.text]);
                break;
              case 'play_list':
                setPlays(data.plays);
                setShowPlayDialog(true);
                break;
              case 'confirm_reload':
                setShowConfirmDialog(true);
                break;
              case 'play_error':
                alert(data.error);  // Simple error handling for now
                break;
              case 'play_loaded':
                setCurrentPlay(data.name);
                break;
              case 'state_update':
                console.log('state_update:', data);
                setSimState(data);
                break;
              case 'world_update':
                console.log('world_update:', data.data);
                setWorldState(data.data);
                break;
              case 'command_complete':
                if (data.command === 'step' || data.command === 'refresh') {
                  setIsProcessing(false);
                }
                break;
              case 'command_ack':
                console.log('Debug - command_ack received:', data);
                setIsProcessing(false);
                setSimState(prev => ({
                  ...prev,
                  running: false,
                  paused: true
                }));
                if (data.command === 'load_play' && pendingPlayLoadRef.current) {
                  console.log('Debug - setting currentPlay to:', pendingPlayLoadRef.current);
                  setCurrentPlay(pendingPlayLoadRef.current);
                  pendingPlayLoadRef.current = null;
                }
                break;
              case 'chat_response':
                console.log('chat_response:', data.text);
                setCharacters(prev => ({
                  ...prev,
                  [data.char_name]: {
                    ...prev[data.char_name],
                    chatOutput: data.text
                  }
                }));
                break;
              case 'replay_event':
                console.log('Replay event:', data);
                handleReplayEvent(data);
                break;
              case 'setActiveTab':
                if (event.arg.panelId === 'characterTabs') {
                  const characterArray = Object.values(characters);
                  if (characterArray.length > 0) {
                    const characterIndex = characterArray.findIndex(char => char.name === event.arg.characterName);
                    if (characterIndex !== -1) {
                      setActiveTab(characterIndex);
                    } else {
                      setPendingTabEvent(event);
                    }
                  } else {
                    setPendingTabEvent(event);
                  }
                }
                break;
              case 'setExplorerTab':
                if (characters[event.arg.characterName]) {
                  setPendingExplorerTabEvent(event);
                } else {
                  setPendingExplorerTabEvent(event);
                }
                break;
              case 'mode_update':
                setAppMode(data.mode);
                break;
              case 'speak':
                console.log('Speak message received:');
                if (speechEnabledRef.current && data.audio) {
                  const mime = data.audio_format === 'mp3' || !data.audio_format ? 'audio/mp3' : `audio/${data.audio_format}`;
                  const audio = new Audio(`data:${mime};base64,${data.audio}`);
                  audio.onended = () => {
                    sendCommand('speech_complete');
                  };
                  audio.onerror = (error) => {
                    console.error('Audio playback error:', error);
                    sendCommand('speech_complete');
                  };
                  audio.play().catch(error => {
                    console.error('Audio play failed:', error);
                    sendCommand('speech_complete');
                  });
                }
                break;
              case 'speech_toggle':
                speechEnabledRef.current = data.enabled;  // sync ref right away
                setSpeechEnabled(data.enabled);
                break;
              default:
                console.log('Unknown message type:', data.type);
            }
          }
        };

      } catch (error) {
        console.error('Failed to initialize session:', error);
        setMessages(prev => [...prev, 'Failed to connect to server']);
      }
    }
    
    initSession();
    return () => websocket.current?.close();
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logText]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim() || !websocket.current) return;

    websocket.current.send(inputValue);
    setInputValue('');
  };

  const handleInitialize = () => {
    recordEvent('initialize', 'simulation');
    sendReplayEvent('initialize', 'simulation', 'initialize');
    sendCommand('initialize');
  };

  const handleStep = async () => {
    setIsProcessing(true);
    recordEvent('step', 'simulation');
    sendReplayEvent('step', 'simulation', 'step');
    sendCommand('step');
  };

  const handleRun = async () => {
    setIsProcessing(true);
    recordEvent('run', 'simulation');
    sendCommand('run');
  };

  const handlePause = () => {
    setIsProcessing(false);
    recordEvent('pause', 'simulation');
    sendCommand('pause');
  };

  const handleRefresh = () => {
    setIsProcessing(false);
    setSimState(prev => ({
      ...prev,
      running: false,
      paused: true
    }));
    sendCommand('refresh');
  };

  const sendMessageSafely = async (message) => {
    messageQueue.current.push(message);
    if (!sendingLock.current) {
      await processQueue();
    }
  };

  const processQueue = async () => {
    if (sendingLock.current || messageQueue.current.length === 0) return;
    
    sendingLock.current = true;
    try {
      while (messageQueue.current.length > 0) {
        const message = messageQueue.current[0];
        if (websocket.current?.readyState === WebSocket.OPEN) {
          websocket.current.send(message);
          // Small delay to ensure message processing
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        messageQueue.current.shift();
      }
    } finally {
      sendingLock.current = false;
    }
  };

  const sendCommand = async (command, payload = {}) => {
    console.log('sendCommand called:', { command, payload });
    const message = JSON.stringify({
      type: 'command',
      action: command,
      ...payload  // Flatten payload into main message
    });
    await sendMessageSafely(message);
  };

  const sendReplayEvent = (action, arg, ui_action) => {
    const message = JSON.stringify({
      type: 'replay_event',
      action: action,
      arg: arg,
      timestamp: Date.now()
    });
    sendMessageSafely(message);
  };

  const handleChoice = (choiceId, customData = null) => {
    if (!choiceRequest) return;
    
    const response = {
      character_name: choiceRequest.character_name,
      selected_id: choiceId
    };

    // Handle narrative act/scene choices differently
    if (choiceRequest.choice_type === 'act' && choiceRequest.act_data) {
      if (customData) {
        response.updated_act = customData;
      } else {
        response.updated_act = choiceRequest.act_data;
      }
    } else if (choiceRequest.choice_type === 'scene' && choiceRequest.scene_data) {
      if (customData) {
        response.updated_scene = customData;
      } else {
        response.updated_scene = choiceRequest.scene_data;
      }
    } else if (customData) {
      // Handle character-level choices (goal, task, action)
      response.custom_data = {
        name: customData.name,
        mode: customData?.mode || '',
        action: customData?.action || '',
        actors: customData.actors,
        description: customData.description,
        target: customData?.target || '',
        termination: customData.termination,
        reason: customData?.reason || '',
        duration: customData?.duration || 1
      };
    }
    
    sendCommand('choice_response', response);
    setChoiceRequest(null);
  };

  const handleSelectPlay = (playName) => {
    setSelectedPlay(playName);
    recordEvent('select_play', playName);
    sendReplayEvent('select_play', playName, 'select_play');
    // Optionally, you could trigger UI updates here if needed
  };

  const handleReplayEvent = (event) => {
    console.log('Handling replay event:', event);
    switch (event.action) {
      case 'setShowPlayDialog':
        setShowPlayDialog(event.arg.show);
        break;
      case 'setActiveTab':
        if (event.arg.panelId === 'characterTabs') {
          const characterArray = Object.values(characters);
          if (characterArray.length > 0) {
            const characterIndex = characterArray.findIndex(char => char.name === event.arg.characterName);
            if (characterIndex !== -1) {
              setActiveTab(characterIndex);
            } else {
              setPendingTabEvent(event);
            }
          } else {
            setPendingTabEvent(event);
          }
        }
        break;
      case 'setShowExplorer':
        if (characters[event.arg.characterName]) {
          setPendingExplorerEvent(event);
        } else {
          setPendingExplorerEvent(event);
        }
        break;
      case 'setExplorerTab':
        if (characters[event.arg.characterName]) {
          setPendingExplorerTabEvent(event);
        } else {
          setPendingExplorerTabEvent(event);
        }
        break;
      default:
        // Handle other replay events
        break;
    }
  };

  console.log('Current characters:', characters);  // See if state is updated
  console.log('RENDER - characters state:', characters);

  useEffect(() => {
    if (pendingTabEvent && pendingTabEvent.action === 'setActiveTab') {
      const characterArray = Object.values(characters);
      if (characterArray.length > 0) {
        const characterIndex = characterArray.findIndex(char => char.name === pendingTabEvent.arg.characterName);
        if (characterIndex !== -1) {
          setActiveTab(characterIndex);
          setPendingTabEvent(null);
        }
      }
    }
  }, [characters, pendingTabEvent]);

  useEffect(() => {
    if (pendingExplorerEvent && pendingExplorerEvent.action === 'setShowExplorer') {
      const characterName = pendingExplorerEvent.arg.characterName;
      if (characters[characterName]) {
        setPendingExplorerEvent(null);
      }
    }
  }, [characters, pendingExplorerEvent]);

  useEffect(() => {
    console.log('Debug - appMode:', appMode);
    console.log('Debug - currentPlay:', currentPlay);
    console.log('Debug - isProcessing:', isProcessing);
    console.log('Debug - simState:', simState);
  }, [appMode, currentPlay, isProcessing, simState]);

  useEffect(() => {
    speechEnabledRef.current = speechEnabled;
  }, [speechEnabled]);

  const handleToggleSpeech = () => {
    sendCommand('toggle_speech');        // tell the server
  };

  return (
    <ReplayProvider>
      <div className="app-container">
        <div className="character-panels">
          <TabPanel 
            characters={Object.values(characters)
              .filter(character => character && character.name)
              .map(character => (
                <CharacterPanel 
                  key={character.name} 
                  character={character}
                  sessionId={sessionId}
                  sendCommand={sendCommand}
                  sendReplayEvent={sendReplayEvent}
                  showExplorer={pendingExplorerEvent?.arg?.characterName === character.name ? 
                                pendingExplorerEvent.arg.show : undefined}
                  onExplorerShown={() => {
                    if (pendingExplorerEvent?.arg?.characterName === character.name) {
                      setPendingExplorerEvent(null);
                    }
                  }}
                />
              ))}
            sendReplayEvent={sendReplayEvent}
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        </div>
        <div className="center-panel">
          <div className="world-container">
            <div className="world-header">
              <div className="world-panel">
                <WorldPanel worldState={worldState} />
              </div>
              <div className="world-description">
                {worldState?.show || ''}
              </div>
            </div>
            <div className="log-area" ref={logRef}>
              {logText}
            </div>
          </div>
        </div>
        <div className="control-panel">
          <div className="control-buttons">
            <button className="control-button" onClick={handleInitialize}>Start</button>
            <button className="control-button" 
              onClick={handleRun} 
              disabled={isProcessing || simState.running || (appMode === 'replay' && !currentPlay)}>Run</button>
            <button className="control-button" 
              onClick={handleStep} 
              disabled={isProcessing || simState.running || (appMode === 'replay' && !currentPlay)}>Step</button>
            <button className="control-button" onClick={handlePause}>Pause</button>
            <button className="control-button" onClick={() => sendCommand('showMap')} disabled={appMode === 'replay'}>Show Map</button>
            <button onClick={handleRefresh} className="control-button" disabled={appMode === 'replay'}>
              Refresh
            </button>
            <button className="control-button" onClick={() => sendCommand('load_known_actors')} disabled={appMode === 'replay'}>Load</button>
            <button className="control-button" onClick={() => sendCommand('save_known_actors')} disabled={appMode === 'replay'}>Save</button>
            <button 
              className="control-button" 
              onClick={handleToggleSpeech}
            >
              {speechEnabled ? 'Voice On' : 'Voice Off'}
            </button>
          </div>
          <div className="status-area">
            {currentPlay && <div>Play: {currentPlay}</div>}
            <div>Status: {
              simState.running ? 'Running' : 
              isProcessing ? 'Processing...' : 
              'Paused'
            }</div>
          </div>
          <button 
            className="director-chair-button"
            onClick={() => setShowDirectorChair(true)}
          >
            Director's Chair
          </button>
        </div>

        {showPlayDialog && (
          <div className="dialog">
            <h3>Select Scenario</h3>
            <select
              value={selectedPlay ?? ""}
              onChange={(e) => {
                setSelectedPlay(e.target.value);
                console.log('Select changed, new selectedPlay:', e.target.value);
                recordEvent('select_play', e.target.value);
                sendReplayEvent('select_play', e.target.value);
              }}
            >
              <option value="" disabled>Select a scenario...</option>
              {plays.map(play => (
                <option key={play} value={play}>{play}</option>
              ))}
            </select>
            <button
              onClick={() => {
                setIsProcessing(true);
                pendingPlayLoadRef.current = selectedPlay;
                sendCommand('load_play', { play: selectedPlay });
                setShowPlayDialog(false);
                sendReplayEvent('setShowPlayDialog', { show: false });
              }}
              disabled={!selectedPlay}
            >
              Load
            </button>
            <button onClick={() => {
              setShowPlayDialog(false);
              sendReplayEvent('setShowPlayDialog', { show: false });
            }}>Cancel</button>
          </div>
        )}

        {showConfirmDialog && (
          <div className="dialog">
            <p>This will reset the current simulation. Continue?</p>
            <button onClick={() => {
              sendCommand('confirm_load_play', { play: selectedPlay });
              setShowConfirmDialog(false);
            }}>Yes</button>
            <button onClick={() => setShowConfirmDialog(false)}>No</button>
          </div>
        )}

        {choiceRequest && choiceRequest.choice_type === 'act' && (
          <ActChoiceModal
            request={choiceRequest}
            onChoice={handleChoice}
            onClose={() => setChoiceRequest(null)}
          />
        )}

        {choiceRequest && choiceRequest.choice_type === 'scene' && (
          <SceneChoiceModal
            request={choiceRequest}
            onChoice={handleChoice}
            onClose={() => setChoiceRequest(null)}
          />
        )}

        {choiceRequest && ['goal', 'task', 'action'].includes(choiceRequest.choice_type) && (
          <DirectorChoiceModal
            request={choiceRequest}
            onChoice={handleChoice}
            onClose={() => setChoiceRequest(null)}
          />
        )}

        {showDirectorChair && (
          <DirectorChairModal
            characters={characters}
            onClose={() => setShowDirectorChair(false)}
            sendCommand={sendCommand}
          />
        )}

        {pendingExplorerEvent && pendingExplorerEvent.action === 'setShowExplorer' && (
          <ExplorerModal
            character={characters[pendingExplorerEvent.arg.characterName]}
            sessionId={sessionId}
            lastState={pendingExplorerEvent.arg.lastState}
            status={pendingExplorerEvent.arg.status}
            onClose={() => setPendingExplorerEvent(null)}
            activeTab={pendingExplorerTabEvent?.arg?.characterName === pendingExplorerEvent.arg.characterName ? 
                       pendingExplorerTabEvent.arg.tabName : undefined}
            onTabChanged={() => {
              if (pendingExplorerTabEvent?.arg?.characterName === pendingExplorerEvent.arg.characterName) {
                setPendingExplorerTabEvent(null);
              }
            }}
            sendCommand={sendCommand}
            sendReplayEvent={sendReplayEvent}
          />
        )}
      </div>
    </ReplayProvider>
  );
}

export default App;
