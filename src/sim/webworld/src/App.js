import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import CharacterPanel from './components/CharacterPanel';
import ShowPanel from './components/ShowPanel';
import WorldPanel from './components/WorldPanel';
import DirectorChoiceModal from './components/DirectorChoiceModal';
import DirectorChairModal from './components/DirectorChairModal';
import TabPanel from './components/TabPanel';
import './components/TabPanel.css';

function App() {
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

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    async function initSession() {
      try {
        const response = await fetch('http://localhost:8000/api/session/new');
        const data = await response.json();
        setSessionId(data.session_id);
        
        websocket.current = new WebSocket(`ws://localhost:8000/ws/${data.session_id}`);
        
        websocket.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log('Message received:', data);  // See full message
          if (data.text === 'goal_choice' || data.text === 'task_choice' || data.text === 'act_choice') {
            setChoiceRequest({
              ...data,
              choice_type: data.text.split('_')[0]  // 'goal', 'task', or 'act'
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
                setLogText(prev => {
                  const newEntry = data.text;
                  return prev ? `${prev} \n${newEntry}` : newEntry;
                });
                break;
              case 'context_update':
                setLogText(prev => {
                  const newEntry = data.text; 
                  return prev ? `${prev} \n\n ${newEntry} \n\n` : newEntry;
                });
                break;
              case 'status_update':
                setSimState(data.status);
                setIsProcessing(data.status.running && !data.status.paused);
                break;
              case 'output':
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
                setSimState(data);
                break;
              case 'world_update':
                setWorldState(data.data);
                break;
              case 'command_complete':
                if (data.command === 'step' || data.command === 'refresh') {
                  setIsProcessing(false);
                }
                break;
              case 'command_ack':
                setIsProcessing(false);
                setSimState(prev => ({
                  ...prev,
                  running: false,
                  paused: true
                }));
                break;
              case 'chat_response':
                setCharacters(prev => ({
                  ...prev,
                  [data.char_name]: {
                    ...prev[data.char_name],
                    chatOutput: data.text
                  }
                }));
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

  const handleStep = async () => {
    setIsProcessing(true); 
    sendCommand('step');
  };

  const handleRun = async () => {
    setIsProcessing(true);
    sendCommand('run');
  };

  const handlePause = () => {
    setIsProcessing(false);
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

  const handleChoice = (choiceId) => {
    if (!choiceRequest) return;
    
    sendCommand('choice_response', {
      character_name: choiceRequest.character_name,
      selected_id: choiceId
    });
    
    setChoiceRequest(null);
  };

  console.log('Current characters:', characters);  // See if state is updated

  return (
    <div className="app-container">
      <div className="character-panels">
        <TabPanel characters={Object.values(characters)
          .filter(character => character && character.name)
          .map(character => (
            <CharacterPanel 
              key={character.name} 
              character={character}
              sessionId={sessionId}
              sendCommand={sendCommand}
            />
          ))}
        />
      </div>
      <div className="center-panel">
        <div className="world-container">
          <div className="world-header">
            <div className="world-panel">
              <WorldPanel worldState={worldState} />
            </div>
            <div className="world-description">
              {worldState.show}
            </div>
          </div>
          <div className="log-area" ref={logRef}>
            {logText}
          </div>
        </div>
      </div>
      <div className="control-panel">
        <div className="control-buttons">
          <button className="control-button" onClick={() => sendCommand('initialize')}>Initialize Play</button>
          <button className="control-button" 
            onClick={handleRun} 
            disabled={isProcessing || simState.running}>Run</button>
          <button className="control-button" 
            onClick={handleStep} 
            disabled={isProcessing || simState.running}>Step</button>
          <button className="control-button" onClick={handlePause}>Pause</button>
          <button onClick={handleRefresh} className="control-button">
            Refresh
          </button>
          <button className="control-button">Load World</button>
          <button className="control-button">Save World</button>
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
          <select onChange={(e) => setSelectedPlay(e.target.value)}>
            {plays.map(play => (
              <option key={play} value={play}>{play}</option>
            ))}
          </select>
          <button onClick={() => {
            setIsProcessing(true);
            sendCommand('load_play', { play: selectedPlay });
            setShowPlayDialog(false);
          }}>Load</button>
          <button onClick={() => setShowPlayDialog(false)}>Cancel</button>
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

      {choiceRequest && (
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
    </div>
  );
}

export default App;
