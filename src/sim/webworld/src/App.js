import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import CharacterPanel from './components/CharacterPanel';
import ShowPanel from './components/ShowPanel';
import WorldPanel from './components/WorldPanel';
import InjectDialog from './components/InjectDialog';

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
  const [showInjectDialog, setShowInjectDialog] = useState(false);

  useEffect(() => {
    async function initSession() {
      try {
        const response = await fetch('http://localhost:8000/api/session/new');
        const data = await response.json();
        
        websocket.current = new WebSocket(`ws://localhost:8000/ws/${data.session_id}`);
        
        websocket.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log('Msg received:', data.type);
          switch(data.type) {
            case 'character_update':
              setCharacters(prev => ({
                ...prev,
                [data.name]: data.data
              }));
              if (data.data.show) {
                setLogText(prev => {
                  const newEntry = `${data.name}: ${data.data.show}`;
                  return prev ? `${prev}\n\n${newEntry}` : newEntry;
                });
              }
              break;
            case 'context_update':
              setLogText(prev => {
                const newEntry = data.text; 
                return prev ? `${prev}<br><br>${newEntry}` : newEntry;
              });
              break;
            case 'status_update':
              setSimState(data.status);  // Expecting { running: bool, paused: bool }
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
            default:
              console.log('Unknown message type:', data.type);
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

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim() || !websocket.current) return;

    websocket.current.send(inputValue);
    setInputValue('');
  };

  const handleStep = async () => {
    if (websocket.current?.readyState === WebSocket.OPEN) {
      websocket.current.send(JSON.stringify({
        type: 'command',
        action: 'step'
      }));
    }
    setSimState({ running: true, paused: false });
  };

  const handleRun = async () => {
    if (websocket.current?.readyState === WebSocket.OPEN) {
      websocket.current.send(JSON.stringify({
        type: 'command',
        action: 'run'
      }));
    }
  };

  const handlePause = async () => {
    if (websocket.current?.readyState === WebSocket.OPEN) {
      websocket.current.send(JSON.stringify({
        type: 'command',
        action: 'pause'
      }));
    }
  };

  const handleInject = (target, text) => {
    sendCommand('inject', { target, text });
  };

  const sendCommand = async (command, payload = {}) => {
    console.log('sendCommand called:', { command, payload });
    if (websocket.current?.readyState === WebSocket.OPEN) {
      websocket.current.send(JSON.stringify({
        type: 'command',
        action: command,
        ...payload
      }));
    }
  };

  return (
    <div className="app-container">
      <div className="character-panel">
        <CharacterPanel characters={characters} />
      </div>
      <div className="center-panel">
        <WorldPanel worldState={worldState} />
        <div className="log-area">
          {logText}
        </div>
      </div>
      <div className="control-panel">
        <button className="control-button" onClick={() => sendCommand('initialize')}>Initialize Play</button>
        <button className="control-button" onClick={handleRun}>Run</button>
        <button className="control-button" onClick={handleStep}>Step</button>
        <button className="control-button" onClick={handlePause}>Pause</button>
        <button className="control-button" onClick={() => setShowInjectDialog(true)}>Inject</button>
        <button className="control-button">Refresh</button>
        <button className="control-button">Load World</button>
        <button className="control-button">Save World</button>
        <div className="status-area">
          {currentPlay && <div>Play: {currentPlay}</div>}
          <div>Status: {simState.running ? 'Running' : simState.paused ? 'Paused' : 'Paused'}</div>
        </div>
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

      {showInjectDialog && (
        <InjectDialog
          characters={characters}
          onSend={handleInject}
          onClose={() => setShowInjectDialog(false)}
        />
      )}
    </div>
  );
}

export default App;
