import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import CharacterPanel from './components/CharacterPanel';
import ShowPanel from './components/ShowPanel';

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
  const websocket = useRef(null);

  useEffect(() => {
    async function initSession() {
      try {
        const response = await fetch('http://localhost:8000/api/session/new');
        const data = await response.json();
        
        websocket.current = new WebSocket(`ws://localhost:8000/ws/${data.session_id}`);
        
        websocket.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
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
              // Handle simulation status if needed
              break;
            case 'output':
              setMessages(prev => [...prev, data.text]);
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

  const handleInject = async () => {
    if (websocket.current?.readyState === WebSocket.OPEN && inputValue) {
      websocket.current.send(JSON.stringify({
        type: 'command',
        action: 'inject',
        text: inputValue
      }));
      setInputValue('');
    }
  };

  return (
    <div className="app-container">
      <div className="character-panel">
        <CharacterPanel characters={characters} />
      </div>
      <div className="center-panel">
        <div className="log-area">
          {logText}
        </div>
      </div>
      <div className="control-panel">
        <button className="control-button" onClick={handleRun}>Run</button>
        <button className="control-button" onClick={handleStep}>Step</button>
        <button className="control-button" onClick={handlePause}>Pause</button>
        <button className="control-button" onClick={handleInject}>Inject</button>
        <button className="control-button">Refresh</button>
        <button className="control-button">Load World</button>
        <button className="control-button">Save World</button>
      </div>
    </div>
  );
}

export default App;
