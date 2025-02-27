import React, { useState, useEffect } from 'react';
import CharacterNarrative from './CharacterNarrative';
import ExplorerModal from './ExplorerModal';
import './CharacterPanel.css';

function CharacterPanel({ character, sessionId, sendCommand }) {
  const [showNarrative, setShowNarrative] = useState(false);
  const [showExplorer, setShowExplorer] = useState(false);
  const [lastExplorerState, setLastExplorerState] = useState(null);
  const [explorerStatus, setExplorerStatus] = useState('idle');

  // Cache explorer state from character updates
  useEffect(() => {
    if (character?.explorer_state) {
      setLastExplorerState(character.explorer_state);
      setExplorerStatus('idle');
    }
  }, [character]);

  // Handle explorer modal open
  const handleExplorerOpen = async () => {
    setShowExplorer(true);
    
    // Use cached state if running or processing
    if (character.status === 'processing' || character.status === 'running') {
      console.log('Using cached explorer state');
      return;
    }

    setExplorerStatus('loading');
    try {
      // Use sendCommand instead of dispatching event
      await sendCommand('get_character_details', { name: character.name });
    } catch (err) {
      console.error('Error requesting explorer state:', err);
      setExplorerStatus('error');
    }
  };

  if (!character) {
    return <div className="character-panel">Loading...</div>;
  }

  return (
    <div className="character-panel">
      <div 
        className="character-image-container"
        onMouseEnter={() => setShowNarrative(true)}
        onMouseLeave={() => setShowNarrative(false)}
      >
        <div className="name-column">
          <div className="character-name">{character.name}</div>
          <div className="character-signals">{character.signals}</div>
        </div>
        <img 
          src={character.image} 
          alt={character.name} 
          className="character-image"
        />
        {showNarrative && character.narrative && (
          <CharacterNarrative narrative={character.narrative} />
        )}
      </div>
      
      <div className="middle-section">
        <div className="section-container">
          <div className="tasks-area">
            {character.tasks?.[0]}
          </div>
        </div>
        <div className="section-container">
          <div className="tasks-area">
            {character.tasks?.[1]}
          </div>
        </div>
        <div className="section-container">
          <div className="history-area">
            <div className="history-item">
              {character.history}
            </div>
          </div>
        </div>
      </div>
      
      <div className="thoughts-area">
        <div className="thoughts-content">
          {character.thoughts}
        </div>
      </div>

      <button 
        className="explore-button"
        onClick={handleExplorerOpen}
      >
        Explore Character State
      </button>

      {showExplorer && (
        <ExplorerModal
          character={character}
          sessionId={sessionId}
          lastState={lastExplorerState}
          status={explorerStatus}
          onClose={() => setShowExplorer(false)}
          sendCommand={sendCommand}
        />
      )}
    </div>
  );
}

export default CharacterPanel; 