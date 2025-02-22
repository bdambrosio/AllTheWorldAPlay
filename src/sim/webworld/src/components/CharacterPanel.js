import React, { useState, useEffect } from 'react';
import CharacterNarrative from './CharacterNarrative';
import ExplorerModal from './ExplorerModal';
import './CharacterPanel.css';

function CharacterPanel({ character, sessionId }) {
  const [showNarrative, setShowNarrative] = useState(false);
  const [showExplorer, setShowExplorer] = useState(false);
  const [lastExplorerState, setLastExplorerState] = useState(null);
  const [explorerStatus, setExplorerStatus] = useState('idle');

  // Update cached explorer state whenever character updates
  useEffect(() => {
    if (character?.explorer_state) {
      setLastExplorerState(character.explorer_state);
    }
  }, [character]);

  // Handle explorer modal open
  const handleExplorerOpen = async () => {
    setShowExplorer(true);
    
    // Only fetch fresh state if simulation is paused
    if (!character.status || character.status !== 'processing') {
      setExplorerStatus('loading');
      try {
        const res = await fetch(`http://localhost:8000/api/character/${character.name}/details?session_id=${sessionId}`);
        const data = await res.json();
        setLastExplorerState(data);
        setExplorerStatus('idle');
      } catch (err) {
        console.error('Error fetching explorer state:', err);
        setExplorerStatus('error');
      }
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
          <h4>Tasks</h4>
          <div className="tasks-area">
            {character.tasks?.map((task, index) => (
              <div key={index} className="task-item">
                {task}
              </div>
            ))}
          </div>
        </div>
        <div className="section-container">
          <h4>History</h4>
          <div className="history-area">
            <div className="history-item">
              {character.history}
            </div>
          </div>
        </div>
      </div>
      
      <div className="thoughts-area">
        <h4>Thoughts</h4>
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
          onClose={() => setShowExplorer(false)}
        />
      )}
    </div>
  );
}

export default CharacterPanel; 