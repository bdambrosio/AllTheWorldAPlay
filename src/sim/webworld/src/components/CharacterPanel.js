import React, { useState } from 'react';
import CharacterNarrative from './CharacterNarrative';
import ExplorerModal from './ExplorerModal';
import './CharacterPanel.css';

function CharacterPanel({ character, sessionId }) {
  const [showNarrative, setShowNarrative] = useState(false);
  const [showExplorer, setShowExplorer] = useState(false);

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
        onClick={() => setShowExplorer(true)}
      >
        Explore Character State
      </button>

      {showExplorer && (
        <ExplorerModal
          character={character}
          sessionId={sessionId}
          onClose={() => setShowExplorer(false)}
        />
      )}
    </div>
  );
}

export default CharacterPanel; 