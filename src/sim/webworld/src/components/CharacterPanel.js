import React, { useState } from 'react';
import CharacterNarrative from './CharacterNarrative';
import './CharacterPanel.css';

function CharacterPanel({ character }) {
  const [showNarrative, setShowNarrative] = useState(false);

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
          <h4>Priorities</h4>
          <div className="priorities-area">
            {character.priorities?.map((priority, index) => (
              <div key={index} className="priority-item">
                {priority}
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
    </div>
  );
}

export default CharacterPanel; 