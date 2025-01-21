import React from 'react';
import './CharacterPanel.css';

function CharacterPanel({ characters }) {
  return (
    <div className="character-info">
      {Object.entries(characters).map(([name, char]) => (
        <div key={name} className="character-block">
          <div className="top-bar">
              <span className="char-name">{name}</span>
              <img 
                src={`data:image/png;base64,${char.image}`}
                alt={name}
                className="character-image"
              />
          </div>
          
          <div className="middle-section">
            <div className="section-container">
              <h4>Priorities</h4>
              <div className="priorities-area">
                {char.priorities?.map((priority, pIndex) => (
                  <div key={pIndex} className="priority-item">
                    {priority}
                  </div>
                ))}
              </div>
            </div>
            <div className="section-container">
              <h4>History</h4>
              <div className="history-area">
                <div className="history-item">
                  {char.history}
                </div>
              </div>
            </div>
          </div>
          
          <div className="thoughts-area">
            <h4>Thoughts</h4>
            {char.thoughts}
          </div>
        </div>
      ))}
    </div>
  );
}

export default CharacterPanel; 