import React from 'react';
import './DirectorChairModal.css';

function DirectorChairModal({ characters, onClose }) {
  const controlTypes = ['Signal', 'Goal', 'Task', 'Action'];

  return (
    <div className="director-chair-modal">
      <div className="director-chair-content">
        <div className="director-chair-header">
          <h3>Director's Chair</h3>
          <button onClick={onClose}>×</button>
        </div>
        
        <div className="character-columns">
          {Object.values(characters).map(character => (
            <div key={character.name} className="character-column">
              <div className="character-name">{character.name}</div>
              <div className="control-buttons">
                {controlTypes.map(controlType => (
                  <div key={controlType} className="control-row">
                    <span className="control-label">{controlType}</span>
                    <button 
                      className="control-toggle automatic"
                      onClick={(e) => {
                        e.target.classList.toggle('automatic');
                        e.target.classList.toggle('manual');
                        e.target.textContent = e.target.classList.contains('automatic') ? 'Automatic' : 'Manual';
                      }}
                    >
                      Automatic
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default DirectorChairModal; 