import React from 'react';
import './DirectorChairModal.css';

function DirectorChairModal({ characters, onClose, sendCommand }) {
  const controlTypes = ['Signal', 'Goal', 'Task', 'Action'];

  return (
    <div className="director-chair-modal">
      <div className="director-chair-content">
        <div className="director-chair-header">
          <h3>Director's Chair</h3>
          <div>
            <button onClick={() => {
              const autonomySettings = {};
              Object.values(characters).forEach(character => {
                autonomySettings[character.name] = {
                  signal: document.querySelector(`[data-character="${character.name}"][data-control="Signal"]`).classList.contains('automatic'),
                  goal: document.querySelector(`[data-character="${character.name}"][data-control="Goal"]`).classList.contains('automatic'),
                  task: document.querySelector(`[data-character="${character.name}"][data-control="Task"]`).classList.contains('automatic'),
                  action: document.querySelector(`[data-character="${character.name}"][data-control="Action"]`).classList.contains('automatic')
                };
              });
              sendCommand('set_autonomy', { autonomy: autonomySettings });
              onClose();
            }}>Save</button>
            <button onClick={onClose}>×</button>
          </div>
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
                      data-character={character.name}
                      data-control={controlType}
                      onClick={(e) => {
                        e.target.classList.toggle('automatic');
                        e.target.classList.toggle('manual');
                        e.target.textContent = e.target.classList.contains('automatic') ? 'Autonomous' : 'Manual';
                      }}
                    >
                      Autonomous
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