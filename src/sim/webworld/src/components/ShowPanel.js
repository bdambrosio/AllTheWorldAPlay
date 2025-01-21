import React from 'react';
import './ShowPanel.css';

function ShowPanel({ characters }) {
  return (
    <div className="show-panel">
      {Object.values(characters).map((char, index) => (
        char.show && (
          <div key={index} className="show-entry">
            <span className="character-name">{char.name}:</span> {char.show}
          </div>
        )
      ))}
    </div>
  );
}

export default ShowPanel; 