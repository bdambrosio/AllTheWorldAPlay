import React from 'react';
import './WorldPanel.css';

function WorldPanel({ worldState }) {
  return (
    <div className="world-panel">
      {worldState?.image && (
        <img 
          src={`data:image/png;base64,${worldState.image}`}
          alt="World State"
          className="world-image"
        />
      )}
      <div className="world-description">
        {worldState?.show || ''}
      </div>
    </div>
  );
}

export default WorldPanel; 