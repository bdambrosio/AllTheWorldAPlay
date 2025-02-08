import React from 'react';
import './WorldPanel.css';

function WorldPanel({ worldState }) {
  return (
    <div>
      {worldState?.image && (
        <img 
          src={`data:image/png;base64,${worldState.image}`}
          alt="World State"
          className="world-image"
        />
      )}
     </div>
  );
}

export default WorldPanel; 