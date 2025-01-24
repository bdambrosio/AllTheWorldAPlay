import React, { useState } from 'react';
import './InjectDialog.css';

function InjectDialog({ characters, onSend, onClose }) {
  const [selectedTarget, setSelectedTarget] = useState('');
  const [injectText, setInjectText] = useState('');

  const handleSend = () => {
    if (!selectedTarget || !injectText.trim()) return;
    onSend(selectedTarget, injectText);
    setInjectText('');
    onClose();
  };

  const targets = [
    'World',
    'All',
    ...Object.keys(characters)
  ];

  return (
    <div className="dialog">
      <h3>Inject Text</h3>
      <div className="dialog-content">
        <div className="form-group">
          <label>Target:</label>
          <select 
            value={selectedTarget} 
            onChange={(e) => setSelectedTarget(e.target.value)}
          >
            <option value="">Select target...</option>
            {targets.map(target => (
              <option key={target} value={target}>{target}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <textarea
            value={injectText}
            onChange={(e) => setInjectText(e.target.value)}
            placeholder="Enter text to inject..."
            rows={4}
          />
        </div>
        <div className="dialog-buttons">
          <button onClick={onClose}>Cancel</button>
          <button 
            onClick={handleSend}
            disabled={!selectedTarget || !injectText.trim()}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default InjectDialog; 