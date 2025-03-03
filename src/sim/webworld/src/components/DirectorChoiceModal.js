import React from 'react';
import './DirectorChoiceModal.css';

function DirectorChoiceModal({ request, onChoice, onClose }) {
  if (!request || !['goal', 'task', 'act'].includes(request.choice_type)) return null;

  const { character_name, options } = request;

  return (
    <div className="director-modal">
      <div className="director-modal-content">
        <div className="director-modal-header">
          <h3>{character_name}'s {request.choice_type === 'goal' ? 'Goal' : request.choice_type === 'task' ? 'Task' : 'Action'} Choice</h3>
          <button onClick={onClose}>×</button>
        </div>

        <div className="director-modal-options">
          {options.map(option => (
            <button 
              key={option.id}
              className="choice-option"
              onClick={() => onChoice(option.id)}
            >
              <div className="option-name">{option.name}</div>
              {option.description && <div className="option-description">{option.description}</div>}
              {option.mode && <div className="option-mode">Mode: {option.mode}</div>}
              {option.action && <div className="option-action">Action: {option.action}</div>}
              {option.reason && <div className="option-reason">Reason: {option.reason}</div>}
              {option.target && <div className="option-target">Target: {option.target}</div>}
              {request.choice_type !== 'act' && option.termination && (
                <div className="option-termination">Until: {option.termination}</div>
              )}
              {option.context && (
                <div className="option-context">
                  <div className="signal-cluster">{option.context.signal_cluster}</div>
                  <div className="emotional-stance">
                    {option.context.emotional_stance.arousal}, {option.context.emotional_stance.tone}, {option.context.emotional_stance.orientation}
                  </div>
                </div>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default DirectorChoiceModal; 