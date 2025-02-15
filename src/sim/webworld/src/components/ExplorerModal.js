import React, { useState, useEffect } from 'react';
import './ExplorerModal.css';

function ExplorerModal({ character, sessionId, onClose }) {
  const [explorerState, setExplorerState] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('core');

  useEffect(() => {
    async function fetchExplorerState() {
      try {
        if (!sessionId) {
          setError("No active session");
          return;
        }
        
        const response = await fetch(
          `http://localhost:8000/api/character/${character.name}/details?session_id=${sessionId}`
        );
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to fetch character details');
        }
        
        const data = await response.json();
        setExplorerState(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    }
    fetchExplorerState();
  }, [character.name, sessionId]);

  return (
    <div className="explorer-modal">
      <div className="modal-header">
        <h3>{character.name} - State Explorer</h3>
        <button onClick={onClose}>×</button>
      </div>

      <div className="modal-tabs">
        <button 
          className={activeTab === 'core' ? 'active' : ''} 
          onClick={() => setActiveTab('core')}
        >
          Core State
        </button>
        <button 
          className={activeTab === 'memory' ? 'active' : ''} 
          onClick={() => setActiveTab('memory')}
        >
          Memory
        </button>
        <button 
          className={activeTab === 'debug' ? 'active' : ''} 
          onClick={() => setActiveTab('debug')}
        >
          Debug
        </button>
        <button 
          className={activeTab === 'social' ? 'active' : ''} 
          onClick={() => setActiveTab('social')}
        >
          Social
        </button>
        <button 
          className={activeTab === 'cognitive' ? 'active' : ''} 
          onClick={() => setActiveTab('cognitive')}
        >
          Cognitive
        </button>
      </div>

      <div className="modal-content">
        {activeTab === 'core' && explorerState && (
          <div className="core-state">
            <h4>Current Task</h4>
            <div>{explorerState.currentTask || 'None'}</div>
            
            <h4>Drives</h4>
            <div className="drives-list">
              {explorerState.drives.map((drive, i) => (
                <div key={i} className="drive-item">
                  <div className="drive-text">{drive.text}</div>
                </div>
              ))}
            </div>

            <h4>Current Perceptions</h4>
            <div className="perceptions-list">
              {explorerState.emotional_state.map((percept, i) => (
                <div key={i} className="percept-item">
                  <div className="percept-header">
                    <span className="percept-mode">{percept.mode}</span>
                    <span className="percept-time">{new Date(percept.time).toLocaleString()}</span>
                  </div>
                  <div className="percept-content">{percept.text}</div>
                </div>
              ))}
            </div>

            <h4>Last Action</h4>
            <div className="last-action">
              <div>Action: {explorerState.lastAction.name}</div>
              <div>Result: {explorerState.lastAction.result}</div>
              <div>Reason: {explorerState.lastAction.reason}</div>
            </div>

            <h4>Current Intentions</h4>
            <div className="intentions-list">
              {explorerState.intentions.map((intention, i) => (
                <div key={i} className="intention">
                  {intention}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'memory' && explorerState && (
          <div className="memory-state">
            <h4>Recent Memories</h4>
            <div className="memories-list">
              {explorerState.memories.map((memory, i) => (
                <div key={i} className="memory-item">
                  <div className="memory-time">{new Date(memory.timestamp).toLocaleString()}</div>
                  <div className="memory-text">{memory.text}</div>
                </div>
              ))}
            </div>

            <h4>Narrative Summary</h4>
            <div className="narrative-summary">
              <h5>Recent Events</h5>
              <div className="narrative-section">{explorerState.narrative.recent_events}</div>
              
              <h5>Ongoing Activities</h5>
              <div className="narrative-section">{explorerState.narrative.ongoing_activities}</div>
            </div>
          </div>
        )}

        {activeTab === 'debug' && (
          <pre className="debug-view">
            {JSON.stringify(explorerState, null, 2)}
          </pre>
        )}

        {activeTab === 'social' && explorerState && (
          <div className="social-state">
            <h4>Known Characters</h4>
            <div className="actors-list">
              {explorerState.social.known_actors.map((actor, i) => (
                <div key={i} className="actor-item">
                  <div className="actor-header">
                    <span className="actor-name">{actor.name}</span>
                    <span className="actor-relationship">{actor.relationship}</span>
                  </div>
                  {actor.dialog && (
                    <div className="dialog-section">
                      <div className="dialog-status">
                        Dialog Active: {actor.dialog.active ? 'Yes' : 'No'}
                      </div>
                      {actor.dialog.transcript && (
                        <div className="dialog-transcript">
                          {actor.dialog.transcript}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'cognitive' && explorerState && (
          <div className="cognitive-state">
            <h4>Goals</h4>
            <div className="goals-list">
              {explorerState.cognitive.goals.map((goal, i) => (
                <div key={i} className="goal-item">
                  <div className="goal-name">{goal.name}</div>
                  <div className="goal-details">
                    <div className="goal-urgency">Urgency: {goal.urgency}</div>
                    <div className="goal-trigger">Trigger: {goal.trigger}</div>
                    <div className="goal-termination">Termination: {goal.termination}</div>
                    <div className="goal-drive">Drive: {goal.drive}</div>
                  </div>
                </div>
              ))}
            </div>

            <h4>Tasks</h4>
            <div className="priorities-list">
              {explorerState.cognitive.priorities.map((priority, i) => (
                <div key={i} className="priority-item">
                  <div className="priority-name">{priority.name}</div>
                  <div className="priority-description">{priority.description}</div>
                  <div className="priority-reason">{priority.reason}</div>
                  <div className="priority-actors">Actors: {priority.actors}</div>
                  {priority.needs && (
                    <div className="priority-needs">Needs: {priority.needs}</div>
                  )}
                  <div className="priority-committed">
                    {priority.committed ? '✓ Committed' : '○ Not committed'}
                  </div>
                </div>
              ))}
            </div>

            <h4>Intentions</h4>
            <div className="intentions-list">
              {explorerState.cognitive.intentions.map((intention, i) => (
                <div key={i} className="intention-item">{intention}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ExplorerModal; 