import React, { useState, useEffect } from 'react';
import './ExplorerModal.css';

function ExplorerModal({ character, sessionId, lastState, onClose, sendCommand, sendReplayEvent }) {
  const [explorerState, setExplorerState] = useState(lastState);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('core');
  const [chatInput, setChatInput] = useState('');
  const [chatOutput, setChatOutput] = useState(character?.chatOutput || '');

  useEffect(() => {
    // Update state when lastState changes
    if (lastState) {
      setExplorerState(lastState);
    }
  }, [lastState]);

  // Add effect to update chat output when character.chatOutput changes
  useEffect(() => {
    if (character?.chatOutput) {
      setChatOutput(prev => prev ? `${prev}\n${character.chatOutput}` : character.chatOutput);
    }
  }, [character?.chatOutput]);

  const refreshData = async () => {
    try {
      await sendCommand('get_character_details', { name: character.name });
    } catch (e) {
      setError(`Error refreshing data: ${e.message}`);
    }
  };

  const handleTabClick = (tabName) => {
    setActiveTab(tabName);
    
    sendReplayEvent('setExplorerTab', {
      characterName: character.name,
      tabName: tabName
    });
  };

  const handleClose = () => {
    onClose();
    sendReplayEvent('setShowExplorer', { 
      characterName: character.name,
      show: false 
    });
  };

  if (error) return <div className="error">{error}</div>;
  if (!explorerState) return <div>Loading...</div>;

  return (
    <div className="explorer-modal">
      <div className="modal-header">
        <h2>{character.name} State Explorer</h2>
        <div>
          <button onClick={refreshData}>Refresh</button>
          <button onClick={handleClose}>Close</button>
        </div>
      </div>

      <div className="tab-bar">
        <button 
          className={activeTab === 'core' ? 'active' : ''} 
          onClick={() => handleTabClick('core')}
        >
          Core State
        </button>
        <button 
          className={activeTab === 'memory' ? 'active' : ''} 
          onClick={() => handleTabClick('memory')}
        >
          Memory
        </button>
        <button 
          className={activeTab === 'debug' ? 'active' : ''} 
          onClick={() => handleTabClick('debug')}
        >
          Debug
        </button>
        <button 
          className={activeTab === 'social' ? 'active' : ''} 
          onClick={() => handleTabClick('social')}
        >
          Social
        </button>
        <button 
          className={activeTab === 'cognitive' ? 'active' : ''} 
          onClick={() => handleTabClick('cognitive')}
        >
          Cognitive
        </button>
        <button 
          className={activeTab === 'signals' ? 'active' : ''} 
          onClick={() => handleTabClick('signals')}
        >
          Signals
        </button>
        <button 
          className={activeTab === 'chat' ? 'active' : ''} 
          onClick={() => handleTabClick('chat')}
        >
          Chat
        </button>
      </div>

      <div className="modal-content">
        {activeTab === 'core' && explorerState && (
          <div className="core-state">
            <h4>Character Description</h4>
            <div className="character-description">
              {explorerState.character}
            </div>

            <h4>Current Task</h4>
            <div className="task-details">
              {explorerState.currentTask}
            </div>

            <h4>Focus</h4>
            <div className="task-details">
              {(() => {
                const f = explorerState.debug?.focus || {};
                const n = explorerState.debug?.narrative || {};
                return (
                  <>
                    <div>Scene: {n.scene ? `${n.scene.number ?? ''} ${n.scene.title || ''}${n.scene.location ? ' @ ' + n.scene.location : ''}` : 'None'}</div>
                    <div>Act: {n.act ? `${n.act.number ?? ''} ${n.act.title || ''}` : 'None'}</div>
                    <div>Goal: {f.goal ? `${f.goal.name} (progress ${f.goal.progress})` : 'None'}</div>
                    <div>Task: {f.task ? `${f.task.name} (progress ${f.task.progress}, ${f.task.acts_done} acts)` : 'None'}</div>
                    <div>Task stack: {f.task_stack?.length ? f.task_stack.join(' > ') : 'empty'}</div>
                    <div>Selected action: {f.action || 'None'}</div>
                  </>
                );
              })()}
            </div>

            <h4>Drives</h4>
            <div className="drives-list">
              {explorerState.drives?.map((drive, i) => (
                <div key={i} className="drive-item">
                  <div className="drive-text">{drive.text}</div>
                  <div className="drive-activation">
                    <div className="activation-bar" style={{width: `${drive.activation * 100}%`}}></div>
                    <span className="activation-value">{Math.round(drive.activation * 100)}%</span>
                  </div>
                </div>
              ))}
            </div>

            <h4>Character Decisions</h4>
            <div className="decisions-list">
              {explorerState.decisions?.length > 0 ? (
                explorerState.decisions.map((decision, i) => (
                  <div key={i} className="decision-item">
                    {typeof decision === 'object' ? JSON.stringify(decision) : decision}
                  </div>
                ))
              ) : (
                <div className="no-decisions">No decisions made yet</div>
              )}
            </div>

            <h4>Emotional Stance</h4>
            <div className="action-details">{explorerState.emotional_stance || 'None'}</div>

            <h4>Recent Perceptions</h4>
            <div className="perceptions-list">
              {explorerState.percepts?.map((percept, i) => (
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
            <div className="action-details">
              <div>Action: {explorerState.lastAction?.mode ? `${explorerState.lastAction.mode}: ${explorerState.lastAction.action}` : 'None'}</div>
              <div>Result: {explorerState.lastAction?.result || 'None'}</div>
              <div>Reason: {explorerState.lastAction?.reason || 'None'}</div>
            </div>

            <h4>Recent Actions</h4>
            <div className="actions-list">
              {explorerState.actions?.map((action, i) => (
                <div key={i} className="action-item">{action}</div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'memory' && explorerState && (
          <div className="memory-state">
            <h4>Recent Memories</h4>
            <div className="memories-list">
              {explorerState.memories?.map((memory, i) => (
                <div key={i} className="memory-item">
                  <div className="memory-time">{new Date(memory.timestamp).toLocaleString()}</div>
                  <div className="memory-text">{memory.text}</div>
                </div>
              ))}
            </div>

            <h4>Narrative Summary</h4>
            <div className="narrative-summary">
              <h5>Recent Events</h5>
              <div className="narrative-section">{explorerState.narrative?.recent_events || 'No recent events'}</div>
              
              <h5>Ongoing Activities</h5>
              <div className="narrative-section">{explorerState.narrative?.ongoing_activities || 'No ongoing activities'}</div>
            </div>
          </div>
        )}

        {activeTab === 'debug' && (
          <div className="debug-state">
            <h4>Recent LLM Calls{explorerState.debug?.llm_calls?.failed ? ` (${explorerState.debug.llm_calls.failed} failed)` : ''}</h4>
            <div className="llm-calls-list">
              {explorerState.debug?.llm_calls?.recent?.length ? (
                explorerState.debug.llm_calls.recent.map((call, i) => (
                  <div key={i} className="action-item" style={call.ok ? {} : {color: '#c0392b'}}>
                    <b>{call.at} {call.tag}</b> {call.secs}s {call.ok ? '' : 'FAILED'}
                    <div style={{whiteSpace: 'pre-wrap', fontSize: '0.85em'}}>{call.ok ? call.response : call.error}</div>
                  </div>
                ))
              ) : <div>No calls recorded</div>}
            </div>

            <h4>Memory</h4>
            <div className="action-details">
              {(() => {
                const m = explorerState.debug?.memory_stats || {};
                return (
                  <>
                    <div>Concrete: {m.concrete ?? '?'} · Abstract: {m.abstract ?? '?'} · Information items: {m.information_items ?? '?'}</div>
                    <div>Last semantic clustering: {m.last_semantic_clustering || 'never'}</div>
                    {m.recent_abstractions?.map((a, i) => (
                      <div key={i} className="memory-item">[{a.type}, {a.instances}] {a.summary}</div>
                    ))}
                  </>
                );
              })()}
            </div>

            <h4>Autonomy · Location · Visible</h4>
            <div className="action-details">
              <div>Autonomy: {explorerState.debug?.autonomy ? Object.entries(explorerState.debug.autonomy).map(([k, v]) => `${k}=${v ? 'auto' : 'manual'}`).join(', ') : '?'}</div>
              <div>Location: {explorerState.debug?.location ? `(${explorerState.debug.location.x}, ${explorerState.debug.location.y})` : '?'}</div>
              <div>Visible characters: {explorerState.debug?.social?.visible?.join(', ') || 'none'}</div>
              <div>Central narrative: {explorerState.debug?.narrative?.central_narrative || 'None'}</div>
              <div>Goal history: {explorerState.debug?.focus?.goal_history?.join(', ') || 'none'}</div>
            </div>

            <h4>Current View</h4>
            <pre className="debug-view">{explorerState.debug?.look_percept || ''}</pre>

            <h4>Raw State</h4>
            <pre className="debug-view">
              {JSON.stringify(explorerState, null, 2)}
            </pre>
          </div>
        )}

        {activeTab === 'social' && explorerState && (
          <div className="social-state">
            <h4>Known Characters</h4>
            <div className="actors-list">
              {explorerState.social?.known_actors?.map((actor, i) => (
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
              {explorerState.cognitive?.goals?.map((goal, i) => (
                <div className="goal-item" key={goal.id || i}>
                  <div>{goal.focused ? '★ ' : ''}{goal.name}</div>
                  <div className="goal-description">{goal.description}</div>
                  <div className="goal-termination">Termination: {goal.termination}</div>
                  <div className="goal-conditions">Preconditions: {goal.preconditions}</div>
                  <div className="goal-progress">Progress: {goal.progress}</div>
                </div>
              ))}
            </div>

            <h4>Tasks</h4>
            <div className="tasks-list">
              {explorerState.cognitive?.tasks?.map((task, i) => (
                <div key={i} className="task-item">
                  <div className="task-name">{task.name}</div>
                  <div className="task-description">{task.description}</div>
                  <div className="task-reason">Reason: {task.reason}</div>
                  <div className="task-actors">Actors: {task.actors?.join(', ')}</div>
                  <div className="task-needs">Needs: {Array.isArray(task.needs) ? task.needs.join(', ') : (task.needs || 'none')}</div>
                  <div className="task-termination">Termination: {task.termination}</div>
                  <div className="task-progress">Progress: {task.progress} · Acts: {task.acts_done}</div>
                </div>
              ))}
            </div>

            <h4>Intentions</h4>
            <div className="tasks-list">
              {explorerState.cognitive?.intensions?.length ? explorerState.cognitive.intensions.join(', ') : 'none'}
            </div>
          </div>
        )}

        {activeTab === 'signals' && explorerState && (
          <div className="signals-state">
            <h4>Drive Signals</h4>
            <div className="signals-list">
              {explorerState.signals?.map((cluster, i) => (
                <div key={i} className="signal-cluster">
                  <div className="cluster-header">
                    <span className="cluster-type">
                      {({chance: '✓ Opportunity', eased: '✓ Eased', threat: '⚠ Threat'})[(cluster.type || '').toLowerCase()] || cluster.type || '?'}
                    </span>
                    <span className="cluster-metrics">
                      Score: {cluster.score?.toFixed(1)}
                    </span>
                  </div>
                  <div className="cluster-text">{cluster.text}</div>
                  <div className="cluster-drive">
                    Drives: {Array.isArray(cluster.drives) ? cluster.drives.join(', ') : cluster.drive}
                  </div>
                  <div className="cluster-signals">
                    Related signals:
                    {cluster.signals?.map((signal, j) => (
                      <div key={j} className="signal-item">• {signal}</div>
                    ))}
                  </div>
                  <div className="cluster-time">
                    Last seen: {new Date(cluster.last_seen).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="chat-interface">
            <textarea
              className="chat-output"
              value={chatOutput}
              readOnly
              placeholder="Chat responses will appear here..."
            />
            <textarea
              className="chat-input"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Type your message..."
            />
            <button
              className="chat-send-button"
              onClick={() => {
                if (chatInput.trim()) {
                  sendCommand('inject', { 
                    target: character.name,
                    text: chatInput
                  });
                  setChatInput('');
                }
              }}
            >
              Send
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default ExplorerModal; 