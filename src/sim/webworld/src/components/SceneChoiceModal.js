import React, { useState, useEffect } from 'react';
import './DirectorChoiceModal.css';

function SceneChoiceModal({ request, onChoice, onClose }) {
  const [sceneData, setSceneData] = useState(null);
  const [formError, setFormError] = useState('');
  const [newCharacterName, setNewCharacterName] = useState('');

  useEffect(() => {
    if (request?.scene_data) {
      setSceneData(JSON.parse(JSON.stringify(request.scene_data))); // Deep clone
    }
  }, [request]);

  if (!request || request.choice_type !== 'scene' || !sceneData) return null;

  const handleInputChange = (field, value) => {
    setFormError('');
    setSceneData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleCharacterGoalChange = (characterName, goal) => {
    setFormError('');
    setSceneData(prev => ({
      ...prev,
      characters: {
        ...prev.characters,
        [characterName]: {
          ...prev.characters[characterName],
          goal: goal
        }
      }
    }));
  };

  const addCharacter = (characterName) => {
    if (!characterName.trim()) return;
    setSceneData(prev => ({
      ...prev,
      characters: {
        ...prev.characters,
        [characterName]: {
          goal: ''
        }
      }
    }));
  };

  const removeCharacter = (characterName) => {
    setSceneData(prev => {
      const newCharacters = { ...prev.characters };
      delete newCharacters[characterName];
      return {
        ...prev,
        characters: newCharacters,
        action_order: prev.action_order.filter(name => name !== characterName)
      };
    });
  };

  const handleActionOrderChange = (value) => {
    setFormError('');
    setSceneData(prev => ({
      ...prev,
      action_order: value.split(',').map(name => name.trim()).filter(name => name)
    }));
  };

  const handleTimeChange = (value) => {
    setFormError('');
    setSceneData(prev => ({
      ...prev,
      time: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Basic validation
    if (!sceneData.scene_title?.trim()) {
      setFormError('Scene title is required');
      return;
    }
    if (!sceneData.location?.trim()) {
      setFormError('Location is required');
      return;
    }
    if (!sceneData.duration || sceneData.duration <= 0) {
      setFormError('Duration must be greater than 0');
      return;
    }

    onChoice('updated', sceneData);
  };

  const handleCancel = () => {
    onChoice('original', request.scene_data);
  };

  return (
    <div className="director-modal">
      <div className="director-modal-content" style={{ maxWidth: '800px', maxHeight: '90vh', overflow: 'auto' }}>
        <div className="director-modal-header">
          <h3>Edit Scene {sceneData.scene_number}: {sceneData.scene_title}</h3>
          <button onClick={onClose}>×</button>
        </div>

        <div className="director-modal-body">
          <form onSubmit={handleSubmit}>
            {formError && <div className="form-error">{formError}</div>}
            
            {/* Basic Scene Information */}
            <div className="form-group">
              <label htmlFor="scene_title">Scene Title:</label>
              <input
                type="text"
                id="scene_title"
                value={sceneData.scene_title || ''}
                onChange={(e) => handleInputChange('scene_title', e.target.value)}
                placeholder="Enter scene title"
              />
            </div>

            <div className="form-group">
              <label htmlFor="location">Location:</label>
              <input
                type="text"
                id="location"
                value={sceneData.location || ''}
                onChange={(e) => handleInputChange('location', e.target.value)}
                placeholder="Enter scene location"
              />
            </div>

            <div className="form-group">
              <label htmlFor="time">Time:</label>
              <input
                type="datetime-local"
                id="time"
                value={sceneData.time ? sceneData.time.slice(0, 16) : ''}
                onChange={(e) => handleTimeChange(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label htmlFor="duration">Duration (minutes):</label>
              <input
                type="number"
                id="duration"
                value={sceneData.duration || ''}
                onChange={(e) => handleInputChange('duration', parseInt(e.target.value) || 0)}
                placeholder="Enter duration in minutes"
                min="1"
              />
            </div>

            {/* Characters and Goals */}
            <div className="form-section">
              <h4>Characters & Goals</h4>
              {Object.entries(sceneData.characters || {}).map(([characterName, characterData]) => (
                <div key={characterName} className="character-goal">
                  <div className="character-goal-header">
                    <h5>{characterName}</h5>
                    <button 
                      type="button" 
                      onClick={() => removeCharacter(characterName)}
                      className="remove-button"
                    >
                      Remove
                    </button>
                  </div>
                  <div className="form-group">
                    <label>Goal:</label>
                    <input
                      type="text"
                      value={characterData.goal || ''}
                      onChange={(e) => handleCharacterGoalChange(characterName, e.target.value)}
                      placeholder="Enter character's goal for this scene"
                    />
                  </div>
                </div>
              ))}
              
              <div className="add-character">
                <input
                  type="text"
                  value={newCharacterName}
                  onChange={(e) => setNewCharacterName(e.target.value)}
                  placeholder="Character name"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      addCharacter(newCharacterName);
                      setNewCharacterName('');
                    }
                  }}
                />
                <button 
                  type="button" 
                  onClick={() => {
                    addCharacter(newCharacterName);
                    setNewCharacterName('');
                  }}
                  className="add-button"
                >
                  Add Character
                </button>
              </div>
            </div>

            {/* Action Order */}
            <div className="form-group">
              <label htmlFor="action_order">Action Order (comma-separated):</label>
              <input
                type="text"
                id="action_order"
                value={sceneData.action_order?.join(', ') || ''}
                onChange={(e) => handleActionOrderChange(e.target.value)}
                placeholder="Enter character names in order of action"
              />
              <small>Available characters: {Object.keys(sceneData.characters || {}).join(', ')}</small>
            </div>

            {/* Narrative */}
            <div className="form-group">
              <label htmlFor="pre_narrative">Pre-Narrative (Scene Setup):</label>
              <textarea
                id="pre_narrative"
                value={sceneData.pre_narrative || ''}
                onChange={(e) => handleInputChange('pre_narrative', e.target.value)}
                placeholder="Describe the situation at the beginning of this scene"
                rows="3"
              />
            </div>

            <div className="form-group">
              <label htmlFor="post_narrative">Post-Narrative (Scene Conclusion):</label>
              <textarea
                id="post_narrative"
                value={sceneData.post_narrative || ''}
                onChange={(e) => handleInputChange('post_narrative', e.target.value)}
                placeholder="Describe the outcome or dominant theme at the end of this scene"
                rows="3"
              />
            </div>

            <div className="form-actions">
              <button type="submit" className="submit-button">
                Update Scene
              </button>
              <button type="button" onClick={handleCancel} className="cancel-button">
                Keep Original
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default SceneChoiceModal; 