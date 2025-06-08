import React, { useState, useEffect } from 'react';
import './DirectorChoiceModal.css';

function ActChoiceModal({ request, onChoice, onClose }) {
  const [actData, setActData] = useState(null);
  const [formError, setFormError] = useState('');

  useEffect(() => {
    if (request?.act_data) {
      setActData(JSON.parse(JSON.stringify(request.act_data))); // Deep clone
    }
  }, [request]);

  if (!request || request.choice_type !== 'act' || !actData) return null;

  const handleInputChange = (field, value) => {
    setFormError('');
    setActData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleGoalChange = (type, value) => {
    setFormError('');
    setActData(prev => ({
      ...prev,
      act_goals: {
        ...prev.act_goals,
        [type]: value
      }
    }));
  };

  const handleTensionPointChange = (index, field, value) => {
    setFormError('');
    setActData(prev => {
      const newTensionPoints = [...prev.tension_points];
      if (field === 'characters') {
        newTensionPoints[index] = {
          ...newTensionPoints[index],
          [field]: value.split(',').map(name => name.trim())
        };
      } else {
        newTensionPoints[index] = {
          ...newTensionPoints[index],
          [field]: value
        };
      }
      return {
        ...prev,
        tension_points: newTensionPoints
      };
    });
  };

  const addTensionPoint = () => {
    setActData(prev => ({
      ...prev,
      tension_points: [
        ...prev.tension_points,
        {
          characters: [],
          issue: '',
          resolution_requirement: 'partial'
        }
      ]
    }));
  };

  const removeTensionPoint = (index) => {
    setActData(prev => ({
      ...prev,
      tension_points: prev.tension_points.filter((_, i) => i !== index)
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Basic validation
    if (!actData.act_title?.trim()) {
      setFormError('Act title is required');
      return;
    }
    if (!actData.act_description?.trim()) {
      setFormError('Act description is required');
      return;
    }

    onChoice('updated', actData);
  };

  const handleCancel = () => {
    onChoice('original', request.act_data);
  };

  return (
    <div className="director-modal">
      <div className="director-modal-content" style={{ maxWidth: '800px', maxHeight: '90vh', overflow: 'auto' }}>
        <div className="director-modal-header">
          <h3>Edit Act {actData.act_number}: {actData.act_title}</h3>
          <button onClick={onClose}>×</button>
        </div>

        <div className="director-modal-body">
          <form onSubmit={handleSubmit}>
            {formError && <div className="form-error">{formError}</div>}
            
            {/* Basic Act Information */}
            <div className="form-group">
              <label htmlFor="act_title">Act Title:</label>
              <input
                type="text"
                id="act_title"
                value={actData.act_title || ''}
                onChange={(e) => handleInputChange('act_title', e.target.value)}
                placeholder="Enter act title"
              />
            </div>

            <div className="form-group">
              <label htmlFor="act_description">Act Description:</label>
              <textarea
                id="act_description"
                value={actData.act_description || ''}
                onChange={(e) => handleInputChange('act_description', e.target.value)}
                placeholder="Enter act description"
                rows="3"
              />
            </div>

            {/* Act Goals */}
            <div className="form-section">
              <h4>Act Goals</h4>
              <div className="form-group">
                <label htmlFor="primary_goal">Primary Goal:</label>
                <input
                  type="text"
                  id="primary_goal"
                  value={actData.act_goals?.primary || ''}
                  onChange={(e) => handleGoalChange('primary', e.target.value)}
                  placeholder="Enter primary goal"
                />
              </div>
              <div className="form-group">
                <label htmlFor="secondary_goal">Secondary Goal:</label>
                <input
                  type="text"
                  id="secondary_goal"
                  value={actData.act_goals?.secondary || ''}
                  onChange={(e) => handleGoalChange('secondary', e.target.value)}
                  placeholder="Enter secondary goal"
                />
              </div>
            </div>

            {/* Pre/Post State */}
            <div className="form-group">
              <label htmlFor="act_pre_state">Pre-State (Beginning):</label>
              <textarea
                id="act_pre_state"
                value={actData.act_pre_state || ''}
                onChange={(e) => handleInputChange('act_pre_state', e.target.value)}
                placeholder="Describe the situation at the beginning of this act"
                rows="2"
              />
            </div>

            <div className="form-group">
              <label htmlFor="act_post_state">Post-State (Ending):</label>
              <textarea
                id="act_post_state"
                value={actData.act_post_state || ''}
                onChange={(e) => handleInputChange('act_post_state', e.target.value)}
                placeholder="Describe the situation at the end of this act"
                rows="2"
              />
            </div>

            {/* Tension Points */}
            <div className="form-section">
              <h4>Tension Points</h4>
              {actData.tension_points?.map((tension, index) => (
                <div key={index} className="tension-point">
                  <div className="tension-point-header">
                    <h5>Tension Point {index + 1}</h5>
                    <button 
                      type="button" 
                      onClick={() => removeTensionPoint(index)}
                      className="remove-button"
                    >
                      Remove
                    </button>
                  </div>
                  <div className="form-group">
                    <label>Characters (comma-separated):</label>
                    <input
                      type="text"
                      value={tension.characters?.join(', ') || ''}
                      onChange={(e) => handleTensionPointChange(index, 'characters', e.target.value)}
                      placeholder="Enter character names"
                    />
                  </div>
                  <div className="form-group">
                    <label>Issue:</label>
                    <input
                      type="text"
                      value={tension.issue || ''}
                      onChange={(e) => handleTensionPointChange(index, 'issue', e.target.value)}
                      placeholder="Describe the conflict or tension"
                    />
                  </div>
                  <div className="form-group">
                    <label>Resolution Requirement:</label>
                    <select
                      value={tension.resolution_requirement || 'partial'}
                      onChange={(e) => handleTensionPointChange(index, 'resolution_requirement', e.target.value)}
                    >
                      <option value="partial">Partial</option>
                      <option value="full">Full</option>
                      <option value="none">None</option>
                    </select>
                  </div>
                </div>
              ))}
              <button type="button" onClick={addTensionPoint} className="add-button">
                Add Tension Point
              </button>
            </div>

            {/* Scenes Display (Read-only for now) */}
            <div className="form-section">
              <h4>Scenes in this Act</h4>
              <div className="scenes-list">
                {actData.scenes?.map((scene, index) => (
                  <div key={index} className="scene-summary">
                    <strong>Scene {scene.scene_number}: {scene.scene_title}</strong>
                    <br />
                    <small>{scene.location} - {scene.duration} minutes</small>
                  </div>
                ))}
              </div>
            </div>

            <div className="form-actions">
              <button type="submit" className="submit-button">
                Update Act
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

export default ActChoiceModal; 