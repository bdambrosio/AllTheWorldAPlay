import React, { useState } from 'react';
import './DirectorChoiceModal.css';

function DirectorChoiceModal({ request, onChoice, onClose }) {
  // State for new goal form
  const [newGoal, setNewGoal] = useState({
    name: '',
    actors: '',
    description: '',
    termination: ''
  });

  // State for new task form
  const [newTask, setNewTask] = useState({
    name: '',
    actors: '',
    description: '',
    reason: '',
    termination: ''
  });

  const [formError, setFormError] = useState('');

  if (!request || !['goal', 'task', 'act'].includes(request.choice_type)) return null;

  const { character_name, options } = request;

  const handleNewGoalChange = (e) => {
    const { name, value } = e.target;
    setFormError(''); // Clear error when user makes changes
    setNewGoal(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleNewTaskChange = (e) => {
    const { name, value } = e.target;
    setFormError(''); // Clear error when user makes changes
    setNewTask(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const validateGoalForm = () => {
    if (!newGoal.name.trim()) {
      setFormError('Goal name is required');
      return false;
    }
    if (!newGoal.actors.trim()) {
      setFormError('At least one actor is required');
      return false;
    }
    if (!newGoal.description.trim()) {
      setFormError('Description is required');
      return false;
    }
    if (!newGoal.termination.trim()) {
      setFormError('Termination condition is required');
      return false;
    }
    return true;
  };

  const validateTaskForm = () => {
    if (!newTask.name.trim()) {
      setFormError('Task name is required');
      return false;
    }
    if (!newTask.actors.trim()) {
      setFormError('At least one actor is required');
      return false;
    }
    if (!newTask.description.trim()) {
      setFormError('Description is required');
      return false;
    }
    if (!newTask.reason.trim()) {
      setFormError('Reason is required');
      return false;
    }
    if (!newTask.termination.trim()) {
      setFormError('Termination condition is required');
      return false;
    }
    return true;
  };

  const handleNewGoalSubmit = (e) => {
    e.preventDefault();
    
    if (!validateGoalForm()) {
      return;
    }

    // Create a custom goal object that matches the format expected by the backend
    const customGoal = {
      name: newGoal.name.trim(),
      actors: newGoal.actors.split(',').map(actor => actor.trim()),
      description: newGoal.description.trim(),
      termination: newGoal.termination.trim()
    };

    // Pass both the 'custom' id and the goal data
    onChoice('custom', customGoal);
  };

  const handleNewTaskSubmit = (e) => {
    e.preventDefault();
    
    if (!validateTaskForm()) {
      return;
    }

    // Create a custom task object
    const customTask = {
      name: newTask.name.trim(),
      actors: newTask.actors.split(',').map(actor => actor.trim()),
      description: newTask.description.trim(),
      reason: newTask.reason.trim(),
      termination: newTask.termination.trim()
    };

    // Pass both the 'custom' id and the task data
    onChoice('custom', customTask);
  };

  return (
    <div className="director-modal">
      <div className="director-modal-content">
        <div className="director-modal-header">
          <h3>{character_name}'s {request.choice_type === 'goal' ? 'Goal' : request.choice_type === 'task' ? 'Task' : 'Action'} Choice</h3>
          <button onClick={onClose}>×</button>
        </div>

        <div className="director-modal-body">
          {/* Existing choices section */}
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

          {/* Goal form - only show for goal choices */}
          {request.choice_type === 'goal' && (
            <div className="new-goal-form">
              <h4>Create New Goal</h4>
              <form onSubmit={handleNewGoalSubmit}>
                {formError && <div className="form-error">{formError}</div>}
                <div className="form-group">
                  <label htmlFor="name">Goal Name:</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={newGoal.name}
                    onChange={handleNewGoalChange}
                    placeholder="Enter goal name"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="actors">Actors (comma-separated):</label>
                  <input
                    type="text"
                    id="actors"
                    name="actors"
                    value={newGoal.actors}
                    onChange={handleNewGoalChange}
                    placeholder="Enter actor names"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="description">Description:</label>
                  <textarea
                    id="description"
                    name="description"
                    value={newGoal.description}
                    onChange={handleNewGoalChange}
                    placeholder="Enter goal description"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="termination">Termination Condition:</label>
                  <input
                    type="text"
                    id="termination"
                    name="termination"
                    value={newGoal.termination}
                    onChange={handleNewGoalChange}
                    placeholder="Enter condition that marks goal completion"
                  />
                </div>
                <button type="submit" className="submit-button">
                  Create New Goal
                </button>
              </form>
            </div>
          )}

          {/* Task form - only show for task choices */}
          {request.choice_type === 'task' && (
            <div className="new-task-form">
              <h4>Create New Task</h4>
              <form onSubmit={handleNewTaskSubmit}>
                {formError && <div className="form-error">{formError}</div>}
                <div className="form-group">
                  <label htmlFor="name">Task Name:</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={newTask.name}
                    onChange={handleNewTaskChange}
                    placeholder="Enter task name"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="actors">Actors (comma-separated):</label>
                  <input
                    type="text"
                    id="actors"
                    name="actors"
                    value={newTask.actors}
                    onChange={handleNewTaskChange}
                    placeholder="Enter actor names"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="description">Description:</label>
                  <textarea
                    id="description"
                    name="description"
                    value={newTask.description}
                    onChange={handleNewTaskChange}
                    placeholder="Enter task description"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="reason">Reason:</label>
                  <input
                    type="text"
                    id="reason"
                    name="reason"
                    value={newTask.reason}
                    onChange={handleNewTaskChange}
                    placeholder="Enter reason for this task"
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="termination">Termination Condition:</label>
                  <input
                    type="text"
                    id="termination"
                    name="termination"
                    value={newTask.termination}
                    onChange={handleNewTaskChange}
                    placeholder="Enter condition that marks task completion"
                  />
                </div>
                <button type="submit" className="submit-button">
                  Create New Task
                </button>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default DirectorChoiceModal; 