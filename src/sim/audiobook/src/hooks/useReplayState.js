import { useState, useCallback, useRef } from 'react';

export const REPLAY_STATES = {
  IDLE: 'idle',
  PROCESSING: 'processing', 
  WAITING_SPEECH: 'waiting_speech',
  PAUSED: 'paused',
  ERROR: 'error'
};

export function useReplayState() {
  const [state, setState] = useState(REPLAY_STATES.IDLE);
  const [currentOperation, setCurrentOperation] = useState(null);
  const [canAcceptCommands, setCanAcceptCommands] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const stateRef = useRef(state);
  
  stateRef.current = state;
  
  const handleStateUpdate = useCallback((data) => {
    if (data.type === 'state_update') {
      setState(data.state);
      setCurrentOperation(data.operation);
      setCanAcceptCommands(data.can_accept_commands);
      setIsProcessing(data.is_processing);
      
      console.log(`Replay state: ${data.state} (operation: ${data.operation})`);
    }
  }, []);
  
  const isButtonDisabled = useCallback((buttonType) => {
    if (buttonType === 'pause') {
      return !isProcessing;
    }
    
    return !canAcceptCommands || isProcessing;
  }, [canAcceptCommands, isProcessing]);
  
  const getStatusText = useCallback(() => {
    switch (state) {
      case REPLAY_STATES.IDLE:
        return 'Ready';
      case REPLAY_STATES.PROCESSING:
        return currentOperation ? `Processing ${currentOperation}...` : 'Processing...';
      case REPLAY_STATES.WAITING_SPEECH:
        return 'Playing audio...';
      case REPLAY_STATES.PAUSED:
        return 'Paused';
      case REPLAY_STATES.ERROR:
        return 'Error - click refresh to recover';
      default:
        return 'Unknown state';
    }
  }, [state, currentOperation]);
  
  return {
    state,
    currentOperation,
    canAcceptCommands,
    isProcessing,
    handleStateUpdate,
    isButtonDisabled,
    getStatusText
  };
}
