// src/sim/webworld/src/contexts/ReplayContext.js
import React, { createContext, useContext, useState } from 'react';

const ReplayContext = createContext({
  isRecording: false,
  events: [],
  startRecording: () => {},
  stopRecording: () => {},
  recordEvent: (type, target) => {}
});

export const ReplayProvider = ({ children }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [events, setEvents] = useState([]);

  const startRecording = () => {
    setIsRecording(true);
    setEvents([]);
  };

  const stopRecording = () => {
    setIsRecording(false);
  };

  const recordEvent = (type, target) => {
    if (isRecording) {
      setEvents(prev => [...prev, {
        timestamp: Date.now(),
        type,
        target
      }]);
    }
  };

  return (
    <ReplayContext.Provider value={{
      isRecording,
      events,
      startRecording,
      stopRecording,
      recordEvent
    }}>
      {children}
    </ReplayContext.Provider>
  );
};

export const useReplay = () => {
  const context = useContext(ReplayContext);
  if (context === undefined) {
    throw new Error('useReplay must be used within a ReplayProvider');
  }
  return context;
};