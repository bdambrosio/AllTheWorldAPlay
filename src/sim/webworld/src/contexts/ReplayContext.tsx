// src/sim/webworld/src/contexts/ReplayContext.tsx
import React, { createContext, useContext, useState } from 'react';

interface ReplayEvent {
  timestamp: number;
  type: string;
  target: string;
  ui_action?: string;  // The UI action to take when replaying this event
}

interface ReplayContextType {
  isRecording: boolean;
  events: ReplayEvent[];
  startRecording: () => void;
  stopRecording: () => void;
  recordEvent: (type: string, target: string) => void;
}

const ReplayContext = createContext({
  isRecording: false,
  events: [] as ReplayEvent[],
  startRecording: () => {},
  stopRecording: () => {},
  recordEvent: (type: string, target: string) => {}
});

export const ReplayProvider = ({ children }: { children: any }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [events, setEvents] = useState([] as ReplayEvent[]);

  const startRecording = () => {
    setIsRecording(true);
    setEvents([]);
  };

  const stopRecording = () => {
    setIsRecording(false);
  };

  const recordEvent = (type: string, target: string) => {
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