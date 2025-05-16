import React, { useState, useEffect } from 'react';
import { useReplay } from '../contexts/ReplayContext';

const TabPanel = ({ characters, sendReplayEvent, activeTab, onTabChange }) => {
    console.log('TabPanel render - activeTab:', activeTab);  // Debug log
    console.log('TabPanel render - characters:', characters);  // Debug log
    
    const { recordEvent } = useReplay();
    const [pendingTabEvent, setPendingTabEvent] = useState(null);

    const handleTabClick = (index) => {
        console.log('Tab clicked:', index);  // Debug log
        onTabChange(index);
        const characterName = characters[index].props.character.name;
        recordEvent('select_tab', characterName, 'select_tab');
        sendReplayEvent('setActiveTab', { 
            panelId: 'characterTabs', 
            characterName: characterName 
        });
    };

    useEffect(() => {
        if (pendingTabEvent) {
            const characterArray = Object.values(characters);
            const characterIndex = characterArray.findIndex(char => char.name === pendingTabEvent.arg.characterName);
            if (characterIndex !== -1) {
                onTabChange(characterIndex);
                setPendingTabEvent(null); // Clear the pending event
            }
        }
    }, [characters, pendingTabEvent, onTabChange]);

    return (
        <div className="tab-panel">
            <div className="tabs">
                {characters.map((char, index) => {
                    console.log('Rendering tab:', index, 'active:', index === activeTab);  // Debug log
                    return (
                        <button 
                            key={char.props.character.name}
                            className={`tab ${index === activeTab ? 'active' : ''}`}
                            onClick={() => handleTabClick(index)}
                        >
                            {char.props.character.name}
                        </button>
                    );
                })}
            </div>
            <div className="tab-content">
                {characters[activeTab]}
            </div>
        </div>
    );
};

export default TabPanel; 