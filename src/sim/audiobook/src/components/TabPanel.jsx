import React, { useState, useEffect } from 'react';
import { useReplay } from '../contexts/ReplayContext';

const TabPanel = ({ characters, sendReplayEvent, onCharacterClick }) => {
    const { recordEvent } = useReplay();
    const [isCollapsed, setIsCollapsed] = useState(false);

    const handleCharacterClick = (character, index) => {
        const characterName = character.props.character.name;
        recordEvent('select_character', characterName, 'select_character');
        sendReplayEvent('openCharacterModal', { 
            characterName: characterName 
        });
        onCharacterClick(character, index);
    };

    const toggleCollapse = () => {
        setIsCollapsed(!isCollapsed);
    };

    return (
        <div className={`character-list ${isCollapsed ? 'collapsed' : ''}`}>
            <div className="character-list-header">
                <button 
                    className="collapse-toggle"
                    onClick={toggleCollapse}
                    title={isCollapsed ? 'Expand character list' : 'Collapse character list'}
                >
                    {isCollapsed ? '▶' : '◀'}
                </button>
                {!isCollapsed && <span className="header-title">Characters</span>}
            </div>
            {!isCollapsed && (
                <div className="character-names">
                    {characters.map((char, index) => (
                        <button 
                            key={char.props.character.name}
                            className="character-name-button"
                            onClick={() => handleCharacterClick(char, index)}
                            title={`Open ${char.props.character.name} details`}
                        >
                            {char.props.character.name}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
};

export default TabPanel;  