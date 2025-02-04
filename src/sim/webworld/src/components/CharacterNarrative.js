import React, { useState } from 'react';
import './CharacterNarrative.css';

function CharacterNarrative({ narrative }) {
    const [activeTab, setActiveTab] = useState('recent');
    
    const tabs = {
        recent: {
            label: 'Recent Events',
            content: narrative.recent_events
        },
        current: {
            label: 'Current',
            content: narrative.ongoing_activities
        },
        relationships: {
            label: 'Relationships',
            content: Object.entries(narrative.relationships)
                .map(([name, desc]) => `${name}: ${desc}`)
                .join('\n')
        },
    };

    return (
        <div className="narrative-popup">
            <div className="narrative-tabs">
                {Object.entries(tabs).map(([key, {label}]) => (
                    <button
                        key={key}
                        className={`tab-button ${activeTab === key ? 'active' : ''}`}
                        onClick={() => setActiveTab(key)}
                    >
                        {label}
                    </button>
                ))}
            </div>
            <div className="tab-content">
                {tabs[activeTab].content}
            </div>
        </div>
    );
}

export default CharacterNarrative; 