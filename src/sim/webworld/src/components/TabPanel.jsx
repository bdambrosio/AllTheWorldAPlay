import React, { useState } from 'react';

const TabPanel = ({ characters }) => {
    const [activeTab, setActiveTab] = useState(0);

    return (
        <div className="tab-panel">
            <div className="tabs">
                {characters.map((char, index) => (
                    <button 
                        key={char.props.character.name}
                        className={`tab ${index === activeTab ? 'active' : ''}`}
                        onClick={() => setActiveTab(index)}
                    >
                        {char.props.character.name}
                    </button>
                ))}
            </div>
            <div className="tab-content">
                {characters[activeTab]}
            </div>
        </div>
    );
};

export default TabPanel; 