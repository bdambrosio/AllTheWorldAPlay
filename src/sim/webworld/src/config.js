const getBackendConfig = () => {
    // If running locally, use localhost with http/ws
    if (window.location.hostname === 'localhost') {
        return {
            httpUrl: 'http://localhost:8000',
            wsProtocol: 'ws',
            wsHost: 'localhost:8000'
        };
    }
    
    // Otherwise, use https/wss with transformed hostname for RunPod proxy
    const transformedHost = window.location.hostname.replace('-3000', '-8000');
    return {
        httpUrl: `https://${transformedHost}`,
        wsProtocol: 'wss',
        wsHost: transformedHost
    };
};

const config = getBackendConfig();

export default config;