const config = {
    // Use the current hostname for the server URL
    serverUrl: window.location.protocol === 'https:' 
        ? `https://${window.location.hostname}:8000`
        : `http://${window.location.hostname}:8000`
};

export default config;