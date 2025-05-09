const getBackendHost = () => {
    // If running locally, use localhost
    if (window.location.hostname === 'localhost') {
        return 'http://localhost:8000';
    }
    // Otherwise, replace -3000 with -8000 in the hostname for RunPod proxy
    return `https://${window.location.hostname.replace('-3000', '-8000')}`;
};

const config = {
    serverUrl: getBackendHost()
};

export default config;