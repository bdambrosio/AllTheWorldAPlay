const BACKEND_PORT = process.env.REACT_APP_BACKEND_PORT || '8000';

const getBackendConfig = () => {
    const isLocal   = window.location.hostname === 'localhost';
    const isRunPod  = window.location.hostname.includes('-3000');
  
    // figure out host:port for the API
    let apiHost;
    if (isLocal) {
        apiHost = `localhost:${BACKEND_PORT}`;
    } else if (isRunPod) {
        apiHost = window.location.hostname.replace('-3000', `-${BACKEND_PORT}`); // runpod magic
    } else if (window.location.protocol === 'https:') {
        // Served through a reverse-proxy → hit same origin, no port
        apiHost = window.location.hostname;           // includes :443 only if non-standard
    } else {
        apiHost = `${window.location.hostname}:${BACKEND_PORT}`; // fallback for naked HTTP demos
    }
    const secure    = window.location.protocol === 'https:';
    const httpUrl   = `${secure ? 'https' : 'http'}://${apiHost}`;
    const wsProtocol = secure ? 'wss' : 'ws';
  
    return { httpUrl, wsProtocol, wsHost: apiHost };
  };
  
  const config = getBackendConfig();
  
  export default config;