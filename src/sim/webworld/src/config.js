const getBackendConfig = () => {
    const isLocal   = window.location.hostname === 'localhost';
    const isRunPod  = window.location.hostname.includes('-3000');
  
    // figure out host:port for the API
    let apiHost;
    if (isLocal) {
      apiHost = 'localhost:8000';
    } else if (isRunPod) {
      apiHost = window.location.hostname.replace('-3000', '-8000'); // runpod magic
    } else {
      apiHost = `${window.location.hostname}:8000`;                  // plain VPS
    }
  
    const secure    = window.location.protocol === 'https:';
    const httpUrl   = `${secure ? 'https' : 'http'}://${apiHost}`;
    const wsProtocol = secure ? 'wss' : 'ws';
  
    return { httpUrl, wsProtocol, wsHost: apiHost };
  };
  
  const config = getBackendConfig();
  
  export default config;