
class LLMRequestOptions:
    def __init__(self, temperature=0.1, top_p=1.0, max_tokens=50, stops=[], stop_on_json=False,
                 endpoint=None, host='http://localhost', port=5004, organization=None, logRequests=False):
        self.temperature=temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stops = stops
        self.stop_on_json=stop_on_json
        self.host = '127.0.0.1'
        self.port = 5000
        self.endpoint = endpoint
        self.organization=organization
        self.logRequests=logRequests
        
    def asdict(self):
        return {
            "temperature": self.temperature,
            "top_p":self.top_p,
            "max_tokens":self.max_tokens,
            "stops": self.stops,
            "stop_on_json":self.stop_on_json,
            "port":self.port,
            "host":self.host,
            "endpoint":self.endpoint,
            "organization":self.organization,
            "logRequest":self.logRequests
        }
