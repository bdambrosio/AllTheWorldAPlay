import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.Messages import SystemMessage
from utils.Messages import AssistantMessage
from openai import OpenAI
from utils.LLMRequestOptions import LLMRequestOptions

client = None

api_key = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
except Exception as e:
    print(f"Error getting OpenAI API key: {e}")

if api_key is not None and api_key != '':
    try:
        # Configure client with timeout and retry settings
        client = OpenAI(
            api_key=api_key,
            timeout=30.0,  # 60 second timeout
            max_retries=2   # Retry up to 3 times
        )
    except Exception as e:
        print(f"Error opening OpenAI client: {e}")
model = 'gpt-4.1-mini'

class OpenAIClient():
    DefaultEndpoint = 'https://api.openai.com'
    UserAgent = 'Owl'

    def __init__(self, client=client, api_key=None):
        self._session = requests.Session()
        self._client = client
        
    def executeRequest(self, prompt, options: LLMRequestOptions):
        startTime = time.time()
        if options.stops: #options.stop only allows a single stop string for now.
            stops=options.stops
        else:
            stops=[]
        if options.stop_on_json:
            stops.append('}')
        if options.model is not None:
            model = options.model
        else:
            model = 'gpt-4.1-mini'
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model, 
                    messages=prompt,
                    max_tokens=options.max_tokens, 
                    temperature=options.temperature, 
                    top_p=options.top_p,
                    stop=options.stops, 
                    stream=False,
                    timeout=60.0  # Per-request timeout override
                )
                item = response.choices[0].message
                return item.content
            except Exception as e:
                print(f"OpenAI request error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {"status":'error', "message":str(e)}
                # Exponential backoff: 1s, 2s, 4s
                time.sleep(2 ** attempt)
        
        return {"status":'error', "message":"Request failed after all retries"}
            

