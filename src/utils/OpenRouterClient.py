import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os
from utils.Messages import SystemMessage
from utils.Messages import AssistantMessage
from openai import OpenAI
from utils.LLMRequestOptions import LLMRequestOptions

api_key=os.getenv("OPENROUTER_API_KEY")
model= "meta-llama/llama-4-maverick"

class OpenRouterClient():
    DefaultEndpoint = 'https://openrouter.ai'
    UserAgent = 'Owl'

    def __init__(self, api_key=None):
        self.api_key = api_key
        
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
            model = 'meta-llama/llama-4-maverick'
        try:

            response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://tuuyi.com", # Optional. Site URL for rankings on openrouter.ai.
                        "X-Title": "Tuuyi", # Optional. Site title for rankings on openrouter.ai.
                    },
                data=json.dumps({
                    "model": model, # Optional
                    "messages": prompt,
                    "max_tokens": options.max_tokens,
                    "temperature": options.temperature,
                    "top_p": options.top_p,
                    "stop": stops,
                    "stream": False,
                         })
                        )
            item = response.content
            item = json.loads(item.decode('utf-8'))
            item = item['choices'][0]['message']['content']
            return item
            #return {"status":'success', "message":{"role":'assistant', "content":item.content}}
        except Exception as e:
            return {"status":'error', "message":str(e)}
            

