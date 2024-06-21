import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os
from utils.Messages import SystemMessage
from utils.Messages import AssistantMessage
from utils.Colorize import Colorize
from openai import OpenAI
from utils.LLMRequestOptions import LLMRequestOptions

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIClient():
    DefaultEndpoint = 'https://api.openai.com'
    UserAgent = 'Owl'

    def __init__(self, api_key=None):
        self._session = requests.Session()
        self._client = client
        
    def executeRequest(self, prompt, options: LLMRequestOptions, template:str):
        startTime = time.time()
        if options.stop: #options.stop only allows a single stop string for now.
            stop=[options.stop]
        else:
            stop=[]
        if options.stop_on_json:
            stop.append('}')

        try:
            response = client.chat.completions.create(
                model=template, messages=prompt,
                max_tokens=options.max_tokens, temperature=options.temperature, top_p=options.top_p,
                stop=options.stop, stream=False)
            item = response.choices[0].message
            return {"status":'success', "message":{"role":'assistant', "content":item.content}}
        except Exception as e:
            return {"status":'error', "message":str(e)}
            

