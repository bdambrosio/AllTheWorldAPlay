import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os
from utils.Messages import SystemMessage
from utils.Messages import AssistantMessage
from openai import OpenAI
from utils.LLMRequestOptions import LLMRequestOptions

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIClient():
    DefaultEndpoint = 'https://api.deepseek.com/v1'
    UserAgent = 'Owl'

    def __init__(self, client, api_key=None):
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

        try:
            response = client.chat.completions.create(
                model='deepseek-chat', messages=prompt,
                max_tokens=options.max_tokens, temperature=options.temperature, top_p=options.top_p,
                stop=options.stops, stream=False, response_format = { "type": "json_object" })
            item = response.choices[0].message
            return item.content
            #return {"status":'success', "message":{"role":'assistant', "content":item.content}}
        except Exception as e:
            return {"status":'error', "message":str(e)}
            

