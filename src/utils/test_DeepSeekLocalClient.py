import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os
from utils.Messages import SystemMessage
from utils.Messages import AssistantMessage
from openai import OpenAI
from utils.LLMRequestOptions import LLMRequestOptions


class DeepSeekLocalClient():
    DefaultEndpoint = 'http://localhost:5000/v1'


    def __init__(self, api_key=None):
        client = None
        try:
            client = OpenAI(api_key=api_key, base_url="http://localhost:5000/v1")
            self.model=client.models.list()[0].id

        except Exception as e:
            print(f"Error opening DeepSeekLocal client: {e}")
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
            print(f"****\ntrying deepseek client:\n {prompt}")
            response = self._client.chat.completions.create(
                model=self.model, messages=prompt,
                max_tokens=options.max_tokens, temperature=0, top_p=options.top_p,
                stop=options.stops, stream=False)#, response_format = { "type": "json_object" })
            item = response.json()['choices'][0]['text']
            print(f'****\nresponse:\n{item}')
            return item.content
            #return {"status":'success', "message":{"role":'assistant', "content":item.content}}
        except Exception as e:
            return {"status":'error', "message":str(e)}
            

