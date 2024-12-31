import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os
from utils.Messages import SystemMessage
from utils.Messages import AssistantMessage
from openai import OpenAI
from utils.LLMRequestOptions import LLMRequestOptions


class DeepSeekClient():
    DefaultEndpoint = 'https://api.deepseek.com/'


    def __init__(self, api_key=None):
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
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
                model='deepseek-chat', messages=prompt,
                max_tokens=options.max_tokens, temperature=options.temperature, top_p=options.top_p,
                stop=options.stops, stream=False)#, response_format = { "type": "json_object" })
            item = response.choices[0].message
            print(f'****\nresponse:\n{item}')
            return item.content
            #return {"status":'success', "message":{"role":'assistant', "content":item.content}}
        except Exception as e:
            return {"status":'error', "message":str(e)}
            

