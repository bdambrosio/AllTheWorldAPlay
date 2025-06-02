import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.LLMRequestOptions import LLMRequestOptions
from utils.Messages import SystemMessage
from utils.Messages import AssistantMessage
from google import genai
from google.genai import types # type: ignore

client = None
api_key = None
model = 'gemini-2.5-flash-preview-04-17'
try:
    api_key = os.getenv("GOOGLE_KEY")
except Exception as e:
    print(f"No Google API key found")

if api_key and api_key != '':
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error opening Google client: {e}")
    #try:
    #    response = client.models.generate_content(model="gemini-2.5-flash-preview-04-17", contents="Explain how AI works in a few words")
    #    print(response.text)        
    #except Exception as e:
    #    print(f"Error opening OpenAI client: {e}")

class GeminiClient():
    DefaultEndpoint = 'https://api.openai.com'
    UserAgent = 'Owl'

    def __init__(self, client=client, api_key=None):
        self._session = requests.Session()
        self._client = client
        
    def executeRequest(self, prompt, options: LLMRequestOptions):
        global model
        startTime = time.time()
        if options.stops: #options.stop only allows a single stop string for now.
            stops=options.stops
        else:
            stops=[]
        if options.stop_on_json:
            stops.append('}')
        if options.model is not None:
            model = options.model
        text_prompt = '\n'.join([msg['content'] for msg in prompt])
        try:
            response = client.models.generate_content(
                model=model,
                contents=text_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=options.max_tokens,
                    #temperature=0.0,
                    stopSequences=stops
                )
            )
            #print(response.text)
            return response.text
            #return {"status":'success', "message":{"role":'assistant', "content":item.content}}
        except Exception as e:
            return {"status":'error', "message":str(e)}
            

