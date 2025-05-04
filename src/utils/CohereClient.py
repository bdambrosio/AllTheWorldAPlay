import requests, time, copy
import traceback
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os, sys
from utils.Messages import SystemMessage, AssistantMessage, asdicts
from utils.LLMRequestOptions import LLMRequestOptions
import cohere

api_key = None
try:
    api_key = os.environ["COHERE_API_KEY"]
except Exception as e:
    print(f"Error getting Cohere API key: {e}")

from cohere import ClientV2

client = None
if api_key is not None and api_key != '':
    try:
        client = ClientV2(api_key=api_key)        
    except Exception as e:
        print(f"Error creating Cohere client: {e}")
# quick test of Claude API

def executeRequest(prompt: list, options: LLMRequestOptions):
        
    ### form claude prompt
    temp = .1; stop_sequences = []; max_t = 50
    system_msg=''  # system message text
    if options.temperature is not None:
        temp=float(options.temperature)
    if options.max_tokens is not None:
        max_t= int(options.max_tokens)
    if options.stops is not None:
        stop_sequences = options.stops

    try:
        response = client.chat(
        model="command-a-03-2025",
        messages=prompt,
        max_tokens=max_t,
        temperature=temp,
        stop_sequences=stop_sequences
    )
        print (f'Cohere response text: {response.message.content[0].text}')
        return response.message.content[0].text

    except Exception as e:
        print (f'Cohere error {str(e)}')
        return {"status":'error', "message":{"content": str(e)}}
    
