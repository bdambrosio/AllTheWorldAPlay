import requests, time, copy
import traceback
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os, sys
from utils.Messages import SystemMessage, AssistantMessage, asdicts
from utils.LLMRequestOptions import LLMRequestOptions
import anthropic

api_key = os.environ["XAI_API_KEY"]
client = anthropic.Client(
  api_key=api_key,
  base_url="https://api.x.ai",
)# quick test of Claude API
"""
message = client.messages.create(model="grok-2-latest",max_tokens=1024,messages=[{"role": "user", "content": "Hello, Claude"}])
message = client.messages.create(model="grok-2-latest",max_tokens=1024,messages=[{"role": "user", "content": "Hello, Grok}])
print(f"\nClaude:\n{message.content}\n\n")
"""

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

    system_msg = ''
    if prompt[0]["role"] == 'system':
        if len(prompt) > 1 and prompt[1]["role"] == 'user':
            # 'system' msgs followed by 'user' msg, change 'system' to messages.create arg
            system_msg = prompt[0]["content"]
            prompt = prompt[1:]
        else:
            #no following user, change role of system msg to user
            prompt[0]["role"] = 'user'
    # claude doesn't like trailing white space on final asst msg
    # below is overkill, but shouldn't hurt
    prompt[-1]["content"] = prompt[-1]["content"].strip()
    
    # Claude doesn't like two user msgs in a row
    m = 0
    msgs = prompt
    print(json.dumps(msgs))
    #
    while m <= len(msgs)-2:
        if msgs[m]['role'] == 'user' and msgs[m+1]['role'] =='user':
            msgs[m]['content'] = msgs[m]['content']+'\n'+msgs[m+1]['content']
            msgs = msgs[:m+1]+msgs[m+2:]
            #print(f'compressed! {msgs}')
        else:
            m = m+1
            
    try:

        response = client.messages.create(#model = 'grok-beta', 
                                          #model = 'grok-mini-beta', 
                                          model="grok-2-latest",
                                          messages = msgs,
                                          stop_sequences = stop_sequences,
                                          max_tokens = max_t)
        #print(json.loads(response.json())["content"][0]["text"])
    except Exception as e:
        print(e)
        return {"status":'error', "message":{"content": str(e)}}

    #print(json.loads(response.json())["content"][0]["text"])
    response_text = json.loads(response.json())["content"][0]["text"].strip()
    if type(stop_sequences) == list and len(stop_sequences) > 0 and response_text.endswith(stop_sequences[0]):
        response_text = response_text[:-len(stop_sequences[0])]
    return response_text
