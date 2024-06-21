import requests, time, copy
import traceback
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os, sys
from utils.Messages import SystemMessage, AssistantMessage, asdicts
from utils.LLMRequestOptions import LLMRequestOptions
import anthropic

api_key = os.environ["CLAUDE_API_KEY"]
template = "claude-sonnet"
client = anthropic.Client(api_key=api_key)
# quick test of Claude API
"""
message = client.messages.create(model="claude-3-sonnet-20240229",max_tokens=1024,messages=[{"role": "user", "content": "Hello, Claude"}])
message = client.messages.create(model="claude-3.5-sonnet-20240620",max_tokens=1024,messages=[{"role": "user", "content": "Hello, Claude"}])
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
    #
    while m <= len(msgs)-2:
        if msgs[m]['role'] == 'user' and msgs[m+1]['role'] =='user':
            msgs[m]['content'] = msgs[m]['content']+'\n'+msgs[m+1]['content']
            msgs = msgs[:m+1]+msgs[m+2:]
            #print(f'compressed! {msgs}')
        else:
            m = m+1
            
    print(f'claude model {template}', end=' ')
    try:
        if 'opus' in template:
            print('using opus')
            response = client.messages.create(model="claude-3-opus-20240229",
                                                   messages = msgs,
                                                   system = system_msg,
                                                   temperature=temp,
                                                   stop_sequences = stop_sequences,
                                                   max_tokens = max_t)
            return json.loads(response.json())["content"][0]["text"]
        
        
        elif 'sonnet' in template:
            print('using sonnet')
            try:
                response = client.messages.create(model="claude-3-5-sonnet-20240620",
                                                       messages = msgs,
                                                       system = system_msg,
                                                       temperature=temp,
                                                       stop_sequences = stop_sequences,
                                                       max_tokens = max_t)
                return json.loads(response.json())["content"][0]["text"]
            except Exception as e:
                return {"status":'error', "message":{"content": str(e)}}
            
        elif 'haiku' in template:
            print('using sonnet')
            response = client.messages.create(model="claude-3-haiku-20240307",
                                                   messages = msgs,
                                                   system = system_msg,
                                                   temperature=temp,
                                                   stop_sequences = stop_sequences,
                                                   max_tokens = max_t)
            return json.loads(response.json())["content"][0]["text"]

    except Exception as e:
        print (f'Anthropic error {str(e)}')
        return {"status":'error', "message":{"content": str(e)}}
    
