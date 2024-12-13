import requests, time, copy
import traceback
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os, sys
from utils.Messages import SystemMessage, AssistantMessage, asdicts
from utils.LLMRequestOptions import LLMRequestOptions
from transformers import AutoTokenizer

tokenizer = None
url='http://localhost:5000/completion'

def init (modelpath):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

def executeRequest(prompt: list, options: LLMRequestOptions):
    global config, template
    
    if tokenizer is None:
        print(f' initializing llcpp client')
        init("/home/bruce/Downloads/models/Hermes-2-Theta-Llama-3-8B")
    ### form prompt
    print(f'using llama.cpp')
    temp = .1; stop_sequences = []; max_t = 50
    system_msg=''  # system message text
    if options.temperature is not None:
        temp=float(options.temperature)
    if options.max_tokens is not None:
        max_tokens= int(options.max_tokens)
    if options.stops is not None:
        stop_sequences = options.stops
    
    try:
        rendered = tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False)
        
        #"samplers":['temperature']
        data = {
            "prompt": rendered,#prompt,
            "n_predict": max_tokens,
            "temperature":temp,
            "stop":stop_sequences,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        #print(response)
        print(response.json()['content'])
        return response.json()['content']
    except Exception as e:
        print (f'Anthropic error {str(e)}')
        return {"status":'error', "message":{"content": str(e)}}
    

if __name__=='__main__':
    init("/home/Bruce/Downloads/models/Hermes-2-Theta-Llama-3-8B")
    executeRequest([{'role':'user', 'content':"Hi, how are you today?"}],
                   LLMRequestOptions())
