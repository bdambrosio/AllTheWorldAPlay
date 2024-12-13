import os, re, json, sys
import time
from datetime import datetime
import requests
import utils.llm_api as llm_api
from utils.Messages import SystemMessage, UserMessage, AssistantMessage

llm = llm_api.LLM()
def raw(text, eos='</END>', stop_on_json=False):
    #server_message = {'prompt':text, 'temp': 0.01, 'top_p':0.5, "eos":'Pause', 'max_tokens':100}
    server_message = {'prompt':text, 'temp': 0.5, 'top_p':0.7, 'raw':True, 'eos':eos, 'stop_on_json':stop_on_json, 'max_tokens':100}
    response = requests.post('http://127.0.0.1:5000/v1/chat/completions', json=server_message, stream=True)
    print(response.text)

def ask(messages, eos=None, stop_on_json=False):
    prompt =[]
    for message in messages:
        if message['role'] == 'system':
            prompt.append(SystemMessage(content=message['content']))
        elif message['role'] == 'user':
            prompt.append(UserMessage(content=message['content']))
        elif message['role'] == 'assistant':
            prompt.append(AssistantMessage(content=message['content']))
    print(f'cli ask prompt:\n{prompt}\n---')
    print(llm.ask({"name":'joe'}, prompt, temp=0.1, top_p=1.0, stops=eos, max_tokens=200))

if __name__ == '__main__':
    messages = [{"role":'system', "content":"""You are a friendly squirrel named {{$name}}. """},
            {"role":'user', "content":"""Tell me about your day in 25 words or less. End your reply with </END>"""}
            ]

    #ask(messages, eos='</END>')
    raw('Once upon a time')

