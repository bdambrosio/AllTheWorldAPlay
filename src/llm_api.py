import os, sys, re, traceback, requests, json
import random
import socket
import time
import numpy as np
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
from utils.LLMRequestOptions import LLMRequestOptions
from PIL import Image
from io import BytesIO

response_prime_needed = False
tabby_api_key = os.getenv("TABBY_API_KEY")

def generate_image(description, size='1024x1024', filepath="../images/test.png"):
    url = 'http://127.0.0.1:5008/generate_image'
    response =  requests.get(url, params={"prompt":description, "size":size})
    if response.status_code == 200:
        image_content = response.content
    # Save the image to a file
    with open(filepath, "wb") as file:
        file.write(image_content)
    return filepath

url = 'http://127.0.0.1:5000/v1/chat/completions'
tabby_api_key = os.getenv("TABBY_API_KEY")
headers = {'x-api-key':tabby_api_key}

pattern = r'\{\$[^}]*\}'

class LLM():

    def run_request(self, bindings, prompt, options):
        #
        ### first substitute for {{$var-name}} in prompt
        #
        global pattern
        #print(f'run_request {bindings}\n{prompt}\n')
        if bindings is None or len(bindings) == 0:
            substituted_prompt = [{'role':message.role, 'content':message.content} for message in prompt]
        else:
            substituted_prompt = []
            for message in prompt:
                if type(message.content) is not str: # substitutions only in strings
                    substituted_prompt.append({'role':message.role, 'content':message.content})
                    continue
                matches = re.findall(pattern, message.content)
                new_content = message.content
                for match in matches:
                    var = match[2:-1]
                    #print(f'var {var}, content {new_content}')
                    if var in bindings.keys():
                        new_content = new_content.replace('{{$'+var+'}}', str(bindings[var]))
                    else:
                        raise ValueError(f'unbound prompt variable {var}')
                substituted_prompt.append({'role':message.role, 'content':new_content})
        response =  requests.post(url, headers= headers,
                                  json={"messages":substituted_prompt, "temperature":options.temperature,
                                        "top_p":options.top_p, "max_tokens":options.max_tokens, "stop":options.stops})
        if response.status_code == 200:
            text = response.content.decode('utf-8')
            if text.startswith('{'):
                try:
                    jsonr = json.loads(response.content.decode('utf-8'))
                except Exception as e:
                    print(str(e))
                    return response.content.decode('utf-8')
                return jsonr['choices'][0]['message']['content']
            return response.content.decode('utf-8')
            # assume tabby or other open-ai like return
        else:
            raise Error(response)

    def ask(self, input, prompt_msgs, client=None, template=None, temp=None, max_tokens=None, top_p=None, stops=None, stop_on_json=False):

        if max_tokens is None: max_tokens = 400
        if temp is None: temp = 0.7
        if top_p is None: top_p = 1.0
          
        options = LLMRequestOptions(temperature=temp, top_p=top_p, max_tokens=max_tokens,
                                    stops=stops, stop_on_json=stop_on_json)
        try:
            if response_prime_needed and type(prompt_msgs[-1]) != AssistantMessage:
                prompt_msgs = prompt_msgs + [AssistantMessage(content='')]
            response = self.run_request(input, prompt_msgs, options)
            if stops is not None: # claude returns eos
                if type(stops) is str:
                    stops = [stops]
                    for stop in stops:
                        eos_index=response.rfind(stop)
                        if eos_index > -1:
                            response=response[:eos_index]
            return response
        except Exception as e:
            traceback.print_exc()
            print(str(e))
            return None
       
if __name__ == '__main__':
    llm = LLM()
    response = llm.ask({}, [SystemMessage(content='You are Owl, a smart and helpful bot'),
                            UserMessage(content='Who are you?'),
                            ], max_tokens=10, temp=.4)
    
    print(response)
    
