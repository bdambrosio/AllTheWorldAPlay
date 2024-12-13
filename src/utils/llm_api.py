import os, sys, re, traceback, requests, json
import random
import socket
import time
from pathlib import Path
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
from utils.LLMRequestOptions import LLMRequestOptions
from PIL import Image
from io import BytesIO
import utils.ClaudeClient as anthropic_client
import utils.OpenAIClient as openai_client
import utils.llcppClient as llcpp_client

response_prime_needed = False
tabby_api_key = os.getenv("TABBY_API_KEY")
url = 'http://127.0.0.1:5000/v1/chat/completions'
tabby_api_key = os.getenv("TABBY_API_KEY")
headers = {'x-api-key': tabby_api_key}
from openai import OpenAI
import openai
openai_api_key = os.getenv("OPENAI_API_KEY")
try:
   openai_api = OpenAI()
   openai_client = openai_client.OpenAIClient(openai_api)
except openai.OpenAIError as e:
   print(e)

IMAGE_PATH = Path.home() / '.local/share/AllTheWorld/images'
IMAGE_PATH.mkdir(parents=True, exist_ok=True)
def generate_image(description, size='512x512', filepath='test.png'):
    cwd = os.getcwd()
    url = 'http://127.0.0.1:5008/generate_image'
    response =  requests.get(url, params={"prompt":description, "size":size})
    if response.status_code == 200:
        image_content = response.content
    # Save the image to a file
    with open(IMAGE_PATH / filepath, "wb") as file:
        file.write(image_content)
    return IMAGE_PATH / filepath

"""
def generate_dalle_image(prompt, size='256x256', filepath='worldsim.png'):
    # Call the OpenAI API to generate the image
    if size != '256x256' and size != '512x512':
        size = '256x256'
    if random.randint(1,4) != 1:
        return filepath
    response = client.images.generate(prompt=prompt, model='dall-e-2',n=1, size=size)
    image_url = response.data[0].url
    image_response = requests.get(image_url)
    image = Image.open(BytesIO(image_response.content))
    filepath = IMAGE_PATH / filepath
    image.save(filepath)
    return filepath
"""

pattern = r'\{\$[^}]*\}'

# options include 'local', 'Claude',
class LLM():
    def __init__(self, llm='local'):
        self.llm = llm
        print(f'will use {self.llm} as llm')
        if llm.startswith('GPT'):
            self.context_size = 32000
        elif llm.startswith('mistral'):
            self.context_size = 32000  # I've read mistral uses sliding 8k window
        elif llm.startswith('claude'):
            self.context_size = 32768  #
        else:
            self.context_size = 8192  # conservative local mis/mixtral default
            try:
                response = requests.post('http://127.0.0.1:5000' + '/template')
                if response.status_code == 200:
                    template = response.json()['template']
                    self.context_size = response.json()['context_size']
                    print(f'template: {template}, context: {self.context_size}, prime: {response_prime_needed}')
            except Exception as e:
                print(f' fail to get prompt template from server {str(e)}')

        if not IMAGE_PATH.exists():
            IMAGE_PATH.mkdir(parents=True, exist_ok=True)
            print(f"Directory '{IMAGE_PATH}' created.")

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
                    if var in bindings.keys():
                        new_content = new_content.replace('{{$'+var+'}}', str(bindings[var]))
                    else:
                        print(f' var not in bindings {var} {bindings}')
                        raise ValueError(f'unbound prompt variable {var}')
                substituted_prompt.append({'role':message.role, 'content':new_content})

        if 'llama.cpp' in self.llm:
            response= llcpp_client.executeRequest(prompt=substituted_prompt, options= options)
            return response
        if 'Claude' in self.llm:
            response= anthropic_client.executeRequest(prompt=substituted_prompt, options= options)
            return response
        if options.model is not None and 'gpt' in options.model:
            response= openai_client.executeRequest(prompt=substituted_prompt, options=options)
            return response
        else:
            if options.stops is None: options.stops = []
            response =  requests.post(url, headers= headers,
                                  json={"messages":substituted_prompt, "temperature":options.temperature,
                                        "top_p":options.top_p, "max_tokens":options.max_tokens, "stop":options.stops})
        if response.status_code == 200:
            text = response.content.decode('utf-8')
            if text.startswith('{'):
                try:
                    jsonr = json.loads(response.content.decode('utf-8'))
                except Exception as e:
                    traceback.print_exc()
                    return response.content.decode('utf-8')
                return jsonr['choices'][0]['message']['content']
            return response.content.decode('utf-8')
            # assume tabby or other open-ai like return
        else:
            traceback.print_exc()
            raise Exception(response)

    def ask(self, input, prompt_msgs, template=None, temp=None, max_tokens=None, top_p=None, stops=None, stop_on_json=False, model=None):

        if max_tokens is None: max_tokens = 400
        if temp is None: temp = 0.7
        if top_p is None: top_p = 1.0
          
        options = LLMRequestOptions(temperature=temp, top_p=top_p, max_tokens=max_tokens,
                                    stops=stops, stop_on_json=stop_on_json, model=model)
        try:
            if response_prime_needed and type(prompt_msgs[-1]) != AssistantMessage:
                prompt_msgs = prompt_msgs + [AssistantMessage(content='')]
            response = self.run_request(input, prompt_msgs, options)
            #response = response.replace('<|im_end|>', '')
            if stops is not None and type(response) is str: # claude returns eos
                if type(stops) is str:
                    stops = [stops]
                    for stop in stops:
                        eos_index=response.rfind(stop)
                        if eos_index > -1:
                            response=response[:eos_index]
            return response
        except Exception as e:
            traceback.print_exc()
            return None
       
if __name__ == '__main__':
    llm = LLM()
    response = llm.ask({}, [UserMessage(content='You are Owl, a smart and helpful bot\n\nWho are you?'),
                            ], max_tokens=10, temp=.4)
    
    print(response)
    
