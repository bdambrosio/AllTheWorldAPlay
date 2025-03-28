#from matplotlib.hatch import Stars
import os, sys, re, traceback, requests, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import socket
import time
from pathlib import Path
from sympy import Ordinal
from utils.DeepSeekLocalClient import DeepSeekLocalClient
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
from utils.LLMRequestOptions import LLMRequestOptions
from PIL import Image
from io import BytesIO
import utils.ClaudeClient as anthropic_client
import utils.OpenAIClient as openai_client
import utils.llcppClient as llcpp_client
import utils.DeepSeekClient as DeepSeekClient
import utils.DeepSeekLocalClient as DeepSeekLocalClient
import utils.CohereClient as cohere_client
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

import utils.GrokClient as GrokClient
grok_client = GrokClient


deepseek_client = DeepSeekClient.DeepSeekClient()
deepseeklocal_client = DeepSeekLocalClient.DeepSeekLocalClient()
IMAGE_PATH = Path.home() / '.local/share/AllTheWorld/images'
IMAGE_PATH.mkdir(parents=True, exist_ok=True)
vllm_model = 'deepseek-r1-distill-llama-70b-awq'
vllm_model = '/home/bruce/Downloads/models/Qwen2.5-32B-Instruct'
vllm_model = '/home/bruce/Downloads/models/gemma-3-27b-it'

def generate_image(llm=None, description='', size='512x512', filepath='test.png'):

    prompt = [UserMessage(content="""You are a specialized image prompt compressor. 
Your task is to compress detailed scene descriptions into optimal prompts for Stable Diffusion 3.5-large-turbo, which has a 128 token limit.
<scene>
{{$input}}
</scene>
Rules:
The input is either a character description or a scene description. If the former, then the first word is the character name.
Preserve key visual elements and artistic direction, including, if it is a character description, character appearance and emotional state as well askey elements of the background scene.
Prioritize descriptive adjectives and specific nouns
Maintain the core mood/atmosphere
Remove narrative elements that don't affect the visual
Use commas instead of conjunctions where possible
Start with the most important visual elements
Output only the compressed prompt, no explanations

End your response with:
</end>
""")]
    #if llm is None: 
    #    llm = LLM('local')
    #compressed_prompt = llm.ask({"input":description}, prompt, stops=['</End>'])
    compressed_prompt = description
    cwd = os.getcwd()
    url = 'http://127.0.0.1:5008/generate_image'
    response =  requests.get(url, params={"prompt":"phtorealistic style: "+compressed_prompt, "size":size})
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

# options include 'local', 'Claude', 'OpenAI', 'deepseek-chat',
class LLM():
    def __init__(self, server_name='local'):
        global vllm_model
        self.server_name = server_name
        print(f'will use {self.server_name} as llm')
        self.context_size = 16384  # conservative local mis/mixtral default

    if not IMAGE_PATH.exists():
        IMAGE_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{IMAGE_PATH}' created.")

    def run_request(self, bindings, prompt, options):
        global vllm_model
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

        print(f'\n{substituted_prompt}\n')
        if options.model is not None and 'deepseek' in options.model and 'deepseeklocal' not in options.model:
            response= deepseek_client.executeRequest(prompt=substituted_prompt, options=options)
            return response
        if 'deepseek' in self.server_name and 'deepseeklocal' not in self.server_name:
            response= deepseek_client.executeRequest(prompt=substituted_prompt, options=options)
            return response
        if 'llama.cpp' in self.server_name:
            response= llcpp_client.executeRequest(prompt=substituted_prompt, options= options)
            return response
        if 'Claude' in self.server_name:
            response= anthropic_client.executeRequest(prompt=substituted_prompt, options= options)
            return response
        if 'Cohere' in self.server_name:
            response= cohere_client.executeRequest(prompt=substituted_prompt, options= options)
            return response
        if 'Grok' in self.server_name:
            response= GrokClient.executeRequest(prompt=substituted_prompt, options=options)
            return response
        if 'openai' in self.server_name or (options.model is not None and 'gpt' in options.model):
            response= openai_client.executeRequest(prompt=substituted_prompt, options=options)
            return response
        else:
            if options.stops is None: options.stops = []
            if 'deepseeklocal' in self.server_name:
                headers = {"Content-Type": "application/json"}
                url = 'http://localhost:5000/v1/completions'
                content = '\n'.join([msg['content'] for msg in substituted_prompt])
                response =  requests.post(url, headers= headers,
                                          json={"model":vllm_model, 
                                                "prompt":content, "temperature":0.0,
                                               "top_p":options.top_p, "max_tokens":options.max_tokens, "stop":options.stops})
            else:
                url = 'http://localhost:5000/v1/chat/completions'
                response =  requests.post(url, headers={"Content-Type":"application/json"},
                                  json={"messages":substituted_prompt, "temperature":options.temperature,
                                        "top_p":options.top_p, "max_tokens":options.max_tokens, "stop":options.stops})
        if response.status_code == 200:
            if 'deepseeklocal' in self.server_name:
                text = response.json()['choices'][0]['text']
                if (index := text.find('</think>')) > -1:
                    text = text[index+8:].strip()
                return text
            elif 'local' in self.server_name:
                text = response.content.decode('utf-8').strip()
                return text

            if 'deepseeklocal' in self.server_name and (index := text.find('</think>')) > -1:
                text = text[index+8:].strip()
                return text
            if text.startswith('{'):
                try:
                    jsonr = json.loads(text)
                except Exception as e:
                    traceback.print_exc()
                    return response.content.decode('utf-8')
                return jsonr
            return response.content.decode('utf-8')
            # assume tabby or other open-ai like return
        else:
            traceback.print_exc()
            raise Exception(response)

    def ask(self, input, prompt_msgs, template=None, temp=None, max_tokens=None, top_p=None, stops=None, stop_on_json=False, model=None):

        if max_tokens is None: max_tokens = 400
        if temp is None: temp = 0.7
        if top_p is None: top_p = 1.0
          
        start = time.time()
        options = LLMRequestOptions(temperature=temp, top_p=top_p, max_tokens=max_tokens,
                                    stops=stops, stop_on_json=stop_on_json, model=model)
        try:
            if response_prime_needed and type(prompt_msgs[-1]) != AssistantMessage:
                prompt_msgs = prompt_msgs + [AssistantMessage(content='')]
            response = self.run_request(input, prompt_msgs, options)
            #response = response.replace('<|im_end|>', '')
            elapsed = time.time()-start
            #print(f'llm: {elapsed:.2f}')
            if elapsed > 4.0:
                print(f'llm excessive time: {elapsed:.2f}')
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
    
