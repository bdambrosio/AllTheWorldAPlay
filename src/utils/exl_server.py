import sys, os
import json
import requests
import subprocess
#add local dir to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize model and cache
models_dir = "/home/bruce/Downloads/models/"

subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
models = [d for d in subdirs if 'gguf' not in d]
#print(models)
contexts = {}

model_number = -1

while model_number < 0 or model_number > len(models) -1:
    print(f'Available models:')
    for i in range(len(models)):
        model_name = models[i]
        try:
            with open(models_dir+model_name+'/config.json', 'r') as j:
                json_config = json.load(j)
                if "max_position_embeddings" in json_config:
                    context_size = json_config["max_position_embeddings"]
                elif 'max_seq_len' in json_config:
                    context_size = json_config['max_seq_len']
                else:
                    raise ValueError('no findable context size in config.json')
        except Exception as e:
            context_size = 8192
        contexts[model_name] = context_size
        print(f'{i}. {models[i]}, context: {context_size}')
    
    number = input('input model # to load: ')
    try:
        model_number = int(number)
    except:
        print(f'Enter a number between 0 and {len(models)-1}')

model_name=models[model_number]
context_size = contexts[model_name]
json_config = None


config = ExLlamaV2Config()
config.max_batch_size=1
config.model_dir = models_dir+model_name
config.prepare()

model = ExLlamaV2(config)

if 'llama3-70B' in model_name:
    print(f"Loading model: {model_name}\n context {context_size}")
    model.load([45, 45, 48])
    print('model load done..')

else:
    print(f"Loading model: {model_name}\n context {context_size}")
    model.load([40, 42, 42])
    
    print('model load done..')

hf_tokenizer = AutoTokenizer.from_pretrained(models_dir+model_name)
exl_tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model)
# Initialize generator
generator = ExLlamaV2StreamingGenerator(model, cache, exl_tokenizer)

# Settings
settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.1
#settings.top_k = 50
settings.top_p = 0.8
#settings.token_repetition_penalty = 1.15
#settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

# Make sure CUDA is initialized so we can measure performance
generator.warmup()

async def stream_data(query: Dict[Any, Any], max_new_tokens, stop_on_json=False):
    generated_tokens = 0
    # this is a very sloppy heuristic for complete json form, but since chunks are short, maybe ok?
    # or maybe they aren't so short? Think they are, at least for first json form...
    open_braces = 0
    open_brace_seen = False
    complete_json_seen = False
    text = ''
    stop_strs = []
    if 'stop' in query:
        stop_strs = query['stop']
        if type(stop_strs) is not list:
            stop_strs = [stop_strs]
        stop_strs = [item for item in stop_strs if type(item) is str]
    print(f'stop: {stop_strs}')
    while True:
        chunk, eos, _ = generator.stream()
        chunk = chunk.replace('\\n', '\n') # weirdness in llama-3
        generated_tokens += 1
        if stop_on_json:
            open_braces += chunk.count('{')
            if open_braces > 0:
                open_brace_seen = True
                close_braces = chunk.count('}')
                open_braces -= close_braces
            if open_brace_seen and open_braces == 0:
                complete_json_seen = True
        print (chunk, end = "")
        text += chunk
        yield chunk
                    
        if eos or generated_tokens == max_new_tokens or (stop_on_json and complete_json_seen):
            print('\n')
            break
        #check for stop strings after token decode and chunk re-assembly
        for stop_str in stop_strs:
            test_len = len(stop_str)+len(chunk)
            if stop_str in text[-test_len:]:
                print(f'Stop_str {chunk}')
                break

app = FastAPI()
print(f"starting server")
@app.post("/template")
async def template(request: Request):
    return {"context_size":context_size}
    
@app.post("/v1/chat/completions")
async def get_stream(request: Request):
    global generator, settings, hf_tokenizer, exl_tokenizer
    query = await request.json()
    print(f'request: {query}')
    message_j = query

    if 'template_query' in message_j.keys():
        return Response(template)
    temp = 0.1
    if 'temp' in message_j.keys():
        temp = message_j['temp']
    
    top_p = 1.0
    if 'top_p' in message_j.keys():
        top_p = message_j['top_p']

    max_tokens = 100
    if 'max_tokens' in message_j.keys():
        max_tokens = message_j['max_tokens']

    stop_conditions = ['###','<|endoftext|>', "Reference(s):"]
    if 'stop' in message_j.keys():
        print(f'\n received stop {message_j["stop"]}')
        stop_conditions = message_j['stop']

    stop_on_json = False
    if 'stop_on_json' in message_j.keys() and message_j['stop_on_json']==True:
        stop_on_json=True

    messages = message_j['messages']

    settings.temperature = temp
    settings.top_p = top_p
    formatted = hf_tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = exl_tokenizer.encode(formatted)
    print(f'input_ids {input_ids.shape}')
    generator.set_stop_conditions(stop_conditions)
    generator.begin_stream(input_ids, settings)
    return StreamingResponse(stream_data(query, max_new_tokens = max_tokens, stop_on_json=stop_on_json))

