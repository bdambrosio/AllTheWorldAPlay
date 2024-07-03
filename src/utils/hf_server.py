import sys, os
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Any, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, GenerationConfig
# Initialize model and cache


# Custom stopping criteria
class StringStoppingCriteria(StoppingCriteria):
    def __init__(self, stopping_strings, tokenizer):
        self.stopping_strings = [tokenizer.encode(s, add_special_tokens=False) for s in stopping_strings]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_tokens = input_ids[0].tolist()
        for stop_tokens in self.stopping_strings:
            if stop_tokens == generated_tokens[-len(stop_tokens):]:
                return True
        return False

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

#if 'llama3-70B' in model_name:
#    print(f"Loading model: {model_name}\n context {context_size}")
#    model.load([45, 45, 48])
#    print('model load done..')

#else:
#    print(f"Loading model: {model_name}\n context {context_size}")
#    model.load([40, 42, 42])
#    
#    print('model load done..')


GENERATION_PROMPT=None
while GENERATION_PROMPT == None:
    prime = input("Does model need add_generation_prompt=True? {t/f}:")
    if prime.lower().startswith('t'):
        GENERATION_PROMPT=True
    elif prime.lower().startswith('f'):
        GENERATION_PROMPT=False
        
tokenizer = AutoTokenizer.from_pretrained(models_dir+model_name)
model = AutoModelForCausalLM.from_pretrained(models_dir+model_name,
                                             device_map="auto",
                                             attn_implementation = 'eager', # for gemma
                                             torch_dtype=torch.bfloat16
                                             )

app = FastAPI()
print(f"starting server")
@app.post("/template")
async def template(request: Request):
    return {"context_size": context_size}
    
@app.post("/v1/chat/completions")
async def get_response(request: Request):
    global generator, settings, hf_tokenizer, exl_tokenizer
    query = await request.json()
    print(f'request: {query}')
    message_j = query

    gconfig = GenerationConfig()

    if 'template_query' in message_j.keys():
        template()
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

    # Define stopping criteria
    stopping_criteria = StoppingCriteriaList([StringStoppingCriteria(stop_conditions, tokenizer)])
    # Define the stopping criteria list
    #gconfig = GenerationConfig(stop_strings = stop_conditions, temperature=temp, top_p=top_p, max_new_tokens=max_tokens, do_sample=True)

    messages = message_j['messages']

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=GENERATION_PROMPT)
    #print(formatted)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs['attention_mask']
    
    response = model.generate(input_ids,
                              do_sample=True,
                              max_new_tokens=max_tokens,
                              top_p=top_p,
                              stopping_criteria=stopping_criteria,
                              temperature=temp,
                              num_return_sequences=1,
                              attention_mask=attention_mask)
    text_response = tokenizer.decode(response[0][len(input_ids[0]):], skip_special_tokens=True)
    
    print(f'\n****************\n{text_response}\n*******************\n')
    return text_response

    #streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # Generate tokens with streaming
    #generation_kwargs = dict(input_ids=inputs["input_ids"],
    #                         attention_mask=inputs["attention_mask"],
    #                         streamer=streamer,
    #                         max_new_tokens=max_tokens,)
    #                         #stop=stop_conditions,
    #                         #temperature=temp,
    #                         #top_p=top_p)
    #print('calling generate')
    #generation_task = model.generate(**generation_kwargs)

    # Stream the output tokens
    #async def generate():
    #    async for output in generation_task:
    #        yield output
    # 
    #return StreamingResponse(generate(), media_type="text/plain")
