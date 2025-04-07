from vllm import LLM, SamplingParams
import sys, os
import json
import requests
import subprocess
import torch
#add local dir to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# Configure VLLM engine arguments
engine_args = AsyncEngineArgs(
    #model="/home/bruce/Downloads/models/DeepSeek-R1-Distill-Qwen-32B",
    model="/home/bruce/Downloads/models/Mistral-Small-3.1-24B-Base-2503",
    tensor_parallel_size=torch.cuda.device_count(),  # Use both GPUs
    dtype="bfloat16",
    #max_seq_len=16384,
    gpu_memory_utilization=0.7,
    trust_remote_code=False,
    enforce_eager=False,
    max_num_batched_tokens=2048,
    quantization=None,
)

# Let VLLM handle its own distributed setup
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Your serving logic here
print("Model is ready to serve")


app = FastAPI()
context_size = 16384
@app.post("/template")
async def template(request: Request):
    print(f'context size request, returning {context_size}')
    return {"context_size": context_size}
    
async def generate_stream(prompt: str, sampling_params, request_id: str):
    """Generate streaming response"""
    async for response in engine.generate(prompt, sampling_params, request_id):
        output = response.outputs[0]

        print(f'output: {output.text}')
        yield output.text

@app.post("/v1/chat/completions")
async def get_stream(request: Request):
    query = await request.json()
    print(f'request: {query}')
    message_j = query
    if 'template_query' in message_j.keys():
        return Response(template)

    messages = message_j['messages']
    prompt = str(messages)
    print(prompt)

    temp = 0.6
    if 'temp' in message_j.keys():
        temp = message_j['temp']

    top_p = 0.95
    if 'top_p' in message_j.keys():
        top_p = message_j['top_p']

    max_tokens = 100
    if 'max_tokens' in message_j.keys():
        max_tokens = message_j['max_tokens']

    stop_conditions = ['###', '<|endoftext|>', "Reference(s):"]
    if 'stop' in message_j.keys():
        print(f'\n received stop {message_j["stop"]}')
        stop_conditions = message_j['stop']

    sampling_params = SamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, stop=stop_conditions)
    #outputs = await generate_response(engine, prompt, sampling_params, 'abc')

    #print(f"Prompt: {prompt!r}, Generated text: {outputs!r}")
    return StreamingResponse(generate_stream(prompt, sampling_params, 'abc'))
    
