
import sys, os
import json
import requests
import subprocess
#add local dir to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Cache_TP,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

from exllamav2.attn import ExLlamaV2Attention
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.moe_mlp import ExLlamaV2MoEMLP
from exllamav2.parallel_decoder import ExLlamaV2ParallelDecoder

import argparse, os, math, time
import torch
import torch.nn.functional as F
from exllamav2.conversion.tokenize import get_tokens
from exllamav2.conversion.quantize import list_live_tensors
import gc

# from exllamav2.mlp import set_catch

import sys
import json

torch.cuda._lazy_init()
torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 150)

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_float32_matmul_precision("medium")

# (!!!) NOTE: These go on top of the engine arguments that can be found in `model_init.py` (!!!)
parser = argparse.ArgumentParser(description = "Test inference on ExLlamaV2 model")
parser.add_argument("-ed", "--eval_dataset", type = str, help = "Perplexity evaluation dataset (.parquet file)")
parser.add_argument("-er", "--eval_rows", type = int, default = None, help = "Number of rows to apply from dataset (default 128)")
parser.add_argument("-el", "--eval_length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-et", "--eval_token", action = "store_true", help = "Evaluate perplexity on token-by-token inference using cache")
parser.add_argument("-e8", "--eval_token_8bit", action = "store_true", help = "Evaluate perplexity on token-by-token inference using 8-bit (FP8) cache")
parser.add_argument("-eq4", "--eval_token_q4", action = "store_true", help = "Evaluate perplexity on token-by-token inference using Q4 cache")
parser.add_argument("-eq6", "--eval_token_q6", action = "store_true", help = "Evaluate perplexity on token-by-token inference using Q6 cache")
parser.add_argument("-eq8", "--eval_token_q8", action = "store_true", help = "Evaluate perplexity on token-by-token inference using Q8 cache")
parser.add_argument("-ecl", "--eval_context_lens", action = "store_true", help = "Evaluate perplexity at range of context lengths")
# parser.add_argument("-eb", "--eval_bos", action = "store_true", help = "Add BOS token to every row in perplexity test (required by Gemma and maybe other models.)")
parser.add_argument("-p", "--prompt", type = str, help = "Generate from prompt (basic sampling settings)")
parser.add_argument("-pnb", "--prompt_no_bos", action = "store_true", help = "Don't add BOS token to prompt")
parser.add_argument("-t", "--tokens", type = int, default = 128, help = "Max no. tokens")
parser.add_argument("-ps", "--prompt_speed", action = "store_true", help = "Test prompt processing (batch) speed over context length")
parser.add_argument("-s", "--speed", action = "store_true", help = "Test raw generation speed over context length")
parser.add_argument("-mix", "--mix_layers", type = str, help = "Load replacement layers from secondary model. Example: --mix_layers 1,6-7:/mnt/models/other_model")
parser.add_argument("-nwu", "--no_warmup", action = "store_true", help = "Skip warmup before testing model")
parser.add_argument("-sl", "--stream_layers", action = "store_true", help = "Load model layer by layer (perplexity evaluation only)")
parser.add_argument("-sp", "--standard_perplexity", choices = ["wiki2"], help = "Run standard (HF) perplexity test, stride 512 (experimental)")
parser.add_argument("-rr", "--rank_reduce", type = str, help = "Rank-reduction for MLP layers of model, in reverse order (for experimentation)")
parser.add_argument("-mol", "--max_output_len", type = int, help = "Set max output chunk size (incompatible with ppl tests)")

# Initialize model and tokenizer

model_init.add_args(parser)
model_name = "/home/bruce/Downloads/models/QwQ-32B-8.0bpw-h8-exl2"
model_name = "/home/bruce/Downloads/models/Qwen2.5-32B-Instruct"
model_name = "/home/bruce/Downloads/models/Llama-3.3-70B-Instruct_exl2_8.0bpw"
#model_name = "/home/bruce/Downloads/models/LatitudeGames_Wayfarer-Large-70B-Llama-3.3-4.25bpw-h6-exl2"
#model_name = "/home/bruce/Downloads/models/Qwen2.5-VL-32B-Instruct"
args = parser.parse_args(['-m',model_name, '--gpu_split', "34,46"])
#args = parser.parse_args(['-m',model_name, '--gpu_split', "0,46"])

model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(
    args,
    skip_load = args.stream_layers,
    benchmark = True,
    max_output_len = args.max_output_len,

    progress = True
)

generator = None
torch.inference_mode()
hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
exl_tokenizer = tokenizer
cache = ExLlamaV2Cache(model) if not model.tp_context else ExLlamaV2Cache_TP(model)

async def generate_response(prompt: str, settings: ExLlamaV2Sampler.Settings, num_tokens: int):
    global cache
    with torch.inference_mode():


        print(f" -- Warmup...")

        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        generator.warmup()

        print(f" -- Generating...")
        print()

 
        time_begin = time.time()

        output = generator.generate_simple(prompt, settings, num_tokens)

        torch.cuda.synchronize()
        time_prompt = time.time()

        time_end = time.time()

        total_gen = time_end - time_begin
        print(f" -- Response generated in {total_gen:.2f} seconds, {args.tokens} tokens, {args.tokens / total_gen:.2f} tokens/second (includes prompt eval.)")
        return output


GENERATION_PROMPT=None
while GENERATION_PROMPT == None:
    prime = input("Does model need add_generation_prompt=True? {t/f}:")
    if prime.lower().startswith('t'):
        GENERATION_PROMPT=True
    elif prime.lower().startswith('f'):
        GENERATION_PROMPT=False
        
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
generator.warmup()

app = FastAPI()
print(f"starting server")
@app.post("/template")
async def template(request: Request):
    print(f'context size request, returning {16384}')
    return {"context_size": 16384}
        
@app.post("/v1/chat/completions")
async def get_stream(request: Request):
        global generator, settings, hf_tokenizer, exl_tokenizer, cache
        query = await request.json()
        print(f'request: {json.dumps(query)}')
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

        stop_conditions = []
        if 'stop' in message_j.keys():
            print(f'\n received stop {message_j["stop"]}')
            stop_conditions = message_j['stop']


        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temp
        settings.top_k = 50
        settings.top_p = top_p
        settings.token_repetition_penalty = 1.02
        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
        settings.temperature = temp
        settings.min_p = 0.0
        start_time = time.time()

        messages = message_j['messages']
        formatted = hf_tokenizer.apply_chat_template(messages, setings=settings, tokenize=False, add_generation_prompt=GENERATION_PROMPT)


        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 1.0
        settings.top_k = 0
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.02
        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

        if type(stop_conditions) == list and len(stop_conditions) > 0:
            stop_str = stop_conditions[0]
            stop_seq = tokenizer.encode(stop_conditions[0])
        else:
            stop_str=None
            stop_seq = None


        output = generator.generate_simple(formatted, settings, max_tokens, completion_only=True, stop_seq=stop_seq)
        #output = generator.generate_simple(formatted, settings, max_tokens)

        torch.cuda.synchronize()
        #output = await generate_response(formatted, settings, max_tokens)
        if stop_str and output.endswith(stop_str):
            output = output[:-len(stop_str)]
        print(output)
        print()
        print(f'prompt len {exl_tokenizer.encode(formatted).shape} response {exl_tokenizer.encode(output).shape} time {time.time() - start_time:.2f}s')
        #return Response('{"choices": [{"message": {"content": '+output+'}}]}')
        return Response(output)

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 1.0
settings.top_k = 0
settings.top_p = 0.8
settings.token_repetition_penalty = 1.02
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

cache = ExLlamaV2Cache(model) if not model.tp_context else ExLlamaV2Cache_TP(model)
load=time.time()
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
create=time.time()-load
warmup_start=time.time()
generator.warmup()
warmup_end=time.time()-warmup_start
inference_start=time.time()
print(generator.generate_simple('hi there', settings, 10))
inference_end=time.time()-inference_start
print(f"load: {load:.4f}s, create: {create:.4f}s, warmup: {warmup_end:.4f}s, inference: {inference_end:.4f}s")
inference_start=time.time()
stop_seq=exl_tokenizer.encode('<end/>')
print('encoded')
print(generator.generate_simple('bye there', settings, 10, completion_only=True, stop_seq=stop_seq))
inference_end=time.time()-inference_start
print(f" inference: {inference_end:.4f}s")
