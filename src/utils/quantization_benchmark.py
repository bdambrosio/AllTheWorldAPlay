import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)
import bitsandbytes as bnb

def load_exllama_model(exl2_model_path):
    config = ExLlamaV2Config()
    config.model_dir = exl2_model_path
    config.prepare()
    
    model = ExLlamaV2(config)
    model.load()
    tokenizer = ExLlamaV2Tokenizer(config)
    
    return model, tokenizer

def load_8bit_model(hf_model_path):
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    
    return model, tokenizer

def benchmark_generation(model, tokenizer, prompt, is_exllama=True):

    
    if is_exllama:
        input_ids = tokenizer.encode(prompt)
        # ExLlamaV2 specific generation
        max_tokens = 100
        temperatures = [0.0, 0.7, 1.0]  # Test different sampling temperatures
        results = []
        
        for temp in temperatures:
            start_time = time.time()
            settings = {
                "temperature": temp,
                "top_p": 0.9,
                "min_p": 0.1,
                "top_k": 40,
            }
            
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                **settings
            )
            
            end_time = time.time()
            tokens_generated = len(output) - len(input_ids)
            time_taken = end_time - start_time
            tokens_per_second = tokens_generated / time_taken
            
            results.append({
                "temperature": temp,
                "tokens_per_second": tokens_per_second,
                "total_time": time_taken,
                "tokens_generated": tokens_generated
            })
    else:
        # 8-bit model generation
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids]).to(model.device)
        max_tokens = 100
        temperatures = [0.0, 0.7, 1.0]
        results = []
        
        for temp in temperatures:
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=0.9,
                    top_k=40,
                    do_sample=(temp > 0)
                )
            
            end_time = time.time()
            tokens_generated = output.shape[1] - input_ids.shape[1]
            time_taken = end_time - start_time
            tokens_per_second = tokens_generated / time_taken
            
            results.append({
                "temperature": temp,
                "tokens_per_second": tokens_per_second,
                "total_time": time_taken,
                "tokens_generated": tokens_generated
            })
    
    return results

def run_benchmark(exl2_model_path, hf_model_path, prompt):
    print("Loading ExLlamaV2 model...")
    exllama_model, exllama_tokenizer = load_exllama_model(exl2_model_path)
    
    print("\nRunning ExLlamaV2 benchmark...")
    exllama_results = benchmark_generation(exllama_model, exllama_tokenizer, prompt, is_exllama=True)

    del exllama_model
    del exllama_tokenizer
    torch.cuda.empty_cache()
    
    print("Loading 8-bit model...")
    model_8bit, tokenizer_8bit = load_8bit_model(hf_model_path)
    
    print("\nRunning 8-bit model benchmark...")
    int8_results = benchmark_generation(model_8bit, tokenizer_8bit, prompt, is_exllama=False)
    
    print("\nResults Summary:")
    print("\nExLlamaV2 Results:")
    for result in exllama_results:
        print(f"Temperature {result['temperature']}:")
        print(f"  Tokens per second: {result['tokens_per_second']:.2f}")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Tokens generated: {result['tokens_generated']}")
    
    print("\n8-bit Model Results:")
    for result in int8_results:
        print(f"Temperature {result['temperature']}:")
        print(f"  Tokens per second: {result['tokens_per_second']:.2f}")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Tokens generated: {result['tokens_generated']}")

if __name__ == "__main__":
    exl2_model_path = "/home/bruce/Downloads/models/DeepSeek-R1-Distill-Qwen-32B-exl2"
    hf_model_path = "/home/bruce/Downloads/models/DeepSeek-R1-Distill-Qwen-32B"
    prompt = "Write a short story about a robot learning to paint."
    
    run_benchmark(exl2_model_path, hf_model_path, prompt)
