#!/bin/bash
#vllm serve /home/bruce/Downloads/models/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 10288 --enforce-eager --port 5000 --gpu_memory_utilization 0.85

#vllm serve /home/bruce/Downloads/models/Qwen2.5-32B-Instruct --tensor-parallel-size 2 --max-model-len 10288 --enforce-eager --port 5000 --gpu_memory_utilization 0.85

vllm serve /home/bruce/Downloads/models/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q8_0.gguf --tensor-parallel-size 2 --max-model-len 10288 --enforce-eager --port 5000 --gpu_memory_utilization 0.85
