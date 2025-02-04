#!/bin/bash
vllm serve /home/bruce/Downloads/models/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 10288 --enforce-eager --port 5000 --gpu_memory_utilization 0.85


