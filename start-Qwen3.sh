#!/bin/bash
# Start the Python backend
cd /app/src/plays/
cp config_Qwen3-14B.py config.py # set config to use openai
vllm serve Qwen/Qwen3-14B --max-model-len 16384 --enforce-eager --port 5000  --gpu-memory-utilization 0.75 &
cd /app/src/sim/
uvicorn main:app --host 0.0.0.0 --port 8000  > main.log 2>&1 &      
cd /app/src/utils/    # main engine
fastapi run lcmLora-serve.py --host 0.0.0.0 --port 5008  > lcmLora-serve.log 2>&1 &    # imageserver
#fastapi run blank_image_serve.py --host 0.0.0.0 --port 5008 &    # image thumbs
cd /app/src/sim/webworld
npm start > webworld.log 2>&1 &                                                    # UI
wait -n            
