#!bin/bash
curl http://localhost:5000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "/home/bruce/Downloads/models/Mistral-Small-3.1-24B-Base-2503", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Who won the world series in 2020?"} ] }'
