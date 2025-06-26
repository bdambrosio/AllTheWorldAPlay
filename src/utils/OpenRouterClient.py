import requests, time, copy
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import json, os
from utils.Messages import SystemMessage
from utils.Messages import AssistantMessage
from openai import OpenAI
from utils.LLMRequestOptions import LLMRequestOptions

api_key = None
try:
    api_key = os.getenv("OPENROUTER_API_KEY")
except Exception as e:
    print(f"Error getting OpenRouter API key: {e}")
#MODEL = 'google/gemini-2.0-flash-001'
#MODEL = 'meta-llama/llama-4-maverick'
MODEL = 'google/gemma-3-27b-it'


class OpenRouterClient():
    DefaultEndpoint = 'https://openrouter.ai'
    UserAgent = 'Owl'
    def __init__(self, api_key=None):
        self.api_key = api_key
        
    def executeRequest(self, prompt, options: LLMRequestOptions):
        startTime = time.time()
        if options.stops: #options.stop only allows a single stop string for now.
            stops=options.stops
        else:
            stops=[]
        if options.stop_on_json:
            stops.append('}')
        if options.model is not None:
            model = options.model
        else:
            model = 'meta-llama/llama-4-maverick'
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://tuuyi.com",  # Optional site URL for rankings
                        "X-Title": "Tuuyi",  # Optional site title
                    },
                    data=json.dumps({
                        "model": MODEL,
                        "messages": prompt,
                        "max_tokens": options.max_tokens,
                        "temperature": options.temperature,
                        "top_p": options.top_p,
                        "stop": stops,
                        "stream": False,
                    }),
                    timeout=30.0,
                )

                response.raise_for_status()
                item = response.content
                item = json.loads(item.decode("utf-8"))
                item = item["choices"][0]["message"]["content"]
                return item

            except Exception as e:
                print(f"OpenRouter request error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {"status": "error", "message": "retries exceeded"}
                # Exponential backoff: 1s, 2s, 4s
                time.sleep(2 ** attempt)

        # If loop exits without return, treat as error
        return {"status": "error", "message": "Request failed after all retries"}
            

