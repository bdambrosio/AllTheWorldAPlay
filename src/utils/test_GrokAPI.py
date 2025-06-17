import os
import time
from openai import OpenAI
from dotenv import load_dotenv

api_key = os.getenv('XAI_API_KEY')
client = OpenAI(
  api_key=api_key,
  base_url="https://api.x.ai/v1",
)

start = time.time()
for i in range(10):
  completion =  client.chat.completions.create(
  model="grok-3-mini",
  messages=[
    {"role": "system", "content": "You are a PhD-level mathematician."},
    {"role": "user", "content": "What is 2 + 2?\n no explanation, just the answer"},
  ],
  reasoning_effort="low",
  max_tokens=40,
)
end = time.time()
print(completion.choices[0].message)
print(f"Time taken: {end - start} seconds")
