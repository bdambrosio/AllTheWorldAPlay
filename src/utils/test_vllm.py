import requests, json
headers = {"Content-Type": "application/json"}
url = 'http://localhost:5000/v1/completions'

content="Given a message, determine the most appropriate sensory mode for it.\nInput may be auditory, visual, or movement, or internal.\n\n<Message>\nyou hear: hi Joe\n</Message>\n\nRespond using this XML format:\n\n<Input>true/false </Input\n<Mode>auditory/visual/movement/internal</Mode>\n<Content>terse description of perceptual content in the given mode</Content>\n<Intensity>0-1</Intensity>\n\nBe sure to include any character name in the content.\nDo not include any introductory, discursive, or explanatory text.\nRespond only with the above XML.\nEnd your response with:\n</End>\n"
response =  requests.post(url, headers= headers,
            json={"model":"/home/bruce/Downloads/models/Qwen2.5-32B-Instruct", 
                  "prompt":content, "temperature":0,"top_p":0.5, "max_tokens":50, "stop":['End/']})


print(response.json()['choices'][0]['text'])

"""import openai
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="/home/bruce/Downloads/models/Qwen2.5-32B-Instruct",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)

completion = client.chat.completions.create(
  model="/home/bruce/Downloads/models/Qwen2.5-32B-Instruct",
  messages=[
    {"role": "user", "content":content},
    {"role": "assistant", "content":'<think>\n\n</think>'}
  ]
)

print(completion.choices[0].message)
"""
