import time
import requests, json
import llm_api
from Messages import UserMessage, SystemMessage, AssistantMessage

llm = llm_api.LLM('deepseeklocal')
#llm = llm_api.LLM('local')
content="""Determine its sensory mode of the following message,
a terse description of the perceptual content,
and the emotionalintensity of the perceptual content.

sense mode may be:
auditory
visual
movement
internal
unclassified

<message>
You see Joe
</message>

Respond using this format:

mode
<end/>

Respond only with the mode. Do not include any introductory, discursive, or explanatory text.
"""

prompt = [UserMessage(content=content)]

for i in range(5):
    start = time.time()
    response = llm.ask({}, prompt, temp=0, max_tokens=10, stops=['<end/>'])
    elapsed = time.time()-start
    print(f'llm: {elapsed:.2f}')
    print(response)
