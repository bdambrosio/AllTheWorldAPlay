import time
import requests, json
import llm_api
from Messages import UserMessage, SystemMessage, AssistantMessage

llm = llm_api.LLM('deepseeklocal')
llm = llm_api.LLM('local')
content="""
<message>
You see Joe
</message>

Your answer must be one of the following:
auditory
visual
movement
internal
unclassified

"""

prompt = [UserMessage(content=f"""Your task is to determine the sensory mode of the following message:
    {content} 
Respond with only the single word for the sensory mode, followed by </end>. Do NOT include any thinking or reasoning in your response.</think>"""),
          ]

for i in range(5):
    start = time.time()
    response = llm.ask({}, prompt, temp=0, max_tokens=200, stops=['</end>'])
    elapsed = time.time()-start
    print(f'llm: {elapsed:.2f}')
    print(response)
