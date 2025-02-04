import requests, json
import llm_api
from Messages import UserMessage, SystemMessage, AssistantMessage

llm = llm_api.LLM('local')
content="""Given a message, determine the most appropriate sensory mode for it.
Input may be auditory, visual, or movement, or internal.

<message>
you hear: hi Joe
</message>

Respond only with the answer, do NOT report your thinking.
Respond using this XML format:

<input>true/false </input>
<mode>auditory/visual/movement/internal</mode>
<content>terse description of perceptual content in the given mode</content>
<intensity>0-1</intensity>
Be sure to include any character name in the content.
Do not include any introductory, discursive, or explanatory text.
Do not include any markdown or other formatting information.
Respond only with the above XML.
End your response with:
<end/>
"""

prompt = [UserMessage(content=content)]

response = llm.ask({}, prompt, max_tokens=100, stops=['<end/>'])

print(response)
