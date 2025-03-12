import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.llm_api as LLM_API
from utils.Messages import UserMessage
llm = LLM_API.LLM(server_name='local')

try:
    response = llm.ask({}, [UserMessage(content="Hello, world! response with goodbye. end your response with <end/>")], max_tokens=20, stops=['<end/>'])
    print(response)
except Exception as e:
    print(e)
