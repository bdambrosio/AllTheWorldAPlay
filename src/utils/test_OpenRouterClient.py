import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.llm_api as LLM_API
from utils.Messages import UserMessage
llm = LLM_API.LLM(server_name='openrouter')

try:
    response = llm.ask({}, [UserMessage(content="Hello, world!")])
    print(response)
except Exception as e:
    print(e)

