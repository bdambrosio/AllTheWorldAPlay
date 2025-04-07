import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.llm_api as LLM_API
from utils.Messages import UserMessage, SystemMessage, AssistantMessage
llm = LLM_API.LLM(server_name='deepseeklocal')

try:
    response = llm.ask({}, [UserMessage(content="Hello, world! response with goodbye. end your response with <end/>")], 
                       max_tokens=5, stops=['<end/>'])
    print(response)
    
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

    prompt = [SystemMessage(content='Your task is to determine the sensory mode of the following message.'),
            UserMessage(content=content),
            AssistantMessage(content='The sensory mode is ')]

    for i in range(5):
        start = time.time()
        response = llm.ask({}, prompt, temp=0, max_tokens=10, stops=['<end/>'])
        elapsed = time.time()-start
        print(f'llm: {elapsed:.2f}')
        print(response)
except Exception as e:
    print(e)
