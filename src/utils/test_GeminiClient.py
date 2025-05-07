import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.llm_api as LLM_API
from utils.Messages import UserMessage
llm = LLM_API.LLM(server_name='Gemini')
from google import genai
from google.genai import types # type: ignore

client = None
api_key = None
model = 'gemini-2.5-flash-preview-04-17'
try:
    api_key = os.getenv("GOOGLE_KEY")
except Exception as e:
    print(f"Error getting Google API key: {e}")

if api_key is not None and api_key != '':
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error opening Google client: {e}")
    try:
        response = client.models.generate_content(model="gemini-2.5-flash-preview-04-17", contents="Explain how AI works in a few words")
        print(response.text)        
    except Exception as e:
        print(f"Error opening OpenAI client: {e}")

try:
    response = llm.ask({}, [UserMessage(content="Explain how AI works in a few words")])
    print(response)
except Exception as e:
    print(e)

