model_name=None
#server_name='deepseek'
#server_name='vllm'       # the local model (Qwen under vLLM on :5000)
server_name='fireworks'   # GLM-5.3-Flash on Fireworks; needs FIREWORKS_PROJECT_KEY in ~/.config/secrets.env
#server_name='openai'; model_name='Qwen3-14B'
#server_name = 'local'
#server_name='Claude'
#server_name='llama.cpp'
#server_name='openai'; model_name='gpt-4.1-mini'
#server_name='openai'; model_name='gpt-4.1'
#server_name='openai'; model_name='gpt-4.1-nano'
#server_name='openai'; model_name='gpt-4o'
#server_name='gemini'
#server_name='gemini-pro'
#server_name='Grok' 
#server_name='Cohere'
#server_name='openrouter'; model_name='google/gemini-2.5-flash' 
#server_name='openrouter'; model_name='meta-llama/llama-4-scout' #google/gemini-2.0-flash-preview'
#server_name='openrouter'; model_name='qwen/qwen3-32b'
#server_name='openrouter'; model_name='moonshotai/kimi-k2'

#server_name='haiku'      # Claude Haiku 5.5; needs CLAUDE_API_KEY. ATWAP_EFFORT=off|low|medium|high (utils/llm_api.py)
# One run on another model without editing this file: ATWAP_LLM=vllm ./start-local.sh
# (the local model). ATWAP_MODEL names a model for the chosen server.
import os
if os.environ.get('ATWAP_LLM'):
    server_name = os.environ['ATWAP_LLM']
    model_name = os.environ.get('ATWAP_MODEL') or None

# The plan of a play (its central question, each character's acts, the acts merged, each scene's
# set-up) is written by this server whatever plays the parts: a small local model, given the
# same characters and drives, planned berries, boar, wolf and left out the play's mystery.
# The calls are named in utils/llm_api.py PLAN_TAGS. None, or ATWAP_PLAN_LLM=same: no planner.
plan_server_name = 'fireworks'
plan_model_name = None
if os.environ.get('ATWAP_PLAN_LLM'):
    plan_server_name = os.environ['ATWAP_PLAN_LLM']
    plan_model_name = os.environ.get('ATWAP_PLAN_MODEL') or None
if plan_server_name in ('same', 'none', '', server_name):
    plan_server_name = None
# With a planner, it also keeps the world and plans each character's part (utils/llm_api.py
# PLAN_GROUPS: 'world', 'character'), so a local acting model writes only the lines.
# ATWAP_PLAN_GROUPS=none sends it only the play's plan; 'world' or 'character' one of the two.
