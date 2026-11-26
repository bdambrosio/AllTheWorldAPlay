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

#server_name='together'   # Inkling 2 Small on Together; needs TOGETHER_API_KEY
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
# Claude Opus 5.5 plans and resolves what deeds and lines do (the 'haiku' server is the Anthropic
# path; needs CLAUDE_API_KEY or ANTHROPIC_API_KEY). On 225 recorded deeds (2026-11-20,
# logs/review-20261120/resolver) it was the cleanest of Opus, GPT-6.2 Sol and Sonnet 5.5.
# ATWAP_PLAN_LLM=sol puts Sol back.
# 2026-11-26: Opus costs too much (Bruce). The planner is GLM-5.3 on Fireworks, the full model, not the Flash that plays
# the parts: it takes every call Opus had (PLAN_TAGS: the plan, the stage manager, the rulings on set-ups and deeds).
# Untested in this role. Opus again: ATWAP_PLAN_LLM=haiku ATWAP_PLAN_MODEL=claude-opus-5-5.
plan_server_name = 'fireworks'
plan_model_name = 'accounts/fireworks/models/glm-5p3'
if os.environ.get('ATWAP_PLAN_LLM'):
    plan_server_name = os.environ['ATWAP_PLAN_LLM']
    plan_model_name = os.environ.get('ATWAP_PLAN_MODEL') or None
if plan_server_name in ('same', 'none', '') or (plan_server_name == server_name and (plan_model_name or model_name) == model_name):
    plan_server_name = None   # the same server with another model named is a planner (GLM-5.3 planning for GLM-5.3-Flash)
# A call with a server of its own (utils/llm_api.py own). None has one: the ruling on a line and its deed was on GPT-6.2
# Sol from 2026-11-23 and goes to the planner again (2026-11-26, Bruce: nothing ever calls an OpenAI model; utils/llm_api.py
# refuses those servers). ATWAP_SAY_LLM names another server for it (ATWAP_SAY_MODEL a model).
tag_servers = {}
if os.environ.get('ATWAP_SAY_LLM'):
    tag_servers = {} if os.environ['ATWAP_SAY_LLM'] in ('plan', 'same', 'none') else {'stage.resolve_say': (os.environ['ATWAP_SAY_LLM'], os.environ.get('ATWAP_SAY_MODEL') or None)}
# With a planner, it also keeps the world and plans each character's part (utils/llm_api.py
# PLAN_GROUPS: 'world', 'character'), so a local acting model writes only the lines.
# ATWAP_PLAN_GROUPS=none sends it only the play's plan; 'world' or 'character' one of the two.
