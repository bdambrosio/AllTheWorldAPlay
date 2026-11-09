"""LLM access for the engine: one LLM class that dispatches to the configured backend.

server_name comes from src/plays/config.py. 'vllm' (an OpenAI-compatible server on
LLM_BASE_URL, default http://localhost:5000/v1) is the path in use; the hosted-provider
clients are imported only when their server_name is selected.
"""
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import deque
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.LLMRequestOptions import LLMRequestOptions
from utils.Messages import AssistantMessage

MODEL = ''  # set by simulation from config.py
logger = logging.getLogger('simulation_core')

LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:5000/v1')
# Fireworks-hosted GLM-5.3-Flash, used as an external judge (tools/judges.py); route and
# settings follow Cognitive_workbench measure/models/fw_glm53flash.yaml. Key: FIREWORKS_API_KEY.
FIREWORKS_URL = 'https://api.fireworks.ai/inference/v1'
FIREWORKS_MODEL = 'accounts/fireworks/models/glm-5p3-flash'
IMAGE_URL = os.environ.get('IMAGE_URL', 'http://127.0.0.1:5008/generate_image')
IMAGE_PATH = Path.home() / '.local/share/AllTheWorld/images'
IMAGE_PATH.mkdir(parents=True, exist_ok=True)

response_prime_needed = False
elapsed_times = {}
iteration_count = 0
pattern = r'\{\$[^}]*\}'


def set_model(model):
    global MODEL
    MODEL = model
    OpenRouterClient = sys.modules.get('utils.OpenRouterClient')
    if OpenRouterClient is not None and hasattr(OpenRouterClient, 'MODEL'):
        OpenRouterClient.MODEL = model


def generate_image(llm=None, description='', size='512x512', filepath='test.png'):
    """Fetch an image from the local image server (utils/klein_serve.py); grey placeholder on failure."""
    image_content = None
    try:
        response = requests.get(IMAGE_URL, params={"prompt": "photorealistic style: " + description, "size": size},
                                timeout=60)
        if response.status_code == 200:
            image_content = response.content
    except requests.RequestException as e:
        print(f'image server unavailable ({e}); using placeholder image')
    if image_content is None:
        # fall back to a plain grey placeholder so the sim never dies for lack of eye candy
        try:
            w, h = (int(v) for v in size.lower().split('x'))
        except Exception:
            w, h = 256, 256
        buf = BytesIO()
        Image.new('RGB', (w, h), color='#808080').save(buf, format='PNG')
        image_content = buf.getvalue()
    with open(IMAGE_PATH / filepath, "wb") as file:
        file.write(image_content)
    return IMAGE_PATH / filepath


RECENT_CALLS = 30

# A local server that crashes restarts itself (the vLLM container: ~2-3 minutes to reload the
# model). Calls made meanwhile wait for it instead of failing, which used to ruin the play.
LLM_RESTART_WAIT_SECS = float(os.environ.get('LLM_RESTART_WAIT_SECS', '420'))


def _post_with_restart_wait(url, body):
    """POST a completion; on connection failure or a 5xx, wait for the server and retry."""
    deadline = time.time() + LLM_RESTART_WAIT_SECS
    delay, announced = 2.0, False
    while True:
        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=body, timeout=600)
            if response.status_code < 500 or time.time() >= deadline:
                if announced:
                    print('LLM server is back')
                    logger.warning('LLM server is back')
                return response
            problem = f'{response.status_code}: {response.text[:120]}'
        except (requests.ConnectionError, requests.Timeout) as e:
            if time.time() >= deadline:
                raise
            problem = type(e).__name__
        if not announced:
            announced = True
            print(f'LLM server unavailable ({problem}); waiting up to {LLM_RESTART_WAIT_SECS:.0f}s for it to restart')
            logger.warning(f'LLM server unavailable ({problem}); waiting for restart')
        time.sleep(delay)
        delay = min(delay * 1.5, 15)

# Called after every LLM.ask with tag, owner, tokens, finish_reason, secs, ok, error
# (sim.trace registers one; utils does not import sim).
_call_listeners = []

def add_call_listener(fn):
    if fn not in _call_listeners:
        _call_listeners.append(fn)


@lru_cache(maxsize=None)
def _served_model_id(base_url):
    """Model id served by an OpenAI-compatible server. Cached: every Character and NPC builds an LLM."""
    try:
        response = requests.get(f'{base_url}/models', timeout=5)
        response.raise_for_status()
        data = response.json().get('data') or []
        if not data:
            raise ValueError("no models listed")
        return data[0]['id']
    except (requests.RequestException, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to get model information from {base_url}: {e}") from e


# Hosted-provider clients, created on first use (importing them all eagerly pulled in
# transformers, google-genai, cohere and anthropic, and built clients nobody used).
@lru_cache(maxsize=None)
def _provider(name):
    if name == 'Claude':
        import utils.ClaudeClient as m; return m
    if name == 'Cohere':
        import utils.CohereClient as m; return m
    if name == 'Grok':
        import utils.GrokClient as m; return m
    if name == 'llama.cpp':
        import utils.llcppClient as m; return m
    if name == 'deepseek':
        from utils.DeepSeekClient import DeepSeekClient; return DeepSeekClient()
    if name == 'gemini':
        from utils.GeminiClient import GeminiClient; return GeminiClient()
    if name == 'openrouter':
        import utils.OpenRouterClient as m
        if MODEL and hasattr(m, 'MODEL'):
            m.MODEL = MODEL
        return m.OpenRouterClient(api_key=os.getenv("OPENROUTER_API_KEY"))
    raise ValueError(f'unknown provider {name}')


class LLM():

    def __init__(self, server_name='local', model_name=None):
        self.server_name = server_name
        print(f'will use {self.server_name} as llm')
        self.context_size = 16384  # conservative local default
        # last calls through this instance (each Character owns one), for the Explore panel
        self.recent_calls = deque(maxlen=RECENT_CALLS)
        self.owner = None        # name of the character (or 'Context') making the calls
        self._last_usage = None  # (usage dict, finish_reason) of the last OpenAI-compatible reply
        self.model = model_name if model_name is not None else MODEL
        if self.server_name == 'openai':
            from utils.OpenAIClient import OpenAIClient
            self.openai_client = OpenAIClient(model_name=self.model)
        if self.server_name == 'vllm':
            self.model = _served_model_id(LLM_BASE_URL)
        if self.server_name == 'fireworks':
            self.model = model_name or FIREWORKS_MODEL
            self.reasoning_effort = 'low'   # 'medium' for whole-play rubrics

    def _provider_name(self, options):
        s = self.server_name
        if options.model is not None and 'deepseek' in options.model:
            return 'deepseek'
        for key in ('deepseek', 'llama.cpp', 'Claude', 'Cohere', 'Grok'):
            if key in s:
                return key
        for key in ('openrouter', 'gemini'):
            if key in s.lower():
                return key
        return None

    def run_request(self, bindings, prompt, options, log=False, trace=True):
        # substitute {{$var-name}} in the prompt
        if bindings is None or len(bindings) == 0:
            substituted_prompt = [{'role': message.role, 'content': message.content} for message in prompt]
        else:
            substituted_prompt = []
            for message in prompt:
                if type(message.content) is not str:  # substitutions only in strings
                    substituted_prompt.append({'role': message.role, 'content': message.content})
                    continue
                new_content = message.content
                for match in re.findall(pattern, message.content):
                    var = match[2:-1]
                    if var in bindings.keys():
                        new_content = new_content.replace('{{$' + var + '}}', str(bindings[var]))
                    else:
                        print(f' var not in bindings {var} {bindings}')
                        raise ValueError(f'unbound prompt variable {var}')
                substituted_prompt.append({'role': message.role, 'content': new_content})

        if trace:
            print(f'\n{json.dumps(substituted_prompt)}\n')
        if log:
            logger.info(f'Prompt: {substituted_prompt}\n')

        if 'openai' in self.server_name:
            return self.openai_client.executeRequest(prompt=substituted_prompt, options=options)
        if self.server_name == 'fireworks':
            return self._fireworks_request(substituted_prompt, options)
        provider = self._provider_name(options)
        if provider is not None:
            return _provider(provider).executeRequest(prompt=substituted_prompt, options=options)

        # OpenAI-compatible local server ('vllm', or 'local')
        if options.stops is None:
            options.stops = []
        url = f'{LLM_BASE_URL}/chat/completions'
        if 'vllm' in self.server_name:
            # Qwen chat templates reject requests with no user turn ("No user query found"),
            # and some engine prompts are a lone SystemMessage: send the last one as the user turn.
            messages = substituted_prompt
            if not any(m['role'] == 'user' for m in messages):
                messages = [dict(m) for m in messages]
                for m in reversed(messages):
                    if m['role'] == 'system':
                        m['role'] = 'user'
                        break
            # thinking disabled: callers use small max_tokens budgets that reasoning would consume
            body = {"model": self.model, "messages": messages,
                    "temperature": options.temperature, "top_p": options.top_p,
                    "max_tokens": options.max_tokens, "stop": options.stops or None,
                    "chat_template_kwargs": {"enable_thinking": False}}
        else:
            body = {"messages": substituted_prompt, "temperature": options.temperature,
                    "top_p": options.top_p, "max_tokens": options.max_tokens, "stop": options.stops}
        response = _post_with_restart_wait(url, body)
        if response.status_code != 200:
            raise Exception(f'{response.status_code} from LLM server: {response.text[:500]}')
        try:
            reply = response.json()
            text = reply['choices'][0]['message']['content'] or ''
            self._last_usage = (reply.get('usage') or {}, reply['choices'][0].get('finish_reason'))
        except (ValueError, KeyError, IndexError):
            return response.content.decode('utf-8')
        if (index := text.find('</think>')) > -1:
            text = text[index + 8:].strip()
        return text

    def _fireworks_request(self, messages, options, retries=6):
        """Fireworks chat completion with backoff on 429/5xx (the account's rate limit, not the
        model's, is what throttles; see the workbench yaml)."""
        key = os.environ.get('FIREWORKS_API_KEY')
        if not key:
            raise RuntimeError('FIREWORKS_API_KEY is not set (source ~/.config/secrets.env)')
        body = {"model": self.model, "messages": messages, "temperature": options.temperature,
                "top_p": options.top_p, "max_tokens": options.max_tokens,
                "stop": options.stops or None, "reasoning_effort": getattr(self, 'reasoning_effort', 'low')}
        delay = 2.0
        for attempt in range(retries):
            response = requests.post(f'{FIREWORKS_URL}/chat/completions', json=body, timeout=300,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            if response.status_code == 200:
                reply = response.json()
                choice = reply['choices'][0]
                self._last_usage = (reply.get('usage') or {}, choice.get('finish_reason'))
                text = choice['message'].get('content') or ''
                if (index := text.find('</think>')) > -1:
                    text = text[index + 8:].strip()
                return text
            if response.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise Exception(f'{response.status_code} from Fireworks: {response.text[:300]}')

    def _record_call(self, tag, start, response, error=None):
        calls = getattr(self, 'recent_calls', None)
        if calls is None:
            calls = self.recent_calls = deque(maxlen=RECENT_CALLS)
        calls.append({'tag': tag or 'default', 'at': time.strftime('%H:%M:%S', time.localtime(start)),
                      'secs': round(time.time() - start, 2), 'ok': error is None,
                      'response': (response or '')[:160] if error is None else '',
                      'error': (error or '')[:300]})
        if _call_listeners:
            usage, finish = getattr(self, '_last_usage', None) or ({}, None)
            self._last_usage = None
            details = usage.get('prompt_tokens_details') or {}
            fields = dict(tag=tag or 'default', owner=getattr(self, 'owner', None),
                          secs=round(time.time() - start, 3), ok=error is None, error=(error or '')[:300],
                          prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                          cached_tokens=details.get('cached_tokens'), finish_reason=finish,
                          response_chars=len(response or ''))
            for listener in _call_listeners:
                try:
                    listener(**fields)
                except Exception:
                    pass

    def ask(self, input, prompt_msgs, template=None, tag='', temp=None, max_tokens=None, top_p=None, stops=None,
            stop_on_json=False, model=None, log=True, trace=False):
        """Returns the response text, or None on any failure (callers must handle None)."""
        global iteration_count
        if max_tokens is None: max_tokens = 400
        if temp is None: temp = 0.7
        if top_p is None: top_p = 1.0

        start = time.time()
        options = LLMRequestOptions(temperature=temp, top_p=top_p, max_tokens=max_tokens,
                                    stops=stops, stop_on_json=stop_on_json, model=model)
        try:
            if response_prime_needed and type(prompt_msgs[-1]) != AssistantMessage:
                prompt_msgs = prompt_msgs + [AssistantMessage(content='')]
            print(f'{tag}...', end='')
            response = self.run_request(input, prompt_msgs, options, log=log, trace=trace)
            elapsed = time.time() - start
            print(f'{elapsed:.2f}')
            tag = tag or 'default'
            elapsed_times[tag] = elapsed_times.get(tag, 0) + elapsed
            iteration_count += 1
            if iteration_count % 100 == 0:
                print(f'iteration {iteration_count}')
                for k, v in elapsed_times.items():
                    print(f'{k}: {v:.2f}')
                print(f'total: {sum(elapsed_times.values()):.2f}\n')
            if elapsed > 4.0:
                print(f'llm excessive time: {elapsed:.2f}')
            if not isinstance(response, str):
                # some provider clients return {"status": "error", ...} instead of raising;
                # callers do string ops on the result, so normalise to None
                print(f'LLM call {tag} failed: {response!r:.300}')
                logger.error(f'LLM call {tag} failed: {response!r:.1000}')
                self._record_call(tag, start, None, error=f'{response!r}')
                return None
            if stops:  # trim at a stop string the backend echoed (Claude returns eos)
                for stop in ([stops] if isinstance(stops, str) else stops):
                    eos_index = response.rfind(stop)
                    if eos_index > -1:
                        response = response[:eos_index]
            if log:
                logger.info(f'Response:\n{response}\n')
            self._record_call(tag, start, response)
            return response
        except Exception as e:
            print(f'LLM call {tag} failed:')
            traceback.print_exc()
            self._record_call(tag, start, None, error=f'{type(e).__name__}: {e}')
            return None
