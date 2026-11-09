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
        self.model = model_name if model_name is not None else MODEL
        if self.server_name == 'openai':
            from utils.OpenAIClient import OpenAIClient
            self.openai_client = OpenAIClient(model_name=self.model)
        if self.server_name == 'vllm':
            self.model = _served_model_id(LLM_BASE_URL)

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
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=body, timeout=600)
        if response.status_code != 200:
            raise Exception(f'{response.status_code} from LLM server: {response.text[:500]}')
        try:
            text = response.json()['choices'][0]['message']['content'] or ''
        except (ValueError, KeyError, IndexError):
            return response.content.decode('utf-8')
        if (index := text.find('</think>')) > -1:
            text = text[index + 8:].strip()
        return text

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
                return None
            if stops:  # trim at a stop string the backend echoed (Claude returns eos)
                for stop in ([stops] if isinstance(stops, str) else stops):
                    eos_index = response.rfind(stop)
                    if eos_index > -1:
                        response = response[:eos_index]
            if log:
                logger.info(f'Response:\n{response}\n')
            return response
        except Exception:
            print(f'LLM call {tag} failed:')
            traceback.print_exc()
            return None
