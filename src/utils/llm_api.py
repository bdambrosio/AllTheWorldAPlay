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
# settings follow Cognitive_workbench measure/models/fw_glm53flash.yaml. Key: FIREWORKS_PROJECT_KEY.
FIREWORKS_URL = 'https://api.fireworks.ai/inference/v1'
FIREWORKS_MODEL = 'accounts/fireworks/models/glm-5p3-flash'
# GLM-5.3 on Fireworks is thinking-only: reasoning_effort 'low' is the least it takes ('none',
# enable_thinking False and thinking disabled are refused). It spent up to ~250 reasoning tokens
# on a check, from the same budget as the answer: at 96 the JSON was cut off. Used only if spent.
FIREWORKS_HEADROOM = 512
# Claude Haiku 5.5 on the Anthropic API (server_name 'haiku'). Key: CLAUDE_API_KEY or ANTHROPIC_API_KEY.
# How much every call thinks before it answers: ATWAP_EFFORT = off | low | medium | high. Calls
# named by set_reasoning think at the effort given there whatever this says, so
# ATWAP_EFFORT=off ATWAP_REASONING='page+plan=medium' thinks only where the judgement is.
# Measured on 24 prompts of a play (2026-11-13), median seconds a call: off 0.9, low 1.5,
# medium 1.8; GLM-5.3-Flash on Fireworks 2.2. The engine's calls answer in about 40 tokens, so
# the time is nearly all thinking.
HAIKU_MODEL = 'claude-haiku-5-5'
HAIKU_EFFORTS = ('off', 'low', 'medium', 'high')
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


def generate_image(llm=None, description='', size='512x512', filepath='test.png', seed=None):
    """Fetch an image from the local image server (utils/klein_serve.py); grey placeholder on failure.
    The prompt is sent as given: its style is the caller's to say (sim/stage.py picture). seed:
    the same seed and words give the same picture, which is how a character keeps a face."""
    image_content = None
    try:
        response = requests.get(IMAGE_URL, params={"prompt": description, "size": size, **({"seed": seed} if seed is not None else {})},
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


# ---- reasoning by call (experiment): which engine calls think before they answer.
# Every call ran with thinking off. set_reasoning('speech=medium') turns it on for groups of
# calls, named by tag; the local server takes reasoning_effort low|medium|high. A thinking call
# gets REASONING_TOKENS on top of its own max_tokens, since the thinking is spent from the same budget.
REASONING_GROUPS = {
    # what the audience reads, and the check on it
    'page': ('Character.generate_acts', 'generate_dialog_turn', 'bystander_cut_in', 'Context.do', 'Context.look',
             'record.check', 'reground_scene'),
    # plans and judgements behind the page
    'plan': ('NarrativeCharacter.write_narrative', 'NarrativeCharacter.replan_narrative_act', 'integrate_narratives',
             'Character.generate_task_plan', 'instantiate_narrative_goal', 'check_post_state_ambiguity',
             'check_decision_required', 'test_termination', 'told_briefly', 'staleness_analysis'),
}
REASONING_TOKENS = 3000
_reasoning: dict = {}


# A line put at the head of every prompt of the play in progress: what no call may be left
# without. The engine has some forty kinds of call, each with its own prompt; the people of the
# play and whether each is he or she had been added to five of them, and the sixth (a bystander
# cutting in) called a man 'her', after which the lines that followed copied it.
_preamble = None


def set_preamble(fn):
    """fn() -> a short text, or '' for none; None to clear."""
    global _preamble
    _preamble = fn


def set_reasoning(spec):
    """spec: 'page=medium', 'page+plan=low', 'Context.do,record.check=high', or None for none."""
    global _reasoning
    _reasoning = {}
    for part in (spec or '').split(';'):
        if not part.strip():
            continue
        names, _, effort = part.partition('=')
        effort = effort.strip() or 'medium'
        if effort not in ('low', 'medium', 'high'):
            raise ValueError(f"reasoning effort '{effort}' is not low, medium or high")
        for name in re.split(r'[+,]', names):
            for tag in REASONING_GROUPS.get(name.strip(), (name.strip(),)):
                if tag:
                    _reasoning[tag] = effort


def reasoning():
    return dict(_reasoning)


# ---- the planner: the calls that write the play's plan go to the server named in
# plays/config.py (plan_server_name), whichever model plays the parts.
PLAN_TAGS = frozenset((
    'NarrativeCharacter.negotiate_central_narrative', 'merge_central_narratives',
    'NarrativeCharacter.write_narrative', 'integrate_narratives', 'act_central_narrative',
    'frame_episode', 'NarrativeCharacter.replan_narrative_act', 'NarrativeCharacter.replan_scene',
    'reground_scene', 'plan_serves_drives',
    'check_decision_required',   # whether a scene turns on a choice is the director's judgement
    'check_post_state_ambiguity'))   # what a scene or act came to: the acting model passed an ending that was not played
_planner = False   # False: not looked up yet; None: no planner; else the planner's LLM


def planner():
    global _planner
    if _planner is False:
        _planner = None
        try:
            import plays.config as config
            if getattr(config, 'plan_server_name', None):
                _planner = LLM(config.plan_server_name, getattr(config, 'plan_model_name', None))
        except Exception as e:
            print(f'no planner: {e}')
    return _planner


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
        if self.server_name == 'haiku':
            import anthropic
            self.model = model_name or HAIKU_MODEL
            self.effort = os.environ.get('ATWAP_EFFORT', 'medium').strip().lower()
            if self.effort not in HAIKU_EFFORTS:
                raise ValueError(f"ATWAP_EFFORT '{self.effort}' is not one of {', '.join(HAIKU_EFFORTS)}")
            key = os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('CLAUDE_API_KEY')
            if not key:
                raise RuntimeError('CLAUDE_API_KEY is not set (source ~/.config/secrets.env)')
            self.anthropic_client = anthropic.Anthropic(api_key=key, max_retries=4, timeout=120.0)
            print(f'  {self.model}, effort {self.effort}')

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
        note = _preamble() if _preamble is not None else ''
        if note and substituted_prompt:   # what every call of this play is told (set_preamble)
            first = dict(substituted_prompt[0])
            if isinstance(first.get('content'), str):
                first['content'] = f"{note}\n\n{first['content']}"
                substituted_prompt = [first] + substituted_prompt[1:]
        self._last_prompt = substituted_prompt   # as sent, for the run's prompt log (sim/trace.py)

        if trace:
            print(f'\n{json.dumps(substituted_prompt)}\n')
        if log:
            logger.info(f'Prompt: {substituted_prompt}\n')

        if 'openai' in self.server_name:
            return self.openai_client.executeRequest(prompt=substituted_prompt, options=options)
        if self.server_name == 'fireworks':
            return self._fireworks_request(substituted_prompt, options)
        if self.server_name == 'haiku':
            return self._haiku_request(substituted_prompt, options)
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
            effort = getattr(options, 'reasoning', None)
            if effort:   # set_reasoning: this call thinks first, on a budget of its own
                del body["chat_template_kwargs"]
                body["reasoning_effort"] = effort
                body["max_tokens"] = options.max_tokens + REASONING_TOKENS
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

    def _haiku_request(self, messages, options):
        """One Messages API call. The system turns that open the prompt go in `system`; a prompt
        that is all system turns is sent as the user's. This model takes no temperature or
        top_p (a value other than the default is refused), so the engine's are not sent: how
        far it thinks is its one setting. The SDK retries 429, 5xx and dropped connections."""
        lead = 0
        while lead < len(messages) and messages[lead]['role'] == 'system':
            lead += 1
        system = '\n\n'.join(m['content'] for m in messages[:lead])
        turns = [{'role': 'user' if m['role'] == 'system' else m['role'], 'content': m['content']}
                 for m in messages[lead:] if str(m['content']).strip()]
        while turns and turns[-1]['role'] == 'assistant':   # a reply cannot be started for this model
            turns.pop()
        if not turns:
            turns, system = [{'role': 'user', 'content': system}], ''
        effort = getattr(options, 'reasoning', None) or self.effort
        request = {'model': self.model, 'messages': turns, 'max_tokens': options.max_tokens}
        if system:
            request['system'] = system
        stops = [s for s in (options.stops or []) if s and s.strip()]
        if stops:
            request['stop_sequences'] = stops
        if effort == 'off':
            request['thinking'] = {'type': 'disabled'}
        else:   # the thinking is spent from the same budget as the answer
            request.update(thinking={'type': 'adaptive'}, output_config={'effort': effort},
                           max_tokens=options.max_tokens + REASONING_TOKENS)
        reply = self.anthropic_client.messages.create(**request)
        usage = reply.usage
        cached = getattr(usage, 'cache_read_input_tokens', 0) or 0
        self._last_usage = ({'prompt_tokens': (usage.input_tokens or 0) + cached, 'completion_tokens': usage.output_tokens,
                             'prompt_tokens_details': {'cached_tokens': cached}}, reply.stop_reason)
        if reply.stop_reason == 'refusal':
            raise Exception(f'{self.model} declined the request')
        return ''.join(block.text for block in reply.content if block.type == 'text')

    def _fireworks_request(self, messages, options, retries=6):
        """Fireworks chat completion with backoff on 429/5xx (the account's rate limit, not the
        model's, is what throttles; see the workbench yaml)."""
        key = os.environ.get('FIREWORKS_PROJECT_KEY')
        if not key:
            raise RuntimeError('FIREWORKS_PROJECT_KEY is not set (source ~/.config/secrets.env)')
        # the engine's budgets are for the answer alone (some are 10 or 20 tokens); this model
        # spends a few reasoning tokens from the same budget even at 'low', so it gets headroom
        body = {"model": self.model, "messages": messages, "temperature": options.temperature,
                "top_p": options.top_p, "max_tokens": options.max_tokens + FIREWORKS_HEADROOM,
                "stop": options.stops or None, "reasoning_effort": getattr(self, 'reasoning_effort', 'low')}
        delay = 2.0
        for attempt in range(retries):
            try:   # a connection that hung held the play for five minutes: give up sooner, try again
                response = requests.post(f'{FIREWORKS_URL}/chat/completions', json=body, timeout=(10, 120),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt >= 1:
                    raise
                continue
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
                          response_chars=len(response or ''),
                          prompt=getattr(self, '_last_prompt', None), response=response)
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
        options.reasoning = _reasoning.get(tag)
        if tag in PLAN_TAGS and (plan := planner()) is not None and plan is not self and plan.server_name != self.server_name:
            plan.owner = self.owner
            response = plan.ask(input, prompt_msgs, template=template, tag=tag, temp=temp, max_tokens=max_tokens,
                                top_p=top_p, stops=stops, stop_on_json=stop_on_json, model=model, log=log, trace=trace)
            if response is not None:
                return response
            print(f'planner failed on {tag}: the playing model writes it')
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
