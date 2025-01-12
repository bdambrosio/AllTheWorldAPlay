import os, sys, glob, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import openai
import requests
import re
import time
import random
import traceback
import concurrent.futures
import threading as th
import json
import tracemalloc
import os
import linecache
import nltk
import selenium
import inspect
import importlib
from typing import Any, Dict, List, Tuple
import openai
import requests
import time
import traceback
import json
import os
from typing import Any, Dict, List, Tuple
#import chat.OwlCot as owlCot
from utils.Messages import LLMMessage, UserMessage, AssistantMessage, asdicts
#import utils.pyexts.conversation as cv 
openai.api_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_KEY")
google_cx = os.getenv("GOOGLE_CX")
GOOGLE = 'google'

host='127.0.0.1'
port = 5004

def get_available_models():
    return list(cv.conv_templates.keys())

"""
def ask_LLM(model, gpt_message, max_tokens=100, temp=0.3, top_p=1.0, host=host, port=port,
            stop=None, stop_on_json=False, choice_set=[], display=None):
    completion = None
    response = ''
    #print(f'***** utility ask_LLL temperature {temp}')
    try:
      if model.lower().startswith('gpt'):
        stream= openai.ChatCompletion.create(
            model=model, messages=gpt_message, max_tokens=max_tokens, temperature=temp, top_p=1, stop=stop, stream=True)
        response = ''
        if stream is None:
          return response
        for chunk in stream:
          item = chunk['choices'][0]['delta']
          if 'content' in item.keys():
            if display is not None:
              display(chunk['choices'][0]['delta']['content'])
              response += chunk['choices'][0]['delta']['content']

      else:
          completion = llm.run_query(model, gpt_message, max_tokens, temp, top_p,
                                     stop=stop, stop_on_json=stop_on_json, choice_set=choice_set,
                                     host=host, port=port, display=display)
          if completion is not None:
              response = completion

    except:
        traceback.print_exc()
    return response

def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

def extract_site(url):
    site = ''
    base= findnth(url, '/',2)
    if base > 2: site = url[:base].split('.')
    if len(site) > 1: site = site[-2]
    site = site.replace('https://','')
    site = site.replace('http://','')
    return site

def extract_domain(url):
    site = ''
    base= findnth(url, '/',2)
    if base > 2: domain = url[:base].split('.')
    if len(domain) > 1: domain = domain[-2]+'.'+domain[-1]
    domain = domain.replace('https://','')
    domain = domain.replace('http://','')
    return domain

def part_of_keyword(word, keywords):
    for keyword in keywords:
        if word in keyword:
            return True
    return False

"""

pattern = r'\{\$[^}]*\}'
def run_request(client, bindings, prompt, options, template):
    """ substitute variables, then apply template """
    #
    ### first substitute for {{$var-name}} in prompt
    #
    global pattern
    #print(f'run_request {bindings}\n{prompt}\n')
    if bindings is None or len(bindings) == 0:
        substituted_prompt = prompt
    else:
        substituted_prompt = []
        for message in prompt:
            if type(message.content) is not str: # substitutions only in strings
                substituted_prompt.append(message)
                continue
            matches = re.findall(pattern, message.content)
            new_content = message.content
            for match in matches:
                var = match[2:-1]
                #print(f'var {var}, content {new_content}')
                if var in bindings.keys():
                    new_content = new_content.replace('{{$'+var+'}}', str(bindings[var]))
                else:
                    raise ValueError(f'unbound prompt variable {var}')
            substituted_prompt.append(LLMMessage(message.role, new_content))
    
    #
    ## now apply template
    #

    if template.lower().startswith('gpt'): #gpt doesn't need one
        response = client.executeRequest(asdicts(substituted_prompt), options, template)
        return response

    elif template.lower().startswith('claude'): #claude client handles formatting
        response = client.executeRequest(asdicts(substituted_prompt), options, template)
        return response

    #OSClient - run template
    conv=cv.get_conv_template(template)
    user_role = conv.roles[0]
    asst_role = conv.roles[1]
    system_role = conv.roles[2] if len(conv.roles)>2 else conv.roles[0]
    
    for msg_idx, msg in enumerate(substituted_prompt):
        conv.append_message(msg.role, msg.content)
    # priming prompt
    if conv.response_prime and (substituted_prompt[len(substituted_prompt)-1].role != 'assistant')\
       and conv.name != 'command-r' and not conv.name.startswith('llama'):
        conv.append_message('assistant','')
    prompt = conv.get_prompt()
    prompt = re.sub('\n{3,}', '\n\n', prompt)

    #print(f'calling client.execute \n---\n{prompt}\n---\n')
    response = client.executeRequest(prompt, options, conv)
    if response['status'] != 'success':
        return response
    if not isinstance(response['message'], dict):
        #print(f"***** utility run_request repair {type(response['message'])}\n {response['message']}")
        response['message'] = {'role': 'assistant', 'content': response['message'] or ''}
    return response

def get_search_phrase_and_keywords(cot, query_string):
    print(f'get phrase query_string:{query_string}')
    phrase = ''; keywords = []

    phrase_prompt = [UserMessage(content="""<Text>
{{$input}}
</Text>

Analyze the above Text and generate a concise google search engine search phrase.
Respond  only the the search phrase
Do not include any introductory, explanatory, or discursive text.
End your response with:
</END>
""")
              ]
    response = cot.llm.ask({'input':query_string}, phrase_prompt, max_tokens=100, eos='</END>')
    phrase = response.strip()

    keyword_prompt = [UserMessage(content="""<Text>
{{$input}}
</Text>

Analyze the above Text and extract all keywords including any named-entities.
Respond with the list of keywords, and only the list of keywords (including any named-entities)
Do not include any introductory, explanatory, or discursive text.
End your response with:
</END>
""")
                      ]
    response = cot.llm.ask({'input':query_string}, keyword_prompt, max_tokens=100, eos='</END>')
    keywords = response
    return phrase, keywords


def reform(elements):
  #reformulates text extracted from a webpage by unstructured.partition_html into larger keyword-rankable chunks
  texts = [] # a list of text_strings, each of at most *max* chars, separated on '\n' when splitting an element is needed
  paragraphs = []
  total_elem_len = 0
  for element in elements:
    text = str(element)
    total_elem_len += len(text)
    if len(text) < 4: continue
    elif len(text)< 500:
      texts.append(text)
    else:
      subtexts = text.split('\n')
      for subtext in subtexts:
        if len(subtext) < 500:
          texts.append(subtext)
        else:
          texts.extend(nltk.sent_tokenize(subtext))
      
  # now reassemble shorter texts into chunks
  paragraph = ''
  total_pp_len = 0
  for text in texts:
    if len(text) + len(paragraph) < 500 :
      paragraph += ' '+text
    else:
      if len(paragraph) > 0: # start a new paragraph
        paragraphs.append(paragraph)
        paragraph=''
      paragraph += text
  if len(paragraph) > 0:
    paragraphs.append(paragraph+'.\n')
  total_pp_len = 0
  for paragraph in paragraphs:
    total_pp_len += len(paragraph)
  return paragraphs
