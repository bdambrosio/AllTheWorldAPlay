import concurrent.futures
import json
import sys
import os
import time
import string
from datetime import date
from datetime import datetime
import random
import traceback
import re
import requests
import copy
from itertools import zip_longest
#import urllib3
import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import wordfreq as wf
from unstructured.partition.html import partition_html
import nltk
import urllib.parse as en

# will be provided by client if None
cot = None

import warnings
from utils.Messages import UserMessage
#from utils.DefaultResponseValidator import DefaultResponseValidator
#from utils.JSONResponseValidator import JSONResponseValidator
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_KEY")
google_cx = os.getenv("GOOGLE_CX")
GOOGLE = 'google'

today = " as of "+date.today().strftime("%b-%d-%Y")+'\n\n'

suffix = "\nA: "
client = '\nQ: '
QUICK_SEARCH = 'quick'
NORMAL_SEARCH = 'moderate'
DEEP_SEARCH = 'deep'

system_prime = {"role": "system", "content":"You analyze Text with respect to Query and list any relevant information found, including direct quotes from the text, and detailed samples or examples in the text."}
priming_1 = {"role": "user", "content":"Query:\n"}
priming_2 = {"role": "user", "content":"List relevant information in the provided text, including direct quotes from the text. If none, respond 'no information'.\nText:\n"}

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


# Define a function to make a single URL request and process the response
def process_url(query_phrase, keywords, keyword_weights, url, timeout, max_chars):
    start_time = time.time()
    site = extract_site(url)
    result = ''
    print(f'process_url: {url}')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            options = Options()
            options.page_load_strategy = 'eager'
            options.add_argument("--headless")
            result = ''
            with webdriver.Chrome(options=options) as dr:
                print(f'*****setting page load timeout {timeout} {url}')
                dr.set_page_load_timeout(timeout)
                try:
                    dr.get(url)
                    response = dr.page_source
                    result = response_text_extract(query_phrase, keywords, keyword_weights, url, response,
                                                   int(time.time()-start_time),  max_chars)
                except selenium.common.exceptions.TimeoutException as e:
                    print(str(e))
                    return '', url
    except Exception:
        traceback.print_exc()
        print(f"{site} err")
        pass
    print(f"Processed {site}: {len(response)} / {len(result)} {int((time.time()-start_time)*1000)} ms")
    return result, url

def process_urls(query_phrase, keywords, keyword_weights, urls, search_level, timeout, max_chars):
    response = []
    start_time = time.time()
    full_text = ''
    in_process = []
    max_w = 6
    #print(f' normal {NORMAL_SEARCH}, quick {QUICK_SEARCH}, level {search_level}')
    # Create a ThreadPoolExecutor with 5 worker threads
    print(f'process_urls timeout, max_chars: {timeout}, {max_chars}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
        # initialize scan of google urls
        try:
            print(f'process_urls remaining urls: {len(urls)}')
            while len(urls) > 0 or len(in_process) >0:
                # empty queue if out of time
                if len(full_text) > max_chars or time.time()-start_time > 16:
                    urls = []
                elif len(urls) > 0:
                    url = urls[0]
                    urls = urls[1:]
                    # set timeout so we don't wait for a slow site forever
                    timeout = 20-int(time.time()-start_time)

                    print(f'process_urls submit {url}\n {timeout}, {max_chars}')
                    future = executor.submit(process_url, query_phrase, keywords, keyword_weights, url, timeout, max_chars)
                    in_process.append(future)
                    print(f'added one to in_process {len(urls)}, {len(in_process)}')
                # Process the responses as they arrive
                for future in in_process:
                    if future.done():
                        in_process.remove(future)
                        print(f'removed one from in_process {len(urls)}, {len(in_process)}')
                        try:
                            result, url = future.result()
                            print(f'got result: {result}')
                        except Exception as e:
                            print(str(e))
                            continue
                        if len(result) > 0:
                            site = extract_site(url)
                            domain = extract_domain(url)
                            response.append({'source':extract_domain(url), 'url':url, 'text':result})

                if len(in_process) < max_w:
                    time.sleep(.05)
                else:
                    time.sleep(.5)
                        
            executor.shutdown(wait=True)
            #print(f'process_urls returning')
            return response
        except:
            traceback.print_exc()
            executor.shutdown(wait=False)
        return response

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
    
def extract_subtext(text, query_phrase, keywords, keyword_weights, max_chars):
    ###  maybe we should score based on paragraphs, not lines?
    sentences = reform(text)
    sentence_weights = {}
    final_text = ''
    for sentence in sentences:
        sentence_weights[sentence] = 0
        for keyword in keywords:
            if keyword in sentence or keyword.lower() in sentence:
                if keyword in keyword_weights.keys():
                    sentence_weights[sentence]+=keyword_weights[keyword]
                    
    # now pick out sentences starting with those with the most keywords
    max_sentence_weight = 0
    for keyword in keyword_weights.keys():
        max_sentence_weight += keyword_weights[keyword]
    for i in range(max_sentence_weight,1,-1):
        if len(final_text)> 3000 and i < max(1, int(max_sentence_weight/4)): # make sure we don't miss any super-important text
            return final_text
        for sentence in sentences:
            if len(final_text)+len(sentence)>3001 and i < max(1, int(max_sentence_weight/4)):
                continue
            if sentence_weights[sentence] == i:
                final_text += sentence

    return final_text


def search(query_phrase):
    sort = '&sort=date-sdate:d:w'
    if 'today' in query_phrase or 'latest' in query_phrase:
        sort = '&sort=date-sdate:d:s'
    try:
        google_query=en.quote(query_phrase)
    except:
        return []
    response=[]
    try:
        start_wall_time = time.time()
        url="https://www.googleapis.com/customsearch/v1?key="+google_key+'&cx='+google_cx+'&num=10'+sort+'&q='+google_query
        response = requests.get(url)
        print(f'google raw:\n{response}\n')
        response_json = json.loads(response.text)
        print(f'google json:\n{response_json}\n')
    except:
      traceback.print_exc()
      return []

    #see if we got anything useful from google
    if 'items' not in response_json.keys():
        print(f'no items in keys: {response_json.keys()}\n')
        return []

    # compile list of urls
    urls = []
    for i in range(len(response_json['items'])):
        url = response_json['items'][i]['link'].lstrip().rstrip()
        print(f' appending: {url}')
        urls.append(url)
    return urls

def log_url_process(site, reason, raw_text, extract_text, gpt_text):
    return


def llm_tldr (text, query, max_chars):
    #text = text[:max_chars] # make sure we don't run over token limit
    prompt = [UserMessage(content='Analyze the following Text to identify if there is any content relevant to the query:\n{{$query}}.\nRespond using this JSON template:\n\n{"relevant": "Yes" if there is any text found in the input that is relevant to the query, "No" otherwise>, "extract": "<all relevant content found in Text, with irrelevant text removed. >"}\n\nText:\n{{$input}}\n. ')]
    response_text=''
    completion = None
    schema = {
        "schema_type":"object",
        "title":"tldr",
        "description":"extract query-relevant text",
        "properties":{
            "relevant": {
                "type": "string",
                "description": "Yes if any relevant text, otherwise No"
            },
            "extract": {
                "type": "string",
                "description": "relevant full extract from Text, or empty string"
            }
        },
        "required":["relevant", "extract"],
        "returns":"extract"
    }
    
    # don't clutter primary memory with tldr stuff
    response = cot.llm.ask({'input':text, 'query':query}, prompt, max_tokens=500, model='gpt-4o-mini')
    
    print(f'\n***** google llm_tldr processing {response is dict}\n{response}')
    if response is not dict:
        try:
            response = json.loads(response)
        except Exception as e:
            print(f'google llm_tldr response is not a dict')
            return ''
    if 'relevant' in response:
        if 'yes' in str(response['relevant']).lower(): # type: ignore
            if 'extract' in response: # type: ignore
                return response['extract'] # type: ignore
        else:
            return ''

    else:
        print(f'google llm_tldr response is not a dict')
        return ''
    

def response_text_extract(query_phrase, keywords, keyword_weights, url, response, get_time, max_chars):
    curr=time.time()
    extract_text = ''
    site = extract_site(url)
    if url.endswith('pdf'):
        pass
    else:
        elements = partition_html(text=response)
        str_elements = []
        for e in elements:
            stre = str(e).replace('  ', ' ')
            str_elements.append(stre)
        extract_text = extract_subtext(str_elements, query_phrase, keywords, keyword_weights, int(max_chars))
        print(extract_text)
    if len(extract_text.strip()) < 8:
        return ''

    # now ask openai to extract answer
    response = ''
    response = llm_tldr(extract_text, query_phrase,  int(max_chars))
    return response

def extract_items_from_numbered_list(text):
    items = ''
    elements = text.split('\n')
    for candidate in elements:
        candidate = candidate.lstrip('. \t')
        if len(candidate) > 4 and candidate[0].isdigit():
            candidate = candidate[1:].lstrip('. ')
            if len(candidate) > 4 and candidate[0].isdigit(): # strip second digit if more than 10 items
                candidate = candidate[1:].lstrip('. ')
            items += candidate+' '
    return items

def compute_keyword_weights(keywords):
    index = 0; tried_index = 0
    full_text=''
    keyword_weights = {}
    for keyword in keywords:
        zipf = wf.zipf_frequency(keyword, 'en')
        weight = max(0, int((8-zipf)))
        if weight > 0:
            keyword_weights[keyword] = weight
            subwds = keyword.split(' ')
            if len(subwds) > 1:
                for subwd in subwds:
                    sub_z = wf.zipf_frequency(subwd, 'en')
                    sub_wgt = max(0, int((8-zipf)*1/2))
                    if sub_wgt > 0:
                        keyword_weights[subwd] = sub_wgt
    return keyword_weights


"""
def get_url(query, url, client, model, max_chars):
    try:
        keywords = query.split(' ')
        weights = compute_keyword_weights(keywords)
        full_text = \
            process_urls(query, keywords, weights, [url], search_level, 10, max_chars)
    except:
        traceback.print_exc()
    return  full_text
"""

def search_google(client_cot, original_query, search_level, query_phrase, keywords, max_chars):
  global cot
  if cot is None:
      cot = client_cot
  start_time = time.time()
  all_urls=[]; urls_used=[]; urls_tried=[]
  index = 0; tried_index = 0
  full_text=''
                  
  weights = compute_keyword_weights(keywords)
  try:  # query google for recent info
    sort = ''
    if 'today' in original_query or 'latest' in original_query:
        original_query = today.strip('\n')+' '+original_query
    extract_query = ''
    orig_phrase_urls = []
    if len(original_query) > 0:
        orig_phrase_urls = search(original_query[:min(len(original_query), 128)])
        extract_query = original_query[:min(len(original_query), 128)]
    gpt_phrase_urls = []
    if len(query_phrase) > 0:
        gpt_phrase_urls = search(query_phrase)
        extract_query = query_phrase # prefer more succint query phrase if available
    if len(orig_phrase_urls) == 0 and len(gpt_phrase_urls) == 0:
        return ''

    for url in orig_phrase_urls:
        if url in gpt_phrase_urls:
            gpt_phrase_urls.remove(url)

    # interleave both lists now that duplicates are removed
    urls = [val for tup in zip_longest(orig_phrase_urls, gpt_phrase_urls) for val in tup if val is not None]
    all_urls = copy.deepcopy(urls)
    print(f'all urls:\n{all_urls}\n')
    digest_w_urls = \
        process_urls(extract_query, keywords, weights, all_urls, search_level, 10, max_chars)
  except:
      traceback.print_exc()
  print(f'\n\nFinal google search result\n{digest_w_urls}\n')
  return  digest_w_urls

