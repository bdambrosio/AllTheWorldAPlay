import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import re
import traceback
import readline as rl
from fastapi import FastAPI
from utils.Messages import LLMMessage, SystemMessage, UserMessage, AssistantMessage
#from pyexts import utilityV2 as ut
#import pyexts.llmsearch.google_search_concurrent as gs
import warnings
from PyPDF2 import PdfReader
from pathlib import Path
from itertools import zip_longest
import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import wordfreq as wf
from bs4 import BeautifulSoup
from chat.OwlCoT import OwlInnerVoice
#from unstructured.partition.html import partition_html
history = {}

app = FastAPI()
cot = OwlInnerVoice(None)
PAPERS_DIR = Path.home() / '.local/share/AllTheWorld/library/papers/'
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/search/")
#{self.query}&model={GPT4}&max_chars={max_tokens*4}')
async def search(query: str, model:str = 'gpt-3.5-turbo', max_chars: int = 1200):
    response_text = ''
    storeInteraction = True
    try:
        query_phrase, keywords = ut.get_search_phrase_and_keywords(cot, query)
        print(f'query phrase: {query_phrase}\n keywords: {keywords}')
        google_result= \
            gs.search_google(query, gs.QUICK_SEARCH, query_phrase, keywords, max_chars)
        
        print(f'google_result:\n{google_result}')
        source_urls = []
        if type(google_result) is list:
            text = ''
            for item in google_result:
                text += item['text'].strip()+'\n'
                source_urls.append(item['url'])
                
        print(f'text from google_search\n{text}\n')
        prompt = [
            SystemMessage(content=f'summarize the following text, removing duplications, with respect to {query}'),
            UserMessage(content='Text:\n{{$input}}'),
        ]
        summary = cot.llm.ask({"input": text}, prompt, max_tokens=int(max_chars/2))
        return {"result":summary, "source_urls":source_urls}
    except Exception as e:
        traceback.print_exc()
    return {"result":str(e)}

def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    meta = reader.metadata
    number_of_pages = len(reader.pages)
    info = f"""
    {filepath}: 
      Author: {meta.author}
      Creator: {meta.creator}
      Producer: {meta.producer}
      Subject: {meta.subject}
      Title: {meta.title}
      Number of pages: {number_of_pages}
    """
    print(info)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        page_text = page.extract_text() + f"\nPage Number: {page_number}"
        pdf_text += page_text
        print(f"Page Number: {page_number}, chars: {len(page_text)}")
    return info, pdf_text

def convert_title_to_unix_filename(title):
    filename = title.replace(' ', '_')
    # Remove or replace special characters
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '', filename)
    filename = filename[:64]
    return filename+'.pdf'

def wait_for_download(directory):
    """
    Wait for the download to complete in the specified directory.
    Returns the filename of the downloaded file.
    """
    while True:
        for fname in os.listdir(directory):
            if not fname.endswith('.crdownload'):
                return fname
        time.sleep(1)  # Wait a bit before checking again

@app.get("/retrieve/")
async def retrieve(title: str, url: str, doc_type='str', max_chars: int = 4000):
  response = ''
  try:
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      chrome_options = Options()
      chrome_options.page_load_strategy = 'eager'
      chrome_options.add_argument("--headless")
      download_dir = "/home/bruce/Downloads/pdfs/"
      prefs = {"download.default_directory": download_dir,
         "download.prompt_for_download": False,
         "download.directory_upgrade": True,
         "plugins.always_open_pdf_externally": True}
      chrome_options.add_experimental_option("prefs", prefs)

      result = ''
      with webdriver.Chrome(options=chrome_options) as dr:
        print(f'*****setting page load timeout {20} {url}')
        dr.set_page_load_timeout(20)
        dr.get(url)
        if url.endswith('pdf') or doc_type=='pdf':
          downloaded_file = wait_for_download(download_dir)
          print("5005 retrieve downloaded file:", downloaded_file)

        response = dr.page_source
        dr.close()

        if url.endswith('pdf') or doc_type=='pdf':
          print(f'\n5005 retrieve processing pdf\n')
          filename = convert_title_to_unix_filename(title)
          print(f'5005 retrieve reading pdf {download_dir+downloaded_file}')
          pdf_info, pdf_text = read_pdf(download_dir+downloaded_file)
          # getting lots of fancy chars, bard suggests this:
          pdf_text = pdf_text.encode('utf-8', 'replace').decode('utf-8', 'replace')
          return {"result": pdf_text, "info": pdf_info, "filepath":download_dir+downloaded_file}

      print(f'\nchat processing non-pdf\n')
      soup = BeautifulSoup(response, "html.parser")
      all_text = soup.get_text()
      # Remove script and style tags
      all_text = all_text.split('<script>')[0]
      all_text = all_text.split('<style>')[0]
      # Remove multiple newline chars
      final_text = re.sub(r'(\n){2,}', '\n', all_text)
      print(f'retrieve returning {len(final_text)} chars')
      return {"result":final_text}
  except selenium.common.exceptions.TimeoutException as e:
    print(str(e))
    return {"result":'timeout'}
  except Exception as e:
    traceback.print_exc()
    return {"result":str(e)}
        
  return {"result":final_text}
  
