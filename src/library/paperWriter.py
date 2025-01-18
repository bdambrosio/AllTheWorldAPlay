import os, sys, logging, glob, time
import os, sys, glob, time
print(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import pandas as pd
import arxiv
from arxiv import Client, Search, SortCriterion, SortOrder
import ast
import concurrent, subprocess
from csv import writer
#from IPython.display import display, Markdown, Latex
import json
import re
import random
import argparse
import pickle
import openai
import traceback
from lxml import etree
from utils.pyqt import confirmation_popup
from PyPDF2 import PdfReader
import requests
import urllib.request
import numpy as np
import faiss
from scipy import spatial
from tqdm import tqdm
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
from utils.LLMRequestOptions import LLMRequestOptions
from utils import utilityV2 as ut
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec, QRect
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
from PyQt5.QtCore import pyqtSignal
import signal
import chat.OwlCoT as oiv
from utils.pyqt import ListDialog, generate_faiss_id
import semanticScholar3 as s2
import wordfreq as wf
from wordfreq import tokenize as wf_tokenize
from transformers import AutoTokenizer, AutoModel
import webbrowser
import rewrite as rw
from library.planner import Planner
from utils.llm_api import LLM
import utils.LLMScript as script   
import utils.xml_utils as xml
# startup AI resources


#from adapters import AutoAdapterModel
#embedding_model = AutoAdapterModel.from_pretrained("/home/bruce/Downloads/models/Specter-2-base")
#embedding_adapter_name = embedding_model.load_adapter("/home/bruce/Downloads/models/Specter-2", set_active=True)
embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
embedding_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
embedding_model.eval()
llm = LLM('local')


import re       

def count_keyphrase_occurrences(texts, keyphrases):
    """
    Count the number of keyphrase occurrences in each text.

    Parameters:
    texts (list): A list of texts.
    keyphrases (list): A list of keyphrases to search for in the texts.

    Returns:
    list: A list of tuples, each containing a text and its corresponding keyphrase occurrence count.
    """
    counts = []
    for text in texts:
        count = sum(text.count(keyphrase.strip()) for keyphrase in keyphrases)
        counts.append((text, count))
    
    return counts

def select_top_n_texts(texts, keyphrases, n):
    """
    Select the top n texts with the maximum occurrences of keyphrases.

    Parameters:
    texts (list): A list of texts.
    keyphrases (list): A list of keyphrases.
    n (int): The number of texts to select.

    Returns:
    list: The top n texts with the most keyphrase occurrences.
    """
    #print(f'select top n texts type {type(texts)}, len {len(texts[0])}, keys {keyphrases}')
    #print(f'select top n texts {keyphrases}')
    counts = count_keyphrase_occurrences(texts, keyphrases)
    # Sort the texts by their counts in descending order and select the top n
    sorted_texts = sorted(counts, key=lambda x: x[1], reverse=True)
    return sorted_texts[:n]



plans_filepath = os.path.expanduser("~/.local/share/AllTheWorldAPlay/arxiv/arxiv_plans.json")
plans = {}

# Ensure directory exists
os.makedirs(os.path.dirname(plans_filepath), exist_ok=True)

    
def plan_search():
    # note this uses planner working memory!
    # planner will initialize a new plan if none is selected
    plan = pl.select_plan()
    return plan

def make_search_queries(outline, section_outline, sbar):
    prompt = """
Following is an outline for a research paper, in XML format: 

{{$outline}}

From this outline generate 3 SemanticScholar search queries for the section:
{{$target_section}}

A query can contain no more than 100 characters, Respond in a plain XML format without any Markdown or code block formatting,  using the following format:

<Search>
  <query>query 1 text</query>
  <query>query 2 text</query>
  <query>query 3 text</query>
</Search>

Respond ONLY with the XML above, do not include any commentary or explanatory text.
End your response with:
<End/>
"""
    messages = [SystemMessage(content=prompt)
                ]
    response = llm.ask({"outline":xml.format_xml(outline, ), "target_section":xml.format_xml(section_outline)}, 
                      messages, stops=['<End/>'], max_tokens=150)
    queries = xml.find('<Search>', response)
    if type(queries) is not None and len(queries) >0:
        queries = xml.findall('<query>', queries)
        print(f"\nquery forsection:\n{section_outline}\nqueries:\n{'\n'.join(queries)}")
    else:
        print(f'\nquery forsection:\n{section_outline}\nqueries:\n{queries}')
    return queries

   
def s2_search (outline, section_outline, sbar=None):
    #
    ### Note - ALL THIS IS DOING IS PRE-LOADING LOCAL LIBRARY!!! Doesn't need to return anything!
    # this call is a recursive call, 'section_outline' is a subsection of 'outline' that is being processed
    #convert below to use new xml outline format with nested sections
    section_list = xml.findall('<section>', section_outline)
    if 'sections' in section_outline and xml.findall('<sections>', section_outline) is not None and len(xml.findall('<sections>', section_outline)) > 0: 
        for subsection in xml.findall('<section>', xml.find('<sections>', section_outline)):
            s2_search(outline, subsection, sbar)
    elif section_list is not None and len(section_list) > 1: 
            for section in section_list:
                s2_search(outline, section, sbar)
    else:
        queries = make_search_queries(outline, section_outline, sbar)   
        bads = ['(',')',"'",' AND', ' OR'] #tulu sometimes tries to make fancy queries, S2 doesn't like them
        if type(queries) is not list:
            print(f's2_search query construction failure')
            return None
        for query in queries:
            result_list, total_papers, next_offset = s2.get_articles(query, confirm=True)
            print(f's2_search found {len(result_list)} new papers')
            while next_offset < total_papers and confirmation_popup('Continue?', query):
                result_list, total_papers, next_offset = s2.get_articles(query, next_offset, confirm=True)
                

def write_report(app, topic):
    global updated_json, pl
    #rows = ["Query", "SBAR", "Outline", "WebSearch", "Write", "ReWrite"]
    plan = plan_search()

    if plan is None or not plan:
        plan = pl.initialize()
        pl.save_plan(plan)
        pl.save_plans()
            
    if '<sbar>' in plan and type(xml.find('<sbar>',plan)) is str and len(xml.find('<sbar>',plan)) > 0:
        if confirmation_popup('Edit existing sbar?', xml.format_xml(xml.find('<sbar>',plan))):
            plan = pl.analyze(plan)
            pl.save_plan(plan)
            pl.save_plans()
    else:
        plan = pl.analyze(plan)
        pl.save_plan(plan)
        pl.save_plans()
        
    outline = xml.find('<outline>',plan) 
    if (outline is None or not outline or len(outline) == 0 or 
        confirmation_popup('Create / Edit outline?', xml.format_xml(outline))):
        plan = pl.outline(plan, length=1200)
        pl.save_plans() # save paper_writer plan memory
        
    # get updated outline
    outline = xml.find('<outline>',plan) 
    # Note this is web search. Local faiss or other resource search will be done in Write below
    if confirmation_popup('Do web search?', 'Yes'):
        # do search - eventually we'll need subsearches: wiki, web, axriv, s2, self?, ...
        # also need to configure prompt, depth (depth in terms of # articles total or new per section?)
        s2_search(outline, outline)

    length = confirmation_popup('Report Length or No to terminate', '1200')
    if length is not None and len(length) > 0:
        try:
            length = int(length.strip())
        except:
            print(f'Invalid length: {length}, assuming 1200')
            length = 1200
    report, paper_ids = write_report_aux(outline, xml.find('<title>',outline), outline, length=length, rewrites=1)
    print(f'write_report final result\n{report}')
    with open(f'{topic}.txt', 'w') as f:
        f.write(report)
    with open(f'{topic}.ids', 'w') as f:
        f.write(str(paper_ids))
    return report
        
def write_report_aux(outline, paper_title, full_outline, length, rewrites=0):
    """Generate report text from outline.
    Returns: (text, paper_ids)
    """
    # Process direct child sections first
    sections = xml.findall('<section>', outline)
    if not sections:
        return "", set()
        
    # Calculate length per section
    section_length = length // len(sections)
    
    # Process each section and collect results
    text = []
    all_paper_ids = set()
    
    for section in sections:
        # Get section metadata
        title = xml.find('<title>', section)
        dscp = xml.find('<dscp>', section)
        
        # Generate section content
        section_text, section_paper_ids = \
            generate_section(paper_title, outline, dscp, section, title, dscp, section_length, rewrites=rewrites)
        text.append(section_text)
        all_paper_ids.update(section_paper_ids)
        
        # Handle subsections if present
        subsections = xml.find('<sections>', section)
        if subsections:
            sub_text, sub_paper_ids = write_report_aux(subsections, paper_title, full_outline, section_length, rewrites=rewrites)
            text.append(sub_text)
            all_paper_ids.update(sub_paper_ids)
            
    return "\n\n".join(text), all_paper_ids

def generate_section(paper_title, outline, background_query,section_outline, title, dscp, length, rewrites=0):
    """Generate content for a single section with rewrites.
    Returns:
        tuple: (section_text, paper_ids)
    """
    # Extract core content from tags
    section_title = xml.get_text(paper_title)
    section_dscp = xml.get_text(dscp)
    
    print(f'subsection topic {section_dscp}')
    query = background_query+', '+section_title+' '+section_dscp
    # below assumes web searching has been done
    source_texts = []
    source_text_ids = []
    print(f'** write_report_aux Doing local search ! **')
    papers = s2.search(query, section_dscp, char_limit=llm.context_size*2)
    ppr_ids = set()
    for title in papers.keys(): # title is the paper title
        paper = s2.paper_from_title(title)
        if paper is not None:
            ppr_ids.add(paper['faiss_id'])
            for section_id in papers[title]:
                source_texts.append(s2.section_from_id(section_id)['synopsis'])
                if sum(len(source_text) for source_text in source_texts) > llm.context_size*2:
                    break
        
    subsection_token_length = max(400,length) # no less than a paragraph
    print(f"\nWriting: {xml.find('<title>',section_outline)} length {length}")
    draft = rw.write(paper_title, outline, section_title, '', source_texts, '', section_dscp,
                    int(subsection_token_length), '', '', '')
    print(f'\nFirst Draft:\n{draft}\n')
    if rewrites < 1:
        return draft, ppr_ids

    return draft, ppr_ids

class DisplayApp(QtWidgets.QWidget):
    def __init__(self, query, paper_id, sections=[], dscp='', template=''):
        super().__init__()
        self.query = query
        self.paper_id = paper_id
        self.paper_row = None
        search_result = s2.paper_library_df[s2.paper_library_df['faiss_id']==int(paper_id)]
        if len(search_result) > 0:
            self.paper_row = search_result.iloc[0]
            print(f'found paper {self.paper_row.title}')
        else:
            print(f"couldn't find paper {self.paper_id}")
        print(f'sections {type(sections)}, {len(sections)}\n{sections[0]}')
        self.sections = sections # [[section_pd, ...], [section_synopsis, ...]]
        self.dscp = dscp
        self.template = template

        self.memory_display = None
        self.windowCloseEvent = self.closeEvent
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("#202020"))  # Use any hex color code
        self.setPalette(palette)
        self.codec = QTextCodec.codecForName("UTF-8")
        self.widgetFont = QFont(); self.widgetFont.setPointSize(14)
        
        # Main Layout
        main_layout = QHBoxLayout()
        # Text Area
        text_layout = QVBoxLayout()
        main_layout.addLayout(text_layout)
        
        class MyTextEdit(QTextEdit):
            def __init__(self, app):
                super().__init__()
                self.app = app
                self.textChanged.connect(self.on_text_changed)
                
            def on_text_changed(self):
                #legacy from Owl, but who knows
                pass
            
            def keyPressEvent(self, event):
                #legacy from Owl, but who knows
                if event.matches(QKeySequence.Paste):
                    clipboard = QApplication.clipboard()
                    self.insertPlainText(clipboard.text())
                else:
                    super().keyPressEvent(event)
            
        self.mainFont = QFont("Noto Color Emoji", 14)

        self.input_area = MyTextEdit(self)
        self.input_area.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.input_area.setFont(self.widgetFont)
        self.input_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
        text_layout.addWidget(self.input_area)
      
        self.display_area = MyTextEdit(self)
        self.display_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.display_area.setFont(self.widgetFont)
        self.display_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
        text_layout.addWidget(self.display_area)

        # Buttons and Comboboxes
        self.discuss_button = QPushButton("discuss")
        self.discuss_button.setFont(self.widgetFont)
        self.discuss_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
        self.discuss_button.clicked.connect(self.discuss)
        text_layout.addWidget(self.discuss_button)
      
        self.describe_button = QPushButton("describe")
        self.describe_button.setFont(self.widgetFont)
        self.describe_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
        self.describe_button.clicked.connect(self.describe)
        text_layout.addWidget(self.describe_button)
      
        self.setLayout(main_layout)
        self.show()
        
        
    def display_response(self, r, clear=False):
        if clear:
            self.input_area.clear()
        self.input_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
        r = str(r)
        # presumably helps handle extended chars
        encoded = self.codec.fromUnicode(r)
        decoded = '\n'+encoded.data().decode('utf-8')+'\n'
        self.input_area.insertPlainText(decoded)  # Insert the text at the cursor position
        self.input_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
        self.input_area.repaint()

    def display_msg(self, r, clear=False):
        if clear:
            self.display_area.clear()
        self.display_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
        r = str(r)
        # presumably helps handle extended chars
        encoded = self.codec.fromUnicode(r)
        decoded = encoded.data().decode('utf-8')+'\n'
        self.display_area.insertPlainText(decoded)  # Insert the text at the cursor position
        self.display_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
        self.display_area.repaint()

    def discuss(self):
        input_text = self.input_area.toPlainText()
        self.input_area.clear()
        print(f'calling discuss query{self.query} sections {len(self.sections[0])}, dscp {self.dscp}')
        discuss_resources(self, input_text, self.paper_id, self.sections, self.query+', '+self.dscp, self.template)

    def describe(self):
        print('describe')
        self.display_response(self.paper_row['extract'], clear=True)


def discuss_resources(display, query, paper_id, sections, dscp='', template=None):
    """ sections is [section_id, ...]"""
    global updated_json

    rows = ["Write", "ReWrite"]

    print(f"discuss resources sections provided {[section_id for section_id in sections[0]]}")
    length = 400 # default response - should pick up from ui max_tokens if run from ui, tbd
    instruction_prompt = """Generate a concise instruction for a research assistant creating a response to the Query below, given the Background information provided.

<Background>
{{$dscp}}
</Background>

<Query>
{{$query}}
</Query>

Respond only with the instruction with no additional introductory, explanatory, or discursive text.
End your response with:
</End>
"""
    messages = [SystemMessage(content='You are a skilled prompt constructor.'),
                UserMessage(content=instruction_prompt),
                ]
    instruction = llm.ask({"dscp":dscp, "query":query, "dscp": dscp},
                              messages,  max_tokens=120, stops=['</End>']) 

    outline = {"title": query, "rewrites": 1, "length": length, "dscp": str(instruction),
               "task": str(instruction)}
    display.display_msg(json.dumps(outline, indent=2))

    print(f'discuss_resources sections {sections}')
    report, paper_ids = write_report_aux(paper_outline=outline, section_outline=outline, heading_1_title=query, 
                                         length=length, resources=[paper_id, sections])
    
    print(f'discuss final result\n{report}')
    display.display_response(report)
    display.display_msg(paper_ids)
    return report

if __name__ == '__main__':
    def parse_arguments():
        """Parses command-line arguments using the argparse library."""
        
        parser = argparse.ArgumentParser(description="Process a single command-line argument.")
        
        parser.add_argument("-discuss", type=str, help="discuss a provided set of resources")
        parser.add_argument("-report", type=str, help="discuss a provided set of resources")
        parser.add_argument("-template", type=str, help="llm prompt template")
        args = parser.parse_args()
        return args
    try:
        args = parse_arguments()
        if hasattr(args, 'discuss') and args.discuss is not None:
            #print(args.discuss)
            with open(args.discuss, 'rb') as f:
                resources = pickle.load(f)
            #print(f'\nResources:\n{json.dumps(resources, indent=2)}')
            app = QApplication(sys.argv)
            cot = oiv.OwlInnerVoice(None, template=resources['template'], port=resources['port'])
            # set cot for rewrite so it can access llm
            rw.cot = cot
            s2.cot = cot
            print(f'resources {json.dumps(resources, indent=2)}')
            print(f'paper_id {resources["papers"][0]}')
            #print(f'resources [0][1:] {resources["papers"][1]}')
            window = DisplayApp(query='Question?',
                                paper_id=resources['papers'][0],
                                sections=resources['papers'][1],
                                dscp=resources['dscp'],
                                template=resources['template'])
            #window.show()
            app.exec()
            sys.exit(0)

        if hasattr(args, 'report') and args.report is not None:
            pl = Planner(None, 'local')
            llm = LLM('local')
            app = QApplication(sys.argv)
            write_report(app, args.report)
            sys.exit(0)
        else:
            llm = LLM('deepseek')
            pl = Planner(None, 'deepseek')
            app = QApplication(sys.argv)
            write_report(app, 'conversational recommender literature review')
            print('paper_writer.py -report expects to be called from Owl with topic')
            pass
    except Exception as e:
        traceback.print_exc()
        print(str(e))
