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
Following is an outline for a research paper, in JSON format: 

{{$outline}}

From this outline generate 3 SemanticScholar search queries for the section:
{{$target_section}}

A query can contain no more than 100 characters, Respond in a plain JSON format without any Markdown or code block formatting,  using the following format:
{"query0":'query 0 text',"query1": 'query 1 text', "query2": query 2 text'}

Respond ONLY with the JSON above, do not include any commentary or explanatory text.
"""
    messages = [SystemMessage(content=prompt),
                AssistantMessage(content='')
                ]
    queries = llm.ask({"outline":json.dumps(outline, indent=2), "target_section":json.dumps(section_outline, indent=2)}, messages, stop_on_json=True, max_tokens=150)
    if type(queries) is dict:
        print(f'\nquery forsection:\n{section_outline}\nqueries:\n{json.dumps(queries, indent=2)}')
    else:
        print(f'\nquery forsection:\n{section_outline}\nqueries:\n{queries}')
    return queries

def generate_s2_search_queries(query):
    information_requirements = script.sufficient_response(query)
    prompt = """
Generate a set of short Semantic Scholar (s2) search queries, in JSON format, given the following set of information requirements withing the overall topic of:

{{$query}}

<InformationNeeds>
{{$needs}}
</InformationNeeds>

A query should contain no more than 10 tokens. 
Respond in a plain JSON format without any Markdown or code block formatting,  using the following format:

{"information-need1":'s2 query text 1',"information-need2": '<s2 query text 2>', ...}

Respond ONLY with the JSON above, do not include any commentary or explanatory text.
"""
    messages = [SystemMessage(content=prompt),
                AssistantMessage(content='')
                ]
    queries = llm.ask({"query":query, "needs":information_requirements}, messages, stop_on_json=True, max_tokens=300)
    if type(queries) is dict:
        print(f'\nquery: {query}\ns2 queries:\n{json.dumps(queries, indent=2)}')
    else:
        print(f'\nquery: {query}\ns2 queries:\n{queries}')
    return queries
    
def s2_search (outline, section_outline, sbar=None):
    #
    ### Note - ALL THIS IS DOING IS PRE-LOADING LOCAL LIBRARY!!! Doesn't need to return anything!
    # this call is a recursive call, 'section_outline' is a subsection of 'outline' that is being processed
    #convert below to use new xml outline format with nested sections
    if 'section' in section_outline and xml.findall('<section>', section_outline) is not None and len(xml.findall('<section>', section_outline)) > 0: 
        for subsection in xml.findall('<section>', section_outline):
            s2_search(outline, subsection, sbar)
    else:
        queries = make_search_queries(outline, section_outline, sbar)
        bads = ['(',')',"'",' AND', ' OR'] #tulu sometimes tries to make fancy queries, S2 doesn't like them
        if type(queries) is not dict:
            print(f's2_search query construction failure')
            return None
        for i in range(3):
            if 'query'+str(i) not in queries:
                continue
            query = queries['query'+str(i)]
            for bad in bads:
                query = query.replace(bad, '')
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
        write_report_aux(paper_outline=outline, section_outline=outline, length=length)
        
def write_report_aux(paper_outline=None, section_outline=None, texts=None, length=400, dscp='', 
                     topic='', paper_title='', abstract='', depth=0, parent_section_title='', 
                     parent_section_partial='', heading_1_title='', heading_1_draft = '', num_rewrites=1, resources=None):
    #
    ## need to mod this to handle 'resources' longer than available context
    ## resources OR texts, not both! - if both, assume texts is unpacked from resources
    #
    if depth == 0: 
        n = 0; paper_ids=[] #set section number initially to 0
    if len(paper_title) == 0 and depth == 0 and 'title' in paper_outline:
        paper_title=xml.find('<title>',paper_outline)
    if 'length' in section_outline:
        length = xml.find('<length>',section_outline)
    if 'rewrites' in section_outline:
        num_rewrites = xml.find('<rewrites>',section_outline)

    # subsection dscp is full path dscp descending from root
    subsection_dscp = dscp
    if 'task' in section_outline:
        # overall instruction for this subsection
        subsection_dscp += '\n'+ xml.find('<task>',section_outline)
        
    # subsection topic is local title or dscp 
    subsection_topic = xml.find('<dscp>', section_outline) if 'dscp' in section_outline else xml.find('<title>',section_outline)
    subsection_title = xml.find('<title>',section_outline)
    #print(f"\nWRITE_PAPER section: {topic}")
    if depth == 0 and resources is not None:
        # do this only once
        # remember, synopsis is the only text field in section and actually contains the original text
        texts = [s2.section_from_id(id)['synopsis'] for id in resources[1][0]]
        print(f'write_report_aux depth 0 len {sum(len(text) for text in texts)}, limit {llm.context_size*2} chars')
        if sum(len(text) for text in texts) >  llm.context_size*2: # context is in tokens, len is chars
            # texts are too long for context, do query-specific extract first
            words = int(llm.context_size*2 * ((llm.context_size*2)/sum(len(text) for text in texts)))
            summary_texts = []
            summary_text = ''
            for text in texts:
                if len(summary_text) + len(text) <  (llm.context_size*2):
                    summary_text += '\n'+text
                    continue
                else:
                    summary_text = script.process2(arg1=topic,
                                                   arg2=summary_text,
                                                   instruction=f"""You are a skilled technical writer. 
Compress the content of Text2, retaining all the information relevant to Text1 and as much other technical content as possible. 
Limit your response to {words} words.
Be concise. Your goal is to retain maximum information content.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</END>
""",
                                                   max_tokens = int(words*1.5),
                                                   eos='</END>')

                    summary_texts.append(summary_text)
                    summary_text = text
            if len(summary_text) > 0:
                summary_texts.append(summary_text)
            texts = summary_texts

    if 'section' in section_outline and len(xml.findall('<section>',section_outline)) > 0:
        #
        ### write section intro first draft
        #
        subsection_depth = 1+depth
        num_sections = len(xml.findall('<section>',section_outline))
        subsection_token_length = int(length/len(xml.findall('<section>',section_outline)))
        section = ''
        n=0
        paper_ids = []
        for subsection in xml.findall('<section>',section_outline):
            if depth == 0:
                heading_1_title = xml.find('<title>',subsection)
                heading_1_draft = ''
            print(f"subsection title {xml.find('<title>',subsection)}")
            text, subsection_paper_ids =\
                write_report_aux(paper_outline=paper_outline,
                                 section_outline=subsection,
                                 texts=texts,
                                 length=subsection_token_length,
                                 dscp=subsection_dscp,
                                 topic=subsection_topic,
                                 paper_title=paper_title,
                                 abstract=abstract,
                                 depth=subsection_depth,
                                 parent_section_title=subsection_title,
                                 parent_section_partial=section,
                                 heading_1_title= heading_1_title,
                                 heading_1_draft=heading_1_draft,
                                 num_rewrites=num_rewrites,
                                 resources=resources)
            subsection_text = '\n\n'+'.'*depth+xml.find('<title>',subsection)+'\n'+text
            section += subsection_text
            for paper_id in subsection_paper_ids:
                if paper_id not in paper_ids:
                    paper_ids.append(paper_id)
            heading_1_draft += subsection_text
            if depth==0:
                citations = []
                for paper_id in paper_ids:
                    citations.append(str(paper_id)+': '+str(s2.cite(paper_id)))
                with open(f'section{n}.txt', 'w') as pf:
                    pf.write(section +'\n\nReferences:+\n'+'\n'.join(citations))
                n += 1
            
        if depth != 0:
            return section, paper_ids
        else:
            citations = []
            for paper_id in paper_ids:
                citations.append(str(s2.cite(paper_id)))
            return section +'\n\nReferences:+\n'+'\n\n'.join(citations)
    
    else:
        # no subsections, write this terminal section
        section = '' if 'title' not in section_outline else xml.find('<title>',section_outline)
        print(f'heading_1 {heading_1_title}\npst {parent_section_title}\nsubsection topic {subsection_topic}')
        query = heading_1_title+', '+parent_section_title+' '+subsection_topic
        # below assumes web searching has been done
        if resources is None and texts is None:
            # do local search, texts to use not provided
            texts = []
            text_ids = []
            print(f'** write_report_aux Doing local search ! **')
            papers = s2.search(query, subsection_dscp, char_limit=llm.context_size*2)
            ppr_ids = set()
            for title in papers.keys(): # title is the paper title
                paper = s2.paper_from_title(title)
                if paper is not None:
                    ppr_ids.add(paper['faiss_id'])
                    resources.append([paper['faiss_id'], papers[title]])
                for section_id in papers[title]:
                    texts.append(s2.section_from_id(section_id)['synopsis'])
                    text_ids.append(section_id)
                    # tbd - what to do if search returns more than will fit in context?
                    # for now assume with report section dscp context will hold most important
                    # although note we aren't enumerating in order by search rating.
                    # maybe should prune more in search? Make it an option?
                    if sum(len(text) for text in texts) > llm.context_size*2:
                        break
        else:
            ppr_ids = set(resources[0])
        subsection_token_length = max(400,length) # no less than a paragraph
        print(f"\nWriting: {xml.find('<title>',section_outline)} length {length}")
        draft = rw.write(paper_title, paper_outline, xml.find('<title>',section_outline), '', texts, '', subsection_topic,
                         int(subsection_token_length), parent_section_title, heading_1_title, heading_1_draft)
        print(f'\nFirst Draft:\n{draft}\n')
        if num_rewrites < 1:
            return draft, ppr_ids

        #
        ### Now do rewrites
        #
        ### first collect entities
        #
        keywds = rw.paper_ners(paper_title, paper_outline, texts, text_ids)
        missing_entities = rw.literal_missing_ners(keywds, draft)
        print(f'\n missing entities in initial draft {len(missing_entities)}\n')

        for i in range(num_rewrites):
            if i < num_rewrites-1:
                #add new entities
                draft = rw.add_pp_rewrite(paper_title, paper_outline, xml.find('<title>',section_outline), 
                                          draft, texts, keywds, subsection_topic, int((1.3**(i+1))*subsection_token_length), 
                                          parent_section_title, heading_1_title, heading_1_draft)
            else:
                # refine in final rewrite
                draft = rw.rewrite(paper_title, paper_outline, xml.find('<title>',section_outline), 
                                   draft, texts, keywds, subsection_topic, 2*subsection_token_length, 
                                   parent_section_title, heading_1_title, heading_1_draft)
            missing_entities = rw.literal_missing_ners(keywds, draft)
            print(f'\n missing entities after rewrite {len(missing_entities)} \n')

        
        section = draft
        # make sure we write out top level sections even if they have no subsections!
        if depth==0: # single top-level section with no subsections
            with open(f'section{n}.txt', 'w') as pf:
                pf.write(draft)
            n += 1

    return section, ppr_ids

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
            app = QApplication(sys.argv)
            write_report(app, args.report)
            sys.exit(0)
        else:
            pl = Planner(None, 'local')
            app = QApplication(sys.argv)
            write_report(app, 'conversational recommender literature review')
            print('paper_writer.py -report expects to be called from Owl with topic')
            pass
    except Exception as e:
        traceback.print_exc()
        print(str(e))
