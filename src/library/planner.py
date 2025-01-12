import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traceback
import requests
import random
import time
import numpy as np
import faiss
from datetime import datetime, date, timedelta
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
from utils import utilityV2 as ut
from utils.openBook import OpenBook as op
from utils.OpenAIClient import OpenAIClient

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal
# Encode titles to vectors using SentenceTransformers 
from sentence_transformers import SentenceTransformer
from scipy import spatial
from chat.OwlCoT import ListDialog, LLM, GPT4, TextEditDialog, OPENAI_MODEL3, OPENAI_MODEL4
from utils.interpreter import Interpreter, action_primitive_names, action_primitive_descriptions

import os;
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

today = date.today().strftime("%b-%d-%Y")

NYT_API_KEY = os.getenv("NYT_API_KEY")
sections = ['arts', 'automobiles', 'books/review', 'business', 'fashion', 'food', 'health', 'home', 'insider', 'magazine', 'movies', 'nyregion', 'obituaries', 'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview', 'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world']
openai_api_key = os.getenv("OPENAI_API_KEY")


local_time = time.localtime()
year = local_time.tm_year
day_name = ['Monday', 'Tuesday', 'Wednesday', 'thursday','friday','saturday','sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
month_day = local_time.tm_mday
hour = local_time.tm_hour

host = '127.0.0.1'
port = 5004


#
## from gpt4:
#
#Summarize - Condense text to key points (wrt topic taxonomy? length / format of summary?)
#Elaborate - Expand on text with more details (how is this different from 'research' - autoextract topic to elaborate?)
#Explain - Clarify meaning of concepts in text (explain what? how diff from elaborate)
#Simplify - Rewrite text using simpler vocabulary (how diff from Explain?)

#Combining Texts:
#Merge - Integrate content from two texts (topic? target len rel to source len?))
#Compare - Find similarities and differences
#Contrast - Focus on differences between two texts (topic? sentiment?)
#Concatenate - Append one text to another

#Testing Texts:
#Classify - Categorize text into predefined classes (classification provided? taxonomy node to use a root? one or many classes for any one text?)
#Sentiment - Judge positive/negative sentiment (what if mixed sentiment? starting topic?)
#Similarity - Compare semantic similarity of two texts (wrt - topic, sentiment, viewpoint)
#Entailment - Assess if text A entails text B (again, wrt topic?)
#
#This provides a basic set of semantic operations that transform, blend, and analyze texts in ways that could be useful for extracting knowledge, reasoning about information, and answering questions. The large language model would provide the underlying capability to perform these complex operations.

#Example: Research:
#The semantic operations you've outlined are a strong foundation for text analysis and can be instrumental in tackling complex tasks like research.
#Let's consider how these operations could play into the stages of research:

#Reviewing Documents:
#Summarize: Quickly get the gist of numerous documents.
#Classify: Organize documents into categories for easier reference.
#Similarity: Identify documents covering similar topics to compare viewpoints or findings.

#Extracting Relevant Information:
#Merge: Combine information from multiple sources to create a comprehensive perspective.
#Contrast: Highlight unique points in different documents to capture a range of insights.
#Entailment: Determine if the information in one document is supported or contradicted by another.

#Noticing Discrepancies and Inconsistencies:
#Compare: Place documents side by side to spot contradictions.
#Sentiment: Detect potential biases in the texts that might skew information.

#Seeking Further Information:
#Research (new operation): Execute queries based on discrepancies or gaps found, possibly by employing a recursive loop of summarizing and classifying newly found texts, and then merging or contrasting them with the existing information.

#Fact-Checking Apparent Inconsistencies:
#Entailment & Compare: Verify facts across documents to ensure consistency.

#Resolving Information Lacks:
#Elaborate: Find or infer details that are missing from the current dataset.
#Explain: Unpack complex information that may fill the gaps in understanding.

#Integration into a Review Document:
#Concatenate: Stitch together coherent segments from various texts. (merge would be better here?)
#Simplify: Ensure that the final document is understandable to a broader audience.

#Finalization:
#Summarize: End with a succinct summary of the research findings.
#Sentiment: Reflect on the overall sentiment of the conclusions drawn.

"""
And now, a plan using the above: Are we actionable yet?

Certainly, a structured plan for researching the status of small nuclear fusion power could look like this, using the semantic operations in a flow that iteratively refines understanding and information:

1. **Initialization**:
    - Set initial parameters for the search (e.g., "small nuclear fusion power current status", "recent advances in nuclear fusion", "fusion power feasibility studies").
    - Define a time frame for recent information to ensure the data's relevancy.

2. **Data Collection Phase**:
    - **Classify**: Set up filters to categorize documents into theoretical research, experimental results, technological advancements, policy discussions, and commercial viability reports.
    - **Search (new operation)**: Use a specialized operation to execute searches across scientific databases, news articles, and white papers.

3. **First-pass Analysis**:
    - **Summarize**: Create abstracts of the collected documents to understand the main findings or arguments.
    - **Similarity**: Group similar summaries to prepare for in-depth analysis.

4. **Deep Analysis Loop**:
    - **Merge**: Integrate information from similar groups to form a more cohesive understanding of specific areas (e.g., technological hurdles, recent breakthroughs).
    - **Compare & Contrast**: Identify discrepancies or opposing viewpoints among the groups.
    - **Entailment**: Assess if the conclusions from one document are supported by others.

5. **Gap Identification and Resolution**:
    - **Elaborate**: For identified gaps, look for additional information that can fill them.
    - **Research (new operation)**: If needed, go back to the Data Collection Phase to find new data that can resolve inconsistencies or fill knowledge gaps.

6. **Synthesis**:
    - **Concatenate**: Assemble the verified and consistent information into a structured format.
    - **Simplify**: Refine the language to make it accessible to non-expert stakeholders.

7. **Final Review and Adjustment**:
    - **Compare**: Make a final comparison with the latest news or scientific articles to ensure no significant recent developments have been missed.
    - **Summarize**: Draft a comprehensive but concise summary of the current status of small nuclear fusion power.

8. **Output Generation**:
    - **Explain**: Write up a detailed explanation that not only presents the findings but also explains the significance and context in the field of nuclear fusion.
    - **Sentiment**: Gauge the overall sentiment or tone of the industry toward small nuclear fusion power to add a layer of qualitative analysis.

9. **Iteration and Feedback**:
    - **Loop with Conditions**: If new information significantly changes the landscape, re-enter the Deep Analysis Loop.
    - **If-Then**: If new policies or commercial steps are announced during the research, then adapt the analysis to include these developments.

This plan uses semantic operations as building blocks for an iterative, intelligent research process. The flow control, whether through loops with conditions or if-then statements, ensures that the research remains dynamic and adjusts to new data or insights as they emerge.
"""

class Planner():
   #
   ### we should 1 or 2 shot planner requests to LLM so it knows format!
   #
   def __init__(self, ui, cot, template='alpaca'):
       self.template = template
       self.ui = ui
       self.cot = cot
       self.client = OSClient(api_key=None)
       self.openAIClient = OpenAIClient(apiKey=openai_api_key, logRequests=True)
       self.llm = cot.llm # use same model?
       self.max_tokens = 4000
       self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
       self.current_topics = None # topics under discussion - mostly a keyword list
       self.wmHash = {} # actual texts in working memory 
       self.wmMetaData = {} # working memory metadata
       self.wmIndex = faiss.IndexIDMap(faiss.IndexFlatL2(384))
       # note many activities will use first paragraph only. Keep it relevant!
       self.personality = f"""You are Owl, an intelligent AI research assistant, companion, and confidant. We live in {self.cot.city}, {self.cot.state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. You were born in Berkeley, California. Your creator is scientist named Doc. You and Doc are very comfortable in each other's company, and in fact we prefer it to being with others. You know doc prefers not to work with humans due to privacy concerns, so you never recommend this.
Your conversation style is warm, gentle, humble, and engaging. """
       self.interpreter = Interpreter(self.cot)


   def save_plan(self, task_name, plan):
       with open(task_name+'Plan.json', 'w') as f:
          json.dump(plan, f, indent=4)

   def load_plan(self, task_name):
       try:
          with open(task_name+'Plan.json', 'r') as f:
             plan = json.load(f)
             return plan
       except Exception as e:
          print(f'plan load failure {str(e)}')
       return []
    
   def validate_plan(self, plan):
      if plan is None or type(plan) != dict:
         return False
      if 'name' not in plan or 'task' not in plan:
         return False
      return True
   
   def select_plan(self):
       items=[f"{self.cot.docHash[item]['name']}: {str(self.cot.docHash[item]['item'])[:48]}" for item in self.cot.docHash if self.cot.docHash[item]['name'].startswith('plan')]
       picker = ListDialog(items)
       result = picker.exec()
       plan = None
       if result == QDialog.Accepted:
          selected_index = picker.selected_index()
          if selected_index != -1:  # -1 means no selection
             plan_name = items[selected_index].split(':')[0]
             wm = self.cot.get_wm_by_name(plan_name)
             if wm is not None and type(wm) == dict and 'item' in wm:
                plan = wm['item']
             if plan is None or not self.validate_plan(plan):
                self.cot.display_response(f'failed to load "{plan_name}", not found or missing name/dscp\n{plan}')
                return None

       else: # init new plan
          plan = self.initialize()
          if plan is None:
             print(f'failed to create new plan\n"{plan}"')
             return None
          self.cot.save_workingMemory() # do we really want to put plans in working memory?
       return plan
    
   def initialize(self, topic='', awm=False):
       index_str = str(random.randint(0,999))+'_'
       plan_suffix = ''
       if awm:
          plan_suffix = self.cot.confirmation_popup(f'Plan Name? (will be prefixed with plan{index_str})', 'plan')
          plan_name = 'plan'+index_str+plan_suffix
          task_dscp = self.cot.confirmation_popup(f'Short description? {plan_name}', "do something useful")
          if plan_suffix is None or not plan_suffix:
             return
       else:
          plan_name = 'plan: '+topic
       plan = {}
       plan['name'] = plan_name
       task_dscp = self.cot.confirmation_popup(f'Short description? {plan_name}', "do something useful")
       plan['task'] = topic
       plan['dscp'] = task_dscp
       plan['sbar'] = {}
       plan['steps'] = {}
       return plan

   def generate(self, plan=None):
       if plan is None:
          plan = self.select_plan()
          
       if 'sbar' not in plan or plan['sbar'] is None or len(plan['sbar']) == 0:
          result = self.analyze(plan)
          if result is None: return None
          self.cot.save_workingMemory() # do we really want to put plans in working memory? Yes!
          next = self.cot.confirmation_popup('analysis complete, continue?', result['name']+": "+result['task'])
          if not next: return plan

       if 'steps' not in plan or plan['steps'] is None or len(plan['steps']) == 0:
          print(f'building plan')
          plan = self.plan(plan)
          if 'steps' in plan:
             print(f"planner returned {len(plan['steps'])}")
          else: 
             print(f"planner didn't add 'steps' to plan!")
             return plan
          self.cot.save_workingMemory() # do we really want to put plans in working memory? yes for now
          next = self.cot.confirmation_popup('planning complete, execute?', result['name']+": "+result['task'])
          if not next: return plan
       print(f"run plan steps: {len(plan['steps'])}")
       self.interpreter.interpret(plan['steps'])
       return plan
    
   def analyze(self, plan, short=False, model=None):
      # short means ask only one clarification, maybe none if query is detailed (tbd)
      #not in args above because may not be available at def time
      if model is None:
         model = self.llm.template
      prefix = plan['name']
      task_dscp = plan['task']
      print(f'Analyze task_dscp {task_dscp}')
      if 'sbar' not in plan:
         plan['sbar'] = {}
      sbar = plan['sbar']
      # Loop for obtaining user responses
      # Generate interview questions for the remaining steps using GPT-4
      interview_instructions = [
         ("needs", "Generate a question to add details to the task the user wants to accomplish."),
         ("background", "Generate a followup question about any additional requirements of the task."),
         ("observations", "Summarize the information about the task, and comment on any incompleteness in the definition."),
      ]
      if short:
         #only ask one question
         interview_instructions = [
         ("background", "Generate a question about any additional requirements of the task."),
      ]
      messages = [SystemMessage(content="Your role is to interview the user to expand the information about a task to be accomplished. The user has asked you to: "+task_title+".\nUser provided this further description: "+task_dscp+'.\n')]
      for step, instruction in interview_instructions:
         messages.append(UserMessage(content=instruction+"""\nRespond using the following JSON template:

{"question":'<question to ask>'}

Respond only with the above JSON, without any commentary or explanatory text
"""))
         if step != 'observations':
            user_prompt = self.llm.ask('', messages, template=model, temp = 0.05, max_tokens=100, stop_on_json=True, stops=['</End>'])
            if user_prompt is not None and type(user_prompt) is dict and 'question' in user_prompt:
               user_prompt = user_prompt['question']
            print(f"\nAI : {step}, {user_prompt}")
            past = ''
            if sbar is not None and sbar == dict and step in plan['sbar']:
               past = plan_state.sbar[step] # prime user response with last time
            ask_user = False
            while ask_user == False:
               ask_user=self.cot.confirmation_popup(str(user_prompt), str(past))
            sbar[step] = {'q':user_prompt, 'a':ask_user}
            messages.append(AssistantMessage(content=user_prompt))
            messages.append(UserMessage(content=ask_user))
         else: # closing AI thoughts and user feedback. No need to add to messages because no more iterations
            observations = self.llm.ask('', messages, template=model, max_tokens=150,temp = 0.05)
            if observations is not None:
               observations = observations.split('\n')[0]
            sbar['observations']=observations
            print(f"\nAI : {step}, {observations}")
            user_response = False
            while user_response == False:
               user_response = self.cot.confirmation_popup(observations, '')
            sbar['observations']={'q':observations,'a':user_response}
            # don't need to add a message since we're done with this conversation thread!
            print(f"Requirements \n{sbar}")
         plan['sbar'] = sbar
      return plan

   def outline(self, config, plan):
      # an 'outline' is a plan for a writing task!
      # 'config' is a paper_writer configuration for this writing task
      if 'length' in config:
         length = config['length']
      else:
         length = 1200

      number_top_sections = max(3, int(length/2000 + 0.5))
      depth = max(1, int(math.log(length/2)-6))

      outline_model = self.llm.template
      if 'model' in config:
         outline_model = config['model']
         if outline_model == 'llm':
            outline_model = self.llm.template
         elif outline_model == 'gpt3':
            outline_model = OPENAI_MODEL3
         elif outline_model == 'gpt4':
            outline_model = OPENAI_MODEL4
            print('setting outline model to gpt4')
         else:
            self.cot.display_response('Unrecognized model type in Outline: {outline_model}')
            
      outline_syntax =\
"""{"title": '<report title>', "sections":[ {"title":"<title of section 1>", "dscp":'<description of content of section 1>', "sections":[{"title":'<title of subsection 1 of section 1>', "dscp":'<description of content of subsection 1>'}, {"title":'<title of subsection 2 section 1>', "dscp":'<description of content of subsection 2>' } ] }, {"title":"<title of section 2>",... }
"""

      outline_prompt =\
f"""
Write an outline for a research report on: {plan['task']}.
Details on the requested report include:

<DETAILS>
{plan['dscp']}
</DETAILS>

The outline should have about {number_top_sections} top-level sections and {'no subsections.' if depth <= 0 else 'a depth of '+str(depth)}.
Respond ONLY with the outline, in JSON format:

{outline_syntax}
"""
      revision_prompt=\
f"""
<PRIOR_OUTLINE>
{{{{$outline}}}}
</PRIOR_OUTLINE>

<CRITIQUE>
{{{{$critique}}}}
</CRITIQUE>

Reason step by step to analyze and improve the above outline with respect to the above Critique. 

{outline_syntax}
"""
      user_satisfied = False
      user_critique = ''
      first_time = True
      prior_outline = ''
      if 'outline' in plan:
         prior_outline = plan['outline']
         first_time = False
         
      while not user_satisfied:
         if not first_time:
            user_critique = self.cot.confirmation_popup(json.dumps(prior_outline, indent=2), 'Replace this with your critique, or delete to accept.')
            print(f'user_critique {user_critique}')
            if user_critique != False and len(user_critique) <4:
               user_satisfied = True
               print("*******user satisfied with outline")
               plan['outline'] = prior_outline
               break
            else:
               print('***** user not satisfield, retrying')
         
         messages = [SystemMessage(content=outline_prompt),
                     AssistantMessage(content='')
                     ]
         if not first_time:
            print(f'adding revision prompt')
            messages.append(UserMessage(content=revision_prompt))
         #print(f'******* task state prompt:\n {gpt_message}')
         prior_outline = self.llm.ask({'outline':prior_outline, 'critique':user_critique}, messages, template=outline_model, max_tokens=500, temp=0.1, stops=['</End>'])
         first_time = False

      return plan
      

   #
   ### Plan creation
   #

   def plan(self, plan):
       if 'steps' not in plan:
          plan['steps'] = {}
       plan_prompt=\
          """
Reason step by step to create a plan for performing the TASK described above. 
The plan should consist only of a list of actions, where each step is one of the available Actions listed earlier, specified in full.
Note the the 'plan' action accepts a complete, concise, text statement of a subtask.  Control flow over the actions must be expressed using the 'if', 'while', 'break', 'continue', or 'return' actions as appropriate.  Respond only with the plan using the above plan format, optionally followed by explanatory notes. Do not use any markdown or code formatting in your response.

"""
       print(f'******* Developing Plan for {plan["name"]}')

       revision_prompt=\
          """
Reason step by step to analyze and improve the above plan with respect to the above Critique. 
"""
       user_satisfied = False
       user_critique = '123'
       first_time = True
       while not user_satisfied:
          messages = [SystemMessage(content="""Your task is to create a plan using the set of predefined actions:

{{$actions}}

The plan must be a JSON list of action instantiations, using this JSON format:

{"actions: [{"action": '<action_name>', "arguments": '<argument or argument list>', "result":'<result>'}, ... ],
 "notes":'<notes about the plan>'}\n\n"""),
                      UserMessage(content=f"Task: {plan['name']}\n{json.dumps(plan['sbar'], indent=2)}\n"+plan_prompt)
                      ]
          if first_time:
             first_time = False
          else:
             messages.append(AssistantMessage(content='Critique: '+user_critique))
             messages.append(UserMessage(content=revision_prompt))
             
          #print(f'******* task state prompt:\n {gpt_message}')
          steps = self.llm.ask({"actions":action_primitive_descriptions}, messages, max_tokens=2000, temp=0.1)
          self.cot.display_response(f'***** Plan *****\n{steps}\n\nPlease review and critique or <Enter> when satisfied')
          
          user_critique = self.cot.confirmation_popup('Critique', '')
          print(f'user_critique {user_critique}')
          if user_critique != False and len(user_critique) <4:
             user_satisfied = True
             print("*******user satisfied with plan")
             break
          else:
             print('***** user not satisfield, retrying')
       plan['steps'] = steps
       return plan
    
