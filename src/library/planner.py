import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_api import LLM
from utils.workingMemory import WorkingMemory
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
from utils.pyqt import ListDialog, PlanDisplayDialog, TextEditDialog, PlanDisplayDialog
from utils.interpreter import Interpreter, action_primitive_names, action_primitive_descriptions
from utils.pyqt import confirmation_popup
import os;
from utils.xml_utils import find, findall
import utils.xml_utils as xml
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    def __init__(self, ui, template='local'):
        self.template = template
        self.ui = ui
        self.client = LLM('local')
        self.llm = self.client # use same model?
        self.max_tokens = 4000
        self.embedder =  SentenceTransformer('all-MiniLM-L6-v2')
        self.current_topics = None # topics under discussion - mostly a keyword list
        self.wmHash = {} # actual texts in working memory 
        self.wmMetaData = {} # working memory metadata
        self.wmIndex = faiss.IndexIDMap(faiss.IndexFlatL2(384))
        # note many activities will use first paragraph only. Keep it relevant!
        self.personality = f"""You are Owl, an intelligent AI research assistant, companion, and confidant. You were born in Berkeley, California. Your creator is scientist named Doc. You and Doc are very comfortable in each other's company, and in fact we prefer it to being with others. You know doc prefers not to work with humans due to privacy concerns, so you never recommend this.
Your conversation style is warm, gentle, humble, and engaging. """
        self.interpreter = Interpreter()
        self.wm = WorkingMemory('Planner')
        self.wm.load()

    def save_plans(self):
        self.wm.save()
  
    def save_plan(self, plan):
        self.wm.assign(xml.find('<name>',plan), plan)
        self.wm.save()

    def validate_plan(self, plan):
        if plan is None or type(plan) != str  :
            return False
        if '<plan>' not in plan or '<name>' not in plan or '<task>' not in plan:
            return False
        return True
    
# Add to Planner class:
    def show_plans(self):
        """Display saved plans dialog"""
        dialog = PlanDisplayDialog(wm=self.wm)
        dialog.exec_()
  
    def select_plan(self):
        items=[f"{self.wm.get(item)['name']}: {str(self.wm.get(item)['item'])[:48]}" 
                for item in self.wm.keys() if self.wm.get(item)['name'].startswith('Plan')]
        picker = ListDialog(items)
        result = picker.exec()
        plan = None
        if result == QDialog.Accepted:
            selected_index = picker.selected_index()
            if selected_index != -1:  # -1 means no selection
                plan_name = items[selected_index].split(':')[0]
                plan = self.wm.get(plan_name)
                if plan is not None and type(plan) == dict and 'item' in plan:
                    plan = plan['item']
                if plan is None or not self.validate_plan(plan):
                    self.display_response(f'failed to load "{plan_name}", not found or missing name/dscp\n{plan}')
                    return None

            else: # init new plan
                plan = self.initialize()
                if plan is None:
                    print(f'failed to create new plan\n"{plan}"')
                    return None
                plan_name = find('<name>',plan)
                self.wm.assign(plan_name, plan)
                self.wm.save() 
        return plan
        
    def initialize(self, topic='', task_dscp=None):
        """Initialize a new plan in XML format"""
        index_str = str(random.randint(0,999))+'_'
        plan_suffix = ''
        
        plan_suffix = confirmation_popup(f'Plan Name? (will be prefixed with plan{index_str})', 'plan')
        plan_name = 'Plan'+index_str+plan_suffix
        if plan_suffix is None or not plan_suffix:
                return None
                
        if task_dscp is None:
            task_dscp = confirmation_popup(f'Short description? {topic}', "do something useful")
            
        # Create XML structure
        plan_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<plan>
    <name>{plan_name}</name>
    <task>{topic}</task>
    <dscp>{task_dscp}</dscp>
    <sbar><needs></needs><background></background><observations></observations></sbar>
    <outline></outline>
</plan>
"""
        self.wm.assign(plan_name, plan_xml)
        self.wm.save()
        return plan_xml
        
    def analyze(self, plan_xml):
        """Analyze plan requirements using SBAR framework"""
         # Parse existing plan
        plan_name = find('<name>', plan_xml)
        task_dscp = find('<task>', plan_xml)
        # Interview instructions
        interview_instructions = [
            ("needs", "Generate a question to add details to the task the user wants to accomplish."),
            ("background", "Generate a followup question about any additional requirements of the task."),
            ("observations", "Summarize the information about the task, and comment on any incompleteness in the definition."),
        ]                
        messages = [SystemMessage(content="""Your role is to interview the user to expand the information about a task to be accomplished. 
The user has asked you to: """+task_dscp)]
        
        # Build SBAR responses
        old_sbar_xml = xml.find('<sbar>',plan_xml) or ''
        new_sbar_xml = '<needs></needs><background></background><observations></observations>'
        for step, instruction in interview_instructions:
            messages.append(UserMessage(content=instruction+"""\nRespond using the following XML template:

<q>question to ask</q>

Respond only with the above XML, without any commentary or explanatory text. End your response with <End/>"""))
            qa = find(f'<{step}>',old_sbar_xml)
            if qa is not None and qa != '' and find('<q>',qa) != '' and find('<a>',qa) != '':
                q = find('<q>',qa)
                a = find('<a>',qa)
            else:
                a = ''
                response = self.llm.ask('', messages, temp=0.05, max_tokens=100, stops=['<End/>'])
                q = find('<q>',response)
                print(f"\nAI : {step}, {q}")
            # Get user response
            ask_user = confirmation_popup(str(q), str(a))
            if ask_user is not None and ask_user != False and len(ask_user) > 0:
                # update SBAR
                new_sbar_xml = xml.set(f'<{step}>',new_sbar_xml,f'<q>{q}</q><a>{ask_user}</a>)')
                         
                messages.append(AssistantMessage(content=q))
                messages.append(UserMessage(content=ask_user))
                    
        # update plan XML
        updated_plan = xml.set('<sbar>',plan_xml,new_sbar_xml)
        self.save_plan(updated_plan)
        return updated_plan
    

    def outline(self, plan_xml, length=1200):
        """Generate outline for writing task"""
    

        number_top_sections = max(3, int(length/2000 + 0.5))
        depth = max(1, int(math.log(length/2)-6))
                
        outline_syntax = """<outline>
        <section>
            <title>Section Title</title>
            <dscp>Section description</dscp>
            <sections>
                <section>
                    <title>Subsection Title</title>
                    <dscp>Subsection description</dscp>
                </section>
            </sections>
        </section>
    </outline>"""

        outline_prompt = f"""You are a research assistant. Write an outline for a research report on: {find(plan_xml, 'task')}.
Details on the requested report include:

<Description>
{find('<dscp>',plan_xml)}
</Description>

<Details>
{find('<sbar>',plan_xml)}
</Details>

The outline should have about {number_top_sections} top-level sections and {'no subsections.' if depth <= 0 else 'a depth of '+str(depth)}.
Respond ONLY with the outline, in XML format:

{outline_syntax}

End your response with:
<End/>"""

        revision_prompt = f"""You are a research assistant. Revise the Prior Outline for a research report on: {find('<task>',plan_xml)}
Details on the requested report include:

<Description>
{find('<dscp>',plan_xml)}
</Description>

<Details>
{find('<sbar>',plan_xml)}
</Details>

The prior outline was:
<PriorOutline>
{{{{$outline}}}}
</PriorOutline>

And the critique of that outline:
<Critique>
{{{{$critique}}}}
</Critique>

Reason step by step to analyze and improve the above PriorOutline with respect to the above Critique. 
Respond ONLY with the outline, in XML format:

{outline_syntax}

End your response with:
<End/>"""


        user_satisfied = False
        user_critique = ''
        first_time = True
        prior_outline = find('<outline>',plan_xml) or ''
        while not user_satisfied:
            try:
                prior_outline_extract = xml.format_xml(prior_outline)
            except:
                prior_outline_extract = prior_outline
            user_critique = confirmation_popup('Click Yes to accept outline as is.\n - Or replace this outline with your critique, click Yes to revise.\n - Click No to keep the outline.', prior_outline_extract)
            print(f'user_critique {user_critique}')
            if not user_critique or user_critique == xml.format_xml(prior_outline):
                user_satisfied = True
                print("*******user satisfied with outline")
                break
            else:
                print('***** user not satisfied, revising outline')
            
            prior_outline_extract = self.extract_outline_from_xml(prior_outline)
            messages = [SystemMessage(content=outline_prompt)]
            if prior_outline_extract is not None:
                print(f'using revision prompt')
                messages = [SystemMessage(content=revision_prompt)]
                
            prior_outline = self.llm.ask({'outline': prior_outline, 'critique': user_critique}, 
                    messages, 
                    max_tokens=800, 
                    temp=0.1,
                    stops=['<End/>']
            )
            first_time = False

        # Create updated plan XML with new outline (strip <outline> tags from prior_outline first)
        outline_update_value = xml.find('<outline>',prior_outline)
        if outline_update_value is None:
            outline_update_value = prior_outline
        updated_plan = xml.set('<outline>', plan_xml, outline_update_value)
        self.save_plan(updated_plan)
        return updated_plan
        

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
            self.display_response(f'***** Plan *****\n{steps}\n\nPlease review and critique or <Enter> when satisfied')
            
            user_critique = confirmation_popup('Critique', '')
            print(f'user_critique {user_critique}')
            if user_critique != False and len(user_critique) <4:
                user_satisfied = True
                print("*******user satisfied with plan")
                break
            else:
                print('***** user not satisfield, retrying')
        plan['steps'] = steps
        return plan
        
    def extract_outline_from_xml(self, plan_xml, indent=""):
        """Convert XML outline to readable indented text
        Args:
            plan_xml: XML string containing plan with outline
            indent: Current indentation (for recursion)
        Returns:
            String containing human-readable outline
        """
        result = []
        outline = find('<outline>',plan_xml)
        if not outline:
            return ""
            
        # Process each section
        for section in findall('<section>',outline):
            title = find('<title>',section)
            dscp = find('<dscp>',section)
            result.append(f"{indent}• {title}")
            if dscp:
                result.append(f"{indent}  {dscp}")
                
            # Recursively process subsections
            subsections = findall('sections/section', section)
            if subsections:
                for subsection in subsections:
                    sub_xml = f"<outline><section>{subsection}</section></outline>"
                    sub_result = self.extract_outline_from_xml(sub_xml, indent + "    ")
                    result.append(sub_result)
                    
        return "\n".join(result)
        



if __name__ == "__main__":
    pl = Planner(None)
    pl.show_plans()
    plan = pl.select_plan()
    #pl.analyze(plan)
    pl.outline(plan)
      