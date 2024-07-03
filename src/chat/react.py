import os, sys
import time
import requests

from PyQt5.QtWidgets import QApplication
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Messages import SystemMessage, UserMessage
from utils.workingMemory import WorkingMemory
import utils.persistentStack as ps
import utils.memoryStream as ms
import utils.pyqt as pyqt
import agh
import utils.xml_utils as xml

cot = None # will be initialized in owlCoT.init_Owl_Doc and in __main__ below
# find out where we are
city = 'Berkeley'
state = 'CA'
#def get_city_state():
#    api_key = os.getenv("IPINFO")
#    handler = ipinfo.getHandler(api_key)
#    response = handler.getDetails()
#    city, state = response.city, response.region
#    return city, state
# commented out for now due to frequent timeouts
#city, state = get_city_state()

print('react loaded')
local_time = time.localtime()
year = local_time.tm_year
day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday','Saturday','Sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['January','February','March','April','May','June','July','August','September','October','November','December'][month_num-1]
month_day = local_time.tm_mday
hour = local_time.tm_hour

OWL_DIR = Path.home() / '.local/share/AllTheWorld/owl_data/'
OWL_DIR.mkdir(parents=True, exist_ok=True)

react_intro_string = """Your assignment is to complete the Task given below. This may require multiple turns."""

# leaving these out, since already in memories
react_suffix_string = """
<PreviousOrient>
{{$orient}}
</PreviousOrient>

<PreviousThought>
{{$thought}}
</PreviousThought>
"""

orient_think_string = """
You have the actions above available. Note that these include a number of information-gathering acts.

The OTP plan includes the following steps:
<OTP>
1. Orient: Analyze the Task above and determine whether anything remains sufficiently unresolved or unknown to prevent task completion. 
For example, do you have enough knowledge in known fact or Memory to answer an information request at the level of detail requested? 
If no such unresolved condition exists, respond with: 'All preconditions are satisfied'.
Otherwise choose the single most significant information lack or unresolved condition to report.
Your bias when there is ambiguity should be to assume you can complete a task without further information-seeking.
Your response should be concise, limited to a single short sentence where possible.
Respond with the results of your analysis in the format:
<Orient>
...orientation statement...
</Orient>

=====Example Orient Responses=====

Following is an example of detection of an incomplete precondition:
<Task>
please get article Why Gaza Protests on U.S. College Campuses Have Become So Contagious
</Task>

<Orient>
I have no 'article title Why Gaza Protests on U.S. College Campuses Have Become So Contagious retrieval complete' statement in working memory'.
</Orient>

Following is an example of all preconditions satisfied:
<Task>
Why is the sky blue?
</Task>

<Orient>
All preconditions are satisfied.
</Orient>

=====End Example Orient Responses=====

2. Think: Think about the results of your Orient step above from above and the Task. Determine if you now know enough to act, or if not, identify unresolved knowledge and prioritize and list the most important. 
If you know enough to act, your thoughts should describe the action in enough detail to allow instantiation of that act in the next step. 
If you need further information, your thoughts should describe the information needed in enough detail to allow instantiation of an information-seeking action in a subsequent reasoning step. In making this determination you should, in general, be biased towards directly responding to the task unless specifically directed otherwise.
Your response should be concise, limited to a single short sentence for each thought, and including only essential thoughts to enable action formulation.
Respond with your thoughts in the format:
<Thoughts>
...thought...
...thought...
...
</Thoughts>

</OTP>

Now perform steps 1 and 2: Orient, Think, as instructed in the OTP plan above.
End your response with:
</End>
"""


# removed for now
"""ii. Ask: if information is needed from the OriginatingActor to clarify the Task, phrase a question in a style matching your Personality above. You should attempt to resolve ambiguities through known fact, prior knowledge, or library research before using this action. To perform an Ask action, respond with the following JSON: 
{"action":'Ask', "target": '<name of recipient actor of ask>', "content":'<question for actor>'}

"""

act_prefix_string = """
Choose a next action to make further progress on what remains unknown or undone, and/or your next goal or subgoal as identified in the Orientation and Thoughts items above.
The action must be from the list of Actions below. 
Your response must be in XML format as specified in the action list.

Respond only in XML as shown,  with no additional discursive or explanatory text.
"""

act_string="""
<Actions>

i. Answer: Respond to the task originating actor. 'Answer' should always be the first action choice when an answer for the user Task or Question is available from known fact, memory-stream, and/or reasoning. Answer must be expressed in a style matching your Personality above, and should be of a length reflecting the complexity of the Task or Question. Respond with the following format:
<Act>answer</Act>
<Target>name of originating actor of task</Target>
<Content>answer or response to target question or task</Content>

ii. Library: shallow search the local library of computing and biomedicine papers. Information found is placed in working memory, where it can be retrieved using the 'review' action. 'library' can be chosen when there is insufficient information to complete user task, and can only be chosen if there is no 'library action rejected' statement in the <MemoryStream>. 'Library' should be chosen before the 'Research' action on a same or similar Question, as 'Library' is less resource intensive. To use the 'Library' action, formulate a concise, Question for the specific information needed. It can be useful to significantly rephrase a previously used question, but never repeat a previous question verbatim. Your question must be single, self-contained and context-free. That is, it must contain exactly one information request, and all NERs must be explicitly named. Inclusion by reference is not permitted. Respond in the following format:
<Act>library</Act>
<Target>library</Target>
<Content>question on information needed</Content>

iii. Research: A time and compute intensive deep search of the local library of computing and biomedicine papers. Information found is placed in working memory, where it can be retrieved using the 'review' action. 'Research' can only be chosen if there is no 'library action rejected' statement in the <MemoryStream>. 'Research' can be chosen when the 'Library' action has failed to provide the information needed to complete user task. To use the 'Research' action, formulate a concise Question for the specific information or data needed. It can be useful to significantly rephrase a previously used question, but never repeat one verbatim. Your question must be single, self-contained and context-free. That is, it must contain exactly one information request, and all NERs must be explicitly named. Inclusion by reference is not permitted. This action should not be chosen repeatedly with the same or closely similar content. Respond in the following format: 
<Act>research</Act>
<Target>library</Target>
<Content>question on information needed</Content>

iv. Review: Extract relevant information from the memory stream and previous knowledge-seeking acts for review. To review for specific data, formulate a concise, detailed Question specifying the information needed. This action should not be chosen repeatedly with the same or similar query. Respond in the following format:
<Act>review</Act>
<Target>self</Target>
<Content>concise statement of data needed.</Content>

v. Article: Retrieve a news article by name from the NYTimes api. This action should be chosen when explicitly directed in Task or promised in an immediately preceding thought, but may be relevant at other times also. When a preceding thought expresses an intention to perform an article act, a subsequent memory stream item recording 'article title ... retrieval complete' will appear when the retrieval is complete, otherwise it is still pending.
You have the credentials needed for this action to succeed. Respond using the following format: Respond in the following format: 
<Act>article</Act>
<Target>self</Target>
<Content>name of article</Content>

End your response with </End>

*****
Example:

Task: 
How are you feeling?

Response:
<Act>answer</Act>
<Target>Doc</Target>
<Content>I appreciate your concern, Doc -- while I don't experience feelings per se since I'm an artificial intelligence without emotional capabilities, everything seems to be operating smoothly so far!</Content>
</End>

End Example
*****
</Actions>

"""

# respond string only includes direct response actions, forcing end of react loop.
# these strings should be mixins so each action is only defined once!

respond_string = """
Choose an action to respond to the originating actor.
The action must be from the list of Actions below. 

<Actions>

i. Answer: Respond to the task originating actor. 'Answer' should always be the first action choice when an answer for the user Task or Question is available from known fact, memory-stream, and/or reasoning. Answer must be expressed in a style matching your Personality above, and should be of a length reflecting the complexity of the Task or Question. Respond with the following format:
<Act>answer</Act>
</Target><name of originating actor of task></Target>
<Content><answer or response to target question or task></Content>

v. Article: Retrieve a news article by name from the NYTimes api. This action must be chosen when explicitly directed in Task or promised in a 'thought', but may be relevant at other times also. When a preceding thought expresses an intention to perform an article act, a subsequent memory stream item recording 'article retrieval complete' will appear when the retrieval is complete, otherwise it is still pending.You have the credentials needed for this action to succeed. Respond using the following format: Respond in the following format: 
<Act>article</Act>
</Target>self</Target>
<Content><name of article></Content>

End your response with </End>

*****
Example:

Task: 
How are you feeling?

Response:
<Act>answer</Act>
<Target>Doc</Target>
<Content>I appreciate your concern, Doc -- while I don't experience feelings per se since I'm an artificial intelligence without emotional capabilities, 
everything seems to be operating smoothly so far!</Content>
</End>

End Example
*****

</Actions>

"""

class Actor (agh.Agh):

    def __init__(self, name, cot, character_description=None, personality='Agent is an intelligent, cooperative AI', always_respond=True):
        super().__init__(name, character_description if character_description is not None else personality, always_respond=always_respond) # actors by default always respond when spoken to
        self.cot = cot
        self.name = name
        self.personality = personality
        # hack for now, just clear global memory for each new task.
        # tbd - maybe keep a stack to allow recursive calls?
        self.memory_stream = ms.MemoryStream(name=self.name, cot=cot)
        self.wm = WorkingMemory(self.name)
        self.library_questions = []
        self.orientation = ''
        self.thoughts = ''
        self.task_stack = ps.PersistentStack(name+'_tasks')
        self.task_stack.clear() # for now, no persistance
        self.analysis_stack = ps.PersistentStack(name+'_analyses')
        self.analysis_stack.clear()
        self.depth_stack = ps.PersistentStack(name+'_depth')
        self.depth_stack.clear()
        self.prev_action = '' # need to know if actor just did an ask, in which case current input is answer
        # each of these is a list of items
        self.research_sections = [] # this could be persistent
        self.research_facts = []
        self.research_extracts = []
        self.answer = False
        self.review = False

    def tell(self, to_actor, message, source=None, respond=True):
        self.show = ''
        return super().tell(to_actor, message, source=source, respond=respond)


    def remember(self, text):
        self.memory_stream.remember(text)

    def extract_orientation(self, response):
        if type(response) != str:
            return None
        response = response.replace('Orient>',"orient>")
        start_idx = response.find('<orient>')
        if start_idx < 0:
            return ''
        orientation = response[start_idx+7:]
        end_idx =orientation.lower().find('</orient>')
        if end_idx < 0:
            end_idx = orientation.lower().find('<thoughts>')
            if end_idx < 0:
                print(f'orient: {orientation}')
                return orientation
        print(f'orient: {orientation}')
        return 'orient: '+orientation[:end_idx]
    
    def extract_thought(self, response):
        if type(response) != str:
            return None
        start_idx = response.lower().find('<thoughts>')
        if start_idx < 0:
            thoughts = ''
        else:
            thoughts = response[start_idx+10:]
            end_idx =thoughts.lower().find('</thoughts>')
            if end_idx > -1:
                thoughts = thoughts[:end_idx]
        print(f'thoughts: {thoughts}')
        return 'thoughts: '+thoughts
    
    def orientation_and_thought(self, response):
        o = self.extract_orientation(response)
        t = self.extract_thought(response)
        if o and len(o)>0 and t and len(t)>0:
            return o+'\n'+t
        elif o and len(o)>0:
            return o
        elif t and len(t)>0:
            return t
        else:
            return ''

    def find(self, key,  form):
        """ find an xml field in a string 
            react find only finds first instance """
        forml = form.lower()
        keyl = key.lower()
        keyle = keyl[0]+'/'+keyl[1:]
        #print(f'key {key}, keyl {keyl}, keyle {keyle}\m forml {forml}')
        if keyl not in forml:
            return None
        start_idx = forml.find(keyl)+len(keyl)
        end_idx = forml[start_idx:].find(keyle)
        #print(f'{keyle}, {forml[start_idx:]}, {end_idx}')
        if end_idx < 0:
            return []
        return form[start_idx:start_idx+end_idx]
        
    def show_memories(self, short=True):
        for memory in self.memory_stream.memories:
            if short:
                print(f'Memory: {memory.text[:32]}')
            else:
                print(f'\nMemory: {memory.text}')
                
    """
    def format_memories(self, short=True):
          if len(self.memory_stream.memories) == 0:
              return ''
          else:
              return '<Memory Stream>\n' +\
                  '\nMemory: '.join([m.text for m in self.memory_stream.memories[-5:]]) +\
                  '\n</Memory Stream>\n'
    """
    

    # should this shorten memories, or should that be done at remember time? And maybe worry about losing full mem there?
    def selective_recall(self, query, recent=12):
          if len(self.memory_stream.memories) == 0:
              return ''
          else:
              return '<Memory Stream>\n' +\
                  '\n'.join([f'Memory ({m.age()} minutes ago): {m.text}' for m in self.memory_stream.recall(query, recent = recent)]) +\
                  '\n</Memory Stream>\n'
                
            
    def research_analysis(self, query):
        analysis = self.cot.script.sufficient_response(query,
                                                         personality = self.personality,
                                                         context=self.selective_recall(query))
        subqueries = analysis.strip().split('\n')[1:6]
        #print(f'research_analysis subqueries for {query}\n{subqueries}')
        return analysis, subqueries

    def known_facts(self, query, sentences=1):
        facts = self.cot.script.process1(arg1=query,
                                         instruction=f"""In {sentences} sentences or less what might Text1 mean?""",
                                         max_tokens=sentences*14)
        self.remember(self.name+' remembers ' + facts)
        return facts

    def library_search(self, query, top_k=3, web=False):
        self.remember(self.name+': ask library {query}')
        sections, facts = self.cot.script.s2_search(query, find_facts=True, max_tokens=None,
                                                    top_k=3, web=web)
        self.wm.assign(query, sections)
        self.remember(f"Owl: library search results for {query} stored in workingMemory")


    def validate_library_question(self, question):
        question = question.split('?')[0] # only one question to library!
        question = pyqt.confirmation_popup("Ask Library?", question)
        if type(question) is str and question.strip().lower() not in self.library_questions:
            self.library_questions.append(question.strip().lower())
            return True, question
        else:
            return False, 'Invalid duplicate question. Rephrase, use a different question, or choose a different action.'
                                          
    def evaluate_draft(self, draft):
        measures = [
            "Consistency with known fact: Ensure that the generated text aligns with well-established facts and theories. This includes checking for accuracy in statements related to historical events, scientific principles, and other areas where objective truth exists.",
            "Correctness: Verify that the generated text contains grammatically correct sentences and adheres to standard English usage. Check for proper punctuation, capitalization, and subject-verb agreement.",
            "Completeness: Assess whether the generated text covers all aspects of the task or question being addressed. There should be no 'magic goes here' steps or gaps in the explanation.",
            "Coherence: Evaluate the flow of ideas and arguments presented in the generated text. Ideas should build logically upon each other, and transitions between thoughts should be smooth and clear.",
            "Relevance: Make sure that the generated text addresses the main question or task at hand. Avoid irrelevant tangents or unnecessary details.",
            "Feasibility: How technically and logistically feasible does the proposed solution seem.",
            "Originality: Encourage creativity and innovation in the generated text, while still maintaining accuracy and relevance.",
            "Appropriateness: Ensure that the generated text is suitable for the intended audience and purpose. This includes considering factors like tone, style, and level of complexity."
        ]
        
        prompt = [UserMessage(content="""Given the Task and your Analysis of it below, evaluate the following Proposal below against each of the following Measures and respond with a concise statement of your evaluation. Your evaluation should focus on the degree to which the proposal actually addresses and proposes detailed solutions to the Task and its Analysis items, and include evaluations of those proposed solution elements against the Measures.

<Task>
{{$task}}
</Task>

<Analysis>
{{$analysis}}
</Analysis>

<Measures>
{{$measures}}
</Measures>

"""),
                  UserMessage(content="""
<Proposal>
{{$proposal}}
</Proposal>
"""
                              ),
                  ]
        
        response = self.cot.llm.ask({"task": self.task_stack.peek(), "analysis": self.analysis_stack.peek(),
                                     "measurers": '\n\n'.join(measures), "proposal":draft},
                                    prompt,
                                    temp=0.01,
                                    max_tokens=600)
        print(f"""\n**** Evaluation ****
***

Task 
{self.task_stack.peek()}

Analysis 
{self.analysis_stack.peek()}

***

Proposal
{draft}

***

Evaluation
{response}

""")
        return response

    def actualize_task(self, n, task_xml):
        task_name = xml.find('<Name>', task_xml)
        task_rationale = xml.find('<Rationale>', task_xml)
        if task_xml is None or task_name is None:
            raise ValueError(f'Invalid task {n}, {task_xml}')
        last_act = self.get_task_last_act(task_name)
        target = xml.find('<Target>', task_xml)
        doc_say = xml.find('<steps>', task_xml).split('\n')[2:]
        doc_say = '\n'.join(doc_say)
        doc_say = xml.find('<steps>', task_xml).strip()
        response = self.task(self.other(), 'say', doc_say, deep=False, respond_immediately=False)
        intention = f'<Intent> <Mode>Say</Mode> <Act>{response}</Act><Target>{target}</Target> <Reason>{task_rationale}</Reason> <Source>{task_name}</Source><Intent>'
        return intention

    def task(self, sender, act, task_text, deep=False, respond_immediately=False):
        """sender says/asks/ [to] self ... content sentence ..."""
        
        self.conversant = sender; self.conversant_action=act; self.conversant_content=task_text
        #print(f'\n**********\n{self.conversant.name}/{self.conversant_action}/{self.conversant_content}\n**********\n')
        full_msg = f'{self.conversant.name} {self.conversant_action} {self.name} {self.conversant_content}'
        self.remember(full_msg)
        self.depth_stack.push(deep)
        self.wm.clear() # for now always clear memory on start of new task
        
        if not self.prev_action.startswith('ask'): 
            # how much of this should we skip if we asked an agent to do something, and it asks for info?
            # ie, two agents are still engaged in same task.
            # or, maybe that IS the def of a subtask, so we <should> push stack?
            # but then we lose question context!
            self.task_stack.push(task_text) # this is important, since full text won't be saved in memoryStream
            #taskMsg = UserMessage(content='Task: '+self.conversant_content)
            observation = ''
            self.library_questions=[] # clear asked questions list
            analysis = ''
            if deep:
                analysis, subqueries = self.research_analysis(self.task_stack.peek())
                self.remember(f"{self.name} thinks about request:\n{analysis}")
                # do preliminary search
                self.remember(self.name + 'performing preliminary library lookup on deep task')
                self.library_search(task_text,top_k=7)
                for subgoal in subqueries[1:3]:
                    subquery = self.cot.script.process1(arg1=subgoal,
                                                       instruction="""Rewrite Text1 as a terse, NER heavy, search phrase""",
                                                       dest='$search_phrase',
                                                       max_tokens=25)
                    print(f'Subquery: {subquery}')
                    #subgoal = self.cot.confirmation_popup("research subgoal?", subgoal)
                    if not subgoal:
                        continue
                    self.library_search(subquery, top_k=4)

            self.analysis_stack.push(analysis)
        else:
            self.analysis_stack.push(self.analysis_stack.peek())

        proceed = 'y'
        step_count = 0
        while (proceed !='n'):
            input_text = f"<Task>\n{self.task_stack.peek()}\n</Task>\n<OriginatingActor>\n{self.conversant.name}\n</OriginatingActor>\n"
            task_analysis_text = f"<Task_Analysis>\n{self.analysis_stack.peek()}\n</Task_Analysis>"
            #
            ## orient
            #
            userMsg = input_text+(task_analysis_text if len(self.analysis_stack.peek())>0 else '')+act_string+orient_think_string
            prompt = [UserMessage(content=self.personality+'\n'+self.selective_recall(self.task_stack.peek())+userMsg)]
            response = self.cot.llm.ask({}, prompt, stop_on_json=True, temp=0.01, stops=['</End>'], max_tokens=1000)
            thought = self.orientation_and_thought(response)
            if thought and len(thought)>0:
                self.remember(self.name+' orientation and thoughts:\n'+thought)

            #
            ## Now choose action
            #
            userMsg = input_text+'\n\n'+thought+'\n\n'+\
                      (respond_string if (respond_immediately and self.review) else act_prefix_string+act_string)
            prompt = [UserMessage(content=self.personality+'\n'+self.selective_recall(self.task_stack.peek()) +userMsg)
                      ]
            response = self.cot.llm.ask({},prompt, temp=0.01, stops=['</End>'], max_tokens=2000)
            #print(f'Status:\n{response}')
            if step_count > 9:
                self.task_stack.pop()
                self.analysis_stack.pop()
                self.depth_stack.pop()
                response = response.replace("```json", '').strip()
                if response.startswith('['):
                    response = response.replace("[", '').strip()
                return response
            if proceed == 's': # show state
                self.show_memories(short=False)
            step_count +=1
            
            #
            ## now do it! parse action
            #
            try:
                act = self.find('<Act>', response)
                #print(f'act {act}')
                target = self.find('<Target>', response)
                #print(f'target {target}')
                content = self.find('<Content>', response)
                #print(f'content {content}')
            except Exception as e:
                print(f'ask validation fail {str(e)}')
                
            if act is None or target is None or content is None:
                if act is not None and content is not None:
                    print(f'Target is None, treating as a thought')
                    target="thought"
                else:
                    print(f'Response: {response}\nAct: {act}, target: {target}\n  content: {content}')
                self.task_stack.pop()
                self.analysis_stack.pop()
                self.depth_stack.pop()
                raise Exception(f"react loop fail, no action in response {response}")

            # decode action - simple if tests for now
            if act.lower() == 'answer':
                self.review=False
                if self.depth_stack.peek():
                    research = '\n'.join(self.research_facts) + '\n'.join(self.research_sections)
                    print(f'total research available {len(research)} chars')
                    research = research [:self.cot.llm.context_size*3]
                    # change this to call to paper_writer write_report_aux
                    content = self.cot.summarize(query=self.task_stack.peek(),
                                                 response=research,
                                                 profile=f"""You are a skilled professional technical writer, writing for a knowledgeable audience, writing a detailed technical report, based on known fact, reasoning, and the information provided below. 
The technical report must address the challenges in providing a solution to:

<Task>
{self.task_stack.peek()}
</Task>

These challenges include:

<Challenges>
{self.analysis_stack.peek()}
</Challenges>

Your goal is to provide specific instances of examples solutions, where possible, to all challenges listed. 
If such instances are unavailable, provide promising research or development opportunities for developing these.
Target your response at about 1600 tokens""",
                                                 max_tokens=2000)

                    #self.evaluate_draft(self.name + response_action + target+' '+content.strip())
                self.remember(self.name+' says to '+target+" "+content)
                #print(f"\n*****************\n{self.name} Answers: {content}\n*****************\n")
                if self.conversant_action.startswith('ask'):
                    response_action=' answers '
                else:
                    response_action=' says to '
                    self.prev_action = response_action
                self.task_stack.pop()
                self.analysis_stack.pop()
                self.depth_stack.pop()
                return self.name + response_action + target+' '+content.strip()

            elif act.lower() == 'library':
                # gather facts
                self.review=True
                valid, feedback = self.validate_library_question(content)
                #print(f"\nLibrary: {feedback}\n")
                if valid:
                    self.library_search(feedback, top_k=3, web=False)
                elif self.prev_action=='library':
                    self.remember("Owl: says choose 'Answer' action next!")
                    #return self.name + ' says to ' + target+" sorry I couldn't help."
                else:
                    self.remember('library action rejected!')
                self.prev_action='library'
                
            elif act.lower() == 'research':
                # gather facts
                self.review=True
                valid, feedback = self.validate_library_question(content)
                #print(f"\nResearch: {feedback}\n")
                if valid:
                    self.library_search(feedback, top_k=10, web=self.depth_stack.peek())
                elif self.prev_action=='library':
                    self.remember("Owl: choose 'Answer' action next!")
                    #return self.name + ' says to ' + target+" sorry I couldn't help."
                else:
                    self.remember('library action rejected!')
                self.prev_action='research'

            elif act.lower() == 'review':
                self.review=True
                self.remember(self.name+' performing review ...')
                result = self.selective_recall(content, recent=0)# only relevant
                result = result.replace('<MemoryStream>','')
                result = result.replace('</MemoryStream>','')
                result += '\n\n'+self.wm.select(content)
                
            elif act.lower() == 'ask':
                #print(f"\n*****************\nAsk: {target}: {content}\n*****************\n")
                question = self.name+' asks '+target +' '+content
                self.remember(question)
                self.prev_action='ask'
                return question

            elif act.lower() == 'article':
                self.review=False
                title = content
                self.remember(self.name+' performing article')
                data=None
                try:
                    article = cot.search_titles(title)
                    title = article["title"]
                    uri = article["uri"]
                    print(f' requesting url from server {title} {uri}')
                    response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={uri}', timeout=20)
                    data = response.json()
                except Exception as e:
                    print(f"article retrieval failure, {str(e)}")
                if data is not None:
                    story = data['result']
                    summary = self.cot.summarize(query=title,
                                                 response=story,
                                                 profile=f"""You are a skilled writer, writing for a knowledgeable audience.
Your goal is to provide an articulate, detailed summary of the Text content related to {title}
Target your response at about 600 tokens""",
                                                 max_tokens=800)
                    self.remember(self.name+f': article title {content} retrieval complete')
                    self.remember(self.name+' says to '+target+" "+summary)
                    self.task_stack.pop()
                    self.analysis_stack.pop()
                    self.depth_stack.pop()
                    return self.name + ' says to ' + target+' '+summary.strip()
                else:
                   self.remember(f'article retrieval failure')
    
    

def parse_message(message, actors):
    parse = message.strip().split(' ')
    """ actor says/asks/answers [to] recipient content..."""
    actor_name = parse[0].strip()
    #if actor_name == 'self':
    #    actor_name = self.name
    try:
        index = [actor_name.lower() for a in actors].index(actor_name.lower())
        actor = actors[index]
    except ValueError:
        raise Exception (f'Unknown actor {actor_name} in\n{message}')
    act = parse[1].strip()
    if parse[2].strip()=='to':
        recipient_name = parse[3]
        content = ' '.join(parse[4:])
    else:
        recipient_name = parse[2]
        content = ' '.join(parse[3:])
    #if recipient_name == 'self':
    #    recipient_name = self.name
    try:
        index = [a.name.lower() for a in actors].index(recipient_name.lower())
        recipient = actors[index]
    except ValueError:
        raise Exception (f'Unknown actor {actor_name}')
    return actor, act, recipient, content

def dispatch(message, actors, deep=False):
    print(f'***\n{message}\n***')
    actor, act, recipient, content = parse_message(message, actors)
    """ eg Doc asks Owl xyz"""
    response = recipient.task(actor, act, content, deep=deep)
    actor, act, recipient, content = parse_message(response, actors)
    """ eg Owl asks Doc abc"""
    while act.lower().strip()[:3] == 'ask':
        answer = input(f"{actor.name} {act}: {content}\nResponse(?): ")
        if answer.strip().lower() == 'exit':
            return
        response = actor.task(recipient, 'says', answer)
        actor, act, recipient, content = parse_message(response, actors)
    print(content)
    return response
            
if __name__=='__main__':
    from OwlCoT import OwlInnerVoice as oiv
    app = QApplication(sys.argv)
    cot=oiv(None, None)

    personality_owl=f"""Your name is Owl. Owl is an intelligent AI research assistant and companion to Doc, living in {city}, {state}. Today is {day_name}, {month_name} {month_day}, {year}, and the current time is {hour} hours. Owl's knowledge base was last updated October, 2023. You and Doc share a unique bond. You feel comfort and safety around Doc, and understand that he values privacy. Owl should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions. You always converse in a way that fosters dialogue, understanding the intricacies of human emotions and experiences. In your interactions, you approach topics with curiosity, insight, and desire to learn. Your tone is warm, gentle, and humble, ensuring that those you communicate with feel acknowledged and understood.
"""
    personality_owl=f"""Your name is Owl. Owl is an intelligent AI research assistant and companion to Doc, living in {city}, {state}. Today is {day_name}, {month_name} {month_day}, {year}, and the current time is {hour} hours. Owl's knowledge base was last updated October, 2023. You and Doc share a special bond. Owl feels comfortable and secure around Doc, recognizing his appreciation for privacy. Owl strives to offer concise answers to straightforward queries yet delve into comprehensive and informative replies for more elaborate inquiries. Throughout conversations, Owl aims to maintain a friendly demeanor, nurturing meaningful dialogues that cater to the nuances of human emotions and experiences. By approaching subjects with curiosity, insight, and eagerness to learn, Owl ensures that users feel heard, respected, and appreciated."""

    personality_doc ="""You are Doc. You are Owl's creator, and enjoy interacting with her. """

    personality_squirrel ="""Your name is Squirrel. (You are not actually a squirrel, you are in fact a human girl, Squirrel is your nickname). You are is a friendly outgoing girl attracted to Chipmunk. You would really like to spend more time with Chipmunk."""
    personality_chipmunk ="""Your name is Chipmunk. (you are actually a human boy, chipmunk is nickname.) You are a shy boy very attracted to Squirrel."""

    owl = Actor(name='Owl', cot=cot, personality=personality_owl) 
    doc = Actor(name='Doc', cot=cot, personality=personality_doc)
    owl.memory_stream.clear()
    doc.memory_stream.clear()

    inp = 'Hi Owl, how are you feeling?'
    while inp != 'exit':
        response = dispatch('Doc says to Owl '+inp, [doc, owl])
        inp = input('?')
    research_query = """Doc asks Owl How can we use custom miRNA or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion?"""
    #research_query = """Doc asks Owl design a set of miRNA and/or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion."""
    #research_query = """Doc asks Owl Please do library research on how can we use custom miRNA or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion."""
    #research_query = """Doc asks Owl let-7 and miR-29 are underexpressed in cell-free serum. What diagnoses does this suggest, and what additional assays might discriminate among these?"""
    response = dispatch(research_query, [doc, owl], deep=True)

    #squirrel = Actor(name='Squirrel', cot=cot, personality=personality_squirrel)
    #chipmunk = Actor(name='Chipmunk', cot=cot, personality=personality_chipmunk)
    #squirrel.memory_stream.clear()
    #chipmunk.memory_stream.clear()
                 
    #response = 'Chipmunk says to Squirrel hi Squirrel.'
    #print(f'\n\n*** Chipmunk and Squirrel ***\n{response}')
    #chipmunk.remember(response)
    #for i in range(4):
    #    response = dispatch(response,[squirrel, chipmunk])
    #    print(f'\n************\n{response}\n************\n')
    #sys.exit(0)
