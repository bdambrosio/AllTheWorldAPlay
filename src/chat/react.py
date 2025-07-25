import os, sys
import time
import requests
from datetime import timedelta

from PyQt5.QtWidgets import QApplication
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Messages import SystemMessage, UserMessage
from utils.workingMemory import WorkingMemory
import utils.persistentStack as ps
import sim.memoryStream as ms
import utils.pyqt as pyqt
import sim.agh as agh
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
<previousOrient>
{{$orient}}
</previousOrient>

<previousThought>
{{$thought}}
</previousThought>
"""

orient_think_string = """
You have the actions above available. Note that these include a number of information-gathering acts.

The OTP plan includes the following steps:
<otp>
1. Orient: Analyze the Task above and determine whether anything remains sufficiently unresolved or unknown to prevent task completion. 
For example, do you have enough knowledge in known fact or Memory to answer an information request at the level of detail requested? 
If no such unresolved condition exists, respond with: 'All preconditions are satisfied'.
Otherwise choose the single most significant information lack or unresolved condition to report.
Your bias when there is ambiguity should be to assume you can complete a task without further information-seeking.
Your response should be concise, limited to a single short sentence where possible.
Respond with the results of your analysis in the format:
<orient>
...orientation statement...
</orient>

=====Example Orient Responses=====

Following is an example of detection of an incomplete precondition:
<task>
please get article Why Gaza Protests on U.S. College Campuses Have Become So Contagious
</task>

<orient>
I have no 'article title Why Gaza Protests on U.S. College Campuses Have Become So Contagious retrieval complete' statement in working memory'.
</orient>

Following is an example of all preconditions satisfied:
<task>
Why is the sky blue?
</task>

<orient>
All preconditions are satisfied.
</orient>

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
</thoughts>

<otp>
</otp>

Now perform steps 1 and 2: Orient, Think, as instructed in the OTP plan above.
End your response with:
<end/>
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
<actions>

i. Answer: Respond to the task originating actor. 'Answer' should always be the first action choice when an answer for the user Task or Question is available from known fact, memory-stream, and/or reasoning. Answer must be expressed in a style matching your Personality above, and should be of a length reflecting the complexity of the Task or Question. Respond with the following format:
<act>answer</act>
<target>name of originating actor of task</target>
<content>answer or response to target question or task</content>

ii. Library: search the local library of computing and biomedicine papers. Information found is placed in working memory, where it can be retrieved using the 'review' action. 'library' can be chosen when there is insufficient information to complete user task, and can only be chosen if there is no 'library action rejected' statement in the <MemoryStream>. 'Library' should be chosen before the 'Research' action on a same or similar Question, as 'Library' is less resource intensive. To use the 'Library' action, formulate a concise, Question for the specific information needed. It can be useful to significantly rephrase a previously used question, but never repeat a previous question verbatim. Your question must be single, self-contained and context-free. That is, it must contain exactly one information request, and all NERs must be explicitly named. Inclusion by reference is not permitted. Respond in the following format:
<act>library</act>
<target>library</target>
<content>question on information needed</content>

iii. Google: search the web for information. Information found is placed in working memory, where it can be retrieved using the 'review' action. 'Google' can be chosen when there is insufficient information to complete user task, and can only be chosen if there is no 'google action rejected' statement in the <MemoryStream>. To use the 'Google' action, formulate a concise, Question for the specific information needed. It can be useful to significantly rephrase a previously used question, but never repeat a previous question verbatim. Your question must be single, self-contained and context-free. That is, it must contain exactly one information request, and all NERs must be explicitly named. Inclusion by reference is not permitted. Respond in the following format:
<act>google</act>
<target>google</target>
<content>question on information needed</content>

iv. Review: Extract relevant information from the memory stream and previous knowledge-seeking acts for review. To review for specific data, formulate a concise, detailed Question specifying the information needed. This action should not be chosen repeatedly with the same or similar query. Respond in the following format:
<act>review</act>
<target>self</target>
<content>concise statement of data needed.</content>

v. Article: Retrieve a news article by name from the NYTimes api. This action should be chosen when explicitly directed in Task or promised in an immediately preceding thought, but may be relevant at other times also. When a preceding thought expresses an intention to perform an article act, a subsequent memory stream item recording 'article title ... retrieval complete' will appear when the retrieval is complete, otherwise it is still pending.
You have the credentials needed for this action to succeed. Respond using the following format: Respond in the following format: 
<act>article</act>
<target>self</target>
<content>name of article</content>

End your response with:
</end>
*****
Example:

Task: 
How are you feeling?

Response:
<act>answer</act>
<target>Doc</target>
<content>I appreciate your concern, Doc -- while I don't experience feelings per se since I'm an artificial intelligence without emotional capabilities, everything seems to be operating smoothly so far!</content>
</end>

End Example
*****
</actions>

"""

# respond string only includes direct response actions, forcing end of react loop.
# these strings should be mixins so each action is only defined once!

respond_string = """
Choose an action to respond to the originating actor.
The action must be from the list of Actions below. 

<actions>

i. Answer: Respond to the task originating actor. 'Answer' should always be the first action choice when an answer for the user Task or Question is available from known fact, memory-stream, and/or reasoning. Answer must be expressed in a style matching your Personality above, and should be of a length reflecting the complexity of the Task or Question. Respond with the following format:
<act>answer</act>
<target><name of originating actor of task></target>
<content><answer or response to target question or task></content>

v. Article: Retrieve a news article by name from the NYTimes api. This action must be chosen when explicitly directed in Task or promised in a 'thought', but may be relevant at other times also. When a preceding thought expresses an intention to perform an article act, a subsequent memory stream item recording 'article retrieval complete' will appear when the retrieval is complete, otherwise it is still pending.You have the credentials needed for this action to succeed. Respond using the following format: Respond in the following format: 
<act>article</act>
<target>self</target>
<content><name of article></content>

End your response with </end>

*****
Example:

Task: 
How are you feeling?

Response:
<act>answer</act>
<target>Doc</target>
<content>I appreciate your concern, Doc -- while I don't experience feelings per se since I'm an artificial intelligence without emotional capabilities, 
everything seems to be operating smoothly so far!</content>
</end>

End Example
*****

</actions>

"""

class Actor (agh.Character):

    def __init__(self, name, cot, context, character_description, personality=None, drives=None, always_respond=False):
        # Initialize Agh first
        super().__init__(name, character_description, server_name='local', mapAgent=False)
        
        # React-specific initialization
        self.cot = cot
        self.context = context
        self.personality = personality if personality else character_description
        self.llm = cot.llm
        
        # Task management
        self.analysis_stack = ps.PersistentStack(name+'_analyses')
        self.library_questions = []
        self.wm = WorkingMemory(name=f"{name}_wm")
        
        # Bridge old and new memory systems
        self.memory_stream = ms.MemoryStream(name, cot)
        
        
        # Initialize drives that will guide action selection
        if drives is not None:
            self.set_drives(drives)
        else:
            self.set_drives = [
            "engaging with others: maintaining meaningful dialogue",
            "world-knowledge: gathering and integrating information",
            "self-knowledge: understanding and using past experiences"
        ]

        # Actor's current state (separate from Agh's drive states)
        self.current_state = self._generate_current_state()

    def _generate_current_state(self):
        """Generate Actor's current cognitive state"""
        # Get Agh's drive states
        narrative = self.narrative.get_summary('medium')  # Updates self.state
        ranked_signalClusters = self.driveSignalManager.get_scored_clusters()
        #focus_signalClusters = choice.pick_weighted(ranked_signalClusters, weight=4.5, n=5) if len(ranked_signalClusters) > 0 else []
        focus_signalClusters = [rc[0] for rc in ranked_signalClusters[:3]] # first 3 in score order
        return narrative + '\n'+"\n".join([sc.to_string() for sc in focus_signalClusters])

    def task(self, sender, act, task_text, deep=False, respond_immediately=False):
        """Main task handling using both Agh's cognitive systems and OTP"""
        # Record the interaction and update states
        full_msg = f"{sender.name} {act} {self.name} {task_text}"
        self.add_to_history(full_msg)
        self.current_state = self._generate_current_state()  # Update Actor's state
        self.generate_goals()
        self.focus_goal = self.goals[0]
        self.generate_task_plan()
        
        # Store task context
        self.focus_task.push(self.tasks[0])
        
        # Build context for OTP using cognitive state
        input_text = f"""
<task>
{task_text}
</task>

<cognitiveState>
{self.current_state}
</cognitiveState>

<goals>
{self.goals}
</goals>

<tasks>
{self.tasks}
</tasks>
"""
        
        # Run OTP with cognitive awareness
        prompt = [
            UserMessage(content=self.personality + 
                       self.selective_recall(task_text) +
                       react_intro_string +
                       input_text +
                       orient_think_string)
        ]
        
        # Get orientation and thoughts
        otp_response = self.llm.ask({}, prompt, max_tokens=1000, stops=["</End>"])
        orientation = self.extract_orientation(otp_response)
        thoughts = self.extract_thought(otp_response)
        
        # Record in memory
        self.remember(f"{self.name} orientation: {orientation}")
        self.remember(f"{self.name} thoughts: {thoughts}")
        
        # Use orientation to determine strategy
        if "All preconditions are satisfied" in orientation:
            return self._handle_response_phase(task_text, sender)
        elif "insufficient information" in orientation.lower():
            # Use thoughts to guide research
            self.analysis_stack.push(thoughts)
            return self._handle_research_phase(task_text, sender)
        else:
            return self._handle_review_phase(task_text, sender)

    def _determine_action_strategy(self):
        """Use Agh's cognitive state to determine next action"""
        # Check current state and priorities
        state_text = str(self.current_state).lower()
        priorities = [str(p).lower() for p in self.priorities]
        
        # Look for indicators in state/priorities
        if any("insufficient information" in p for p in priorities) or \
           "uncertainty" in state_text:
            return 'research'
        elif any("review knowledge" in p for p in priorities) or \
             "recall relevant information" in state_text:
            return 'review'
        else:
            return 'respond'

    def tell(self, to_actor, message, source=None, respond=True):
        self.show = ''
        return super().tell(to_actor, message, source=source, respond=respond)


    def remember(self, text):
        """Bridge between memory systems"""
        # Add to structured memory (Agh's system)
        self.add_to_history(text)
        
        # Maintain compatibility with memory_stream for now
        try:
            self.memory_stream.remember(text)
        except Exception as e:
            print(f"Warning: memory_stream storage failed: {e}")

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
                            print(f'Memory: {memory.to_string()[:32]}')
        else:
            print(f'\nMemory: {memory.to_string()}')
                

    # should this shorten memories, or should that be done at remember time? And maybe worry about losing full mem there?
    def selective_recall(self, query, recent=12):
        """Bridge memory retrieval using both concrete and abstract memories"""
        memories = []
        
        # Get concrete recent memories
        try:
            structured_memories = self.memory_retrieval.get_recent_relevant(
                memory=self.structured_memory,
                query=query,
                time_window=timedelta(hours=1),
                threshold=0.5,
                max_memories=recent
            )
            
            for mem in structured_memories:
                memories.append(f"Recent Memory: {mem.to_string()}")
        except Exception as e:
            print(f"Warning: concrete memory retrieval failed: {e}")
        
        # Get abstract memories/patterns
        try:
            abstract_memories = self.memory_consolidator.get_abstractions(
                memory=self.structured_memory,
                query=query,
                threshold=0.5
            )
            
            for mem in abstract_memories:
                memories.append(f"Life Experience: {mem.to_string()}")
        except Exception as e:
            print(f"Warning: abstract memory retrieval failed: {e}")
            
        if not memories:
            return ''
        
        return '<memory_tream>\n' + '\n'.join(memories) + '\n</memory_stream>\n'
                
            
    def research_analysis(self, query):
        """Analyze research needs using Agh's cognitive state"""
        # First use Agh's state to determine knowledge gaps
        self.generate_state()
        state_text = str(self.current_state).lower()
        
        # Then use script for detailed analysis
        analysis = self.cot.script.sufficient_response(
            query,
            personality=self.personality,
            context=f"""
Current cognitive state indicates:
{state_text}

Recent relevant memories:
{self.selective_recall(query)}
"""
        )
        
        # Extract subqueries but filter based on state priorities
        subqueries = analysis.strip().split('\n')[1:6]
        subqueries = [q for q in subqueries if self._query_matches_priorities(q)]
        
        return analysis, subqueries

    def _query_matches_priorities(self, query):
        """Check if query aligns with current priorities"""
        query = query.lower()
        priorities = [str(p).lower() for p in self.priorities]
        
        # Check if query relates to any current priorities
        return any(
            priority_term in query 
            for priority in priorities
            for priority_term in priority.split()
            if len(priority_term) > 4  # Skip short words
        )

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

<task>
{{$task}}
</task>

<analysis>
{{$analysis}}
</analysis>

<measures>
{{$measures}}
</measures>

End your response with:
</end>"""),
                  UserMessage(content="""
<proposal>
{{$proposal}}
</proposal>
"""
                              ),
                  ]
        
        response = self.cot.llm.ask({"task": self.focus_task.peek(), "analysis": self.analysis_stack.peek(),
                                     "measurers": '\n\n'.join(measures), "proposal":draft},
                                    prompt,
                                    temp=0.01,
                                    max_tokens=600, stops=["</end>"])
        print(f"""\n**** Evaluation ****
***

Task 
{self.focus_task.peek()}

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
        # Call parent implementation to get the intention
        intention = super().actualize_task(n, task_xml)
        
        if intention is None:
            return None
            
        # Return the intention directly since parent already formats it correctly
        return intention

    def _handle_research_phase(self, task_text, sender):
        """Handle information gathering using react's research capabilities"""
        # Store task context
        self.task_stack.push(task_text)
        
        # Use Agh's state to determine research strategy
        state_text = str(self.current_state).lower()
        
        # Analyze research needs
        analysis, subqueries = self.research_analysis(task_text)
        self.analysis_stack.push(analysis)
        self.remember(f"{self.name} research analysis:\n{analysis}")
        
        # Choose research method based on state/priorities
        if "academic knowledge needed" in state_text:
            # Use library search
            for query in subqueries[:2]:  # Limit to top 2 most relevant queries
                if self.validate_library_question(query)[0]:
                    self.library_search(query, top_k=3)
                
        elif "current information needed" in state_text:
            # Use web search
            try:
                response = requests.get(f'http://127.0.0.1:5005/search/?query={task_text}')
                data = response.json()
                if data and 'result' in data:
                    self.remember(f"{self.name} found web information: {data['result'][:200]}...")
            except Exception as e:
                print(f"Web search failed: {str(e)}")
            
        # After gathering information, move to review phase
        return self._handle_review_phase(task_text, sender)

    def _handle_review_phase(self, task_text, sender):
        """Review and integrate gathered information"""
        # Get relevant memories including research results
        relevant_info = self.selective_recall(task_text, recent=12)
        
        # Update cognitive state with new information
        self.add_to_history(f"Reviewing information: {task_text}")
        self.generate_state()
        
        # Check if we have sufficient information
        if "insufficient information" not in str(self.current_state).lower():
            return self._handle_response_phase(task_text, sender)
        else:
            # Need more research
            return self._handle_research_phase(task_text, sender)

    def _handle_response_phase(self, task_text, sender):
        """Formulate and send response using react's action system"""
        # Get relevant context
        relevant_info = self.selective_recall(task_text)
        
        # Use OTP thoughts to guide action selection
        prompt = [
            UserMessage(content=f"""
{self.personality}

<task>
{task_text}
</task>

<context>
{relevant_info}
</context>

<current_state>
{self.current_state}
</current_state>

{act_prefix_string}
{act_string}
""")
        ]
        
        # Get action selection in XML format
        action_response = self.llm.ask({}, prompt, max_tokens=1000, stops=["</end>"])
        
        # Extract action components
        act = xml.find('<act>', action_response)
        target = xml.find('<target>', action_response)
        content = xml.find('<content>', action_response)
        
        if act and target and content:
            if act.lower() == 'answer':
                return f"{self.name} says to {target} {content}"
            elif act.lower() in ['library', 'google', 'article', 'review']:
                # Handle information gathering actions
                self.remember(f"{self.name} {act}: {content}")
                return self._execute_action(act.lower(), content, target)
        
        # Fallback response
        return self.tell(sender, "I'm not sure how to respond to that.")

    def _execute_action(self, action_type, content, target):
        """Execute react's action types"""
        if action_type == 'library':
            if self.validate_library_question(content)[0]:
                self.library_search(content)
                return self._handle_review_phase(self.task_stack.peek(), target)
            
        elif action_type == 'google':
            try:
                response = requests.get(f'http://127.0.0.1:5005/search/?query={content}')
                data = response.json()
                if data and 'result' in data:
                    self.remember(f"Found: {data['result'][:200]}...")
                    return f"{self.name} says to {target} {data['result']}"
            except Exception as e:
                print(f"Web search failed: {str(e)}")
            
        elif action_type == 'article':
            # Handle article retrieval
            return self._handle_article_action(content, target)
            
        return self._handle_review_phase(self.focus_task.peek().to_string(), target)

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
