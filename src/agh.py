import json
import traceback
from utils.Messages import UserMessage
import utils.xml_utils as xml
from utils.memoryStream import MemoryStream
def find_first_digit(s):
    for char in s:
        if char.isdigit():
            return char
    return None  # Return None if no digit is found


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)


class Character():
    def __init__(self, name, character_description):
        self.name = name
        self.character = character_description
        self.llm = None # will be init by Worldsim
        self.history = MemoryStream(name, self)
        self.memory = 'None'
        self.context=None
        self.priorities = [''] # will be displayed by main thread at top in character intentions text widget
        self.show='' # to be displayed by main thread in UI public (main) text widget
        self.state = {}
        self.intentions = []
        self.previous_action = ''
        self.reason='' # reason for action
        self.thought='' # thoughts - displayed in character thoughts window
        self.sense_input = ''
        self.widget = None
        self.active_task = Stack() # task character is actively pursuing.
        self.last_acts = {} # a set of priorities for which actions have been started, and their states.
        self.dialog_status = 'Waiting' # Waiting, Pending
        self.new_memory_cnt = 0 # number of memories since last consolidation
    def other(self):
        for actor in self.context.actors:
            if actor != self:
                return actor
    def add_to_history(self, message):
        message = message.replace('\\','')
        self.new_memory_cnt += 1
        self.history.remember(message)

    def initialize(self):
        pass

    def greet(self):
            for actor in self.context.actors:
                # only insert hellos for actors who will run before me.
                if actor == self: 
                    return
                print(f' {self.name} greeting {actor.name}')
                message = f"Hi, I'm {self.name}"
                actor.tell(self, message)
                actor.show += f'\n{self.name}: {message}'
                return

    def see(self):
        for actor in self.context.actors:
            if actor != self:
                self.add_to_history(f'You see {actor.name}')

    def format_history(self, n=2):
        memories = [memory.text for memory in self.history.n_most_recent(n)]
        return '\n\n'.join(memories)


    def say_target(self, text):
        prompt=[UserMessage(content="""Determine the intended hearer of the following message spoken by you.
{{$character}}

You're recent history has been:

<History>
{{$history}}
</History>
        
Known other actors include:
        
<Actors>
{{$actors}}
</Actors>
        
The message is:
        
<Message>
{{$message}}
</Message>
        
The recipient may be:
- explicitly mentioned by name in the message, e.g. "Samantha, let's rest."
- appear in the message as a pronoun reference to an actor named explicitly in History, e.g. "You go that way/"
- may not appear at all in the message, as in conversational convention, but be inferrable from History e.g. "Ok, let's go"
- may not be an actor named in the list of known actors.

Respond using the following XML format:

<Target>
  <Name>intended recipient name</Name>
</Target>

Respond only with the above XML, instantiated with the name of the intended recipient
Do not add any introductory, explanatory, or discursive matter.
End your response with:
</END>
""")]
        response = self.llm.ask({'character':self.character, 'history':self.format_history(),
                                 'actors':'\n'.join([actor.name for actor in self.context.actors]),
                                "message":text
                                },
                               prompt, temp=0.2, stops=['END'], max_tokens=180)

        print(f'say target name: {response}')
        return xml.find('<Name>', response)

class Agh(Character):
    def __init__ (self, name, character_description):
        super().__init__(name, character_description)
        self.previous_action = ''
        self.sense_input = ''
        self.drives = [
            "immediate physiological needs: survival, water, food, clothing, shelter, rest.",
            "safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events.",
            "assurance of short-term future physiological needs (e.g. adequate water and food supplies, shelter maintenance).",
            "love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friendship, intimacy, trust, acceptance."
            ]

        self.act_result = ''
        self.last_acts = {} # a set of priorities for which actions have been started, and their states.
        # Waiting - Waiting for input, InputPending, OutputPending - say intention pending
        self.dialog_status = 'Waiting' # Waiting, Pending
        self.dialog_length = 0 # stop dialogs in tell after a few turns

    def save(self, filepath):
        allowed_types = (int, float, str, list, dict)  # Allowed types for serialization
        filtered_data = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, allowed_types):
                # Handle lists of strings specifically
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    filtered_data[attr] = value
                elif isinstance(value, dict):
                    filtered_data[attr] = {k: v for k, v in value.items() if isinstance(v, allowed_types)}
                else:
                    filtered_data[attr] = value
        with open(filepath, 'w') as fp:
            json.dump(filtered_data, fp, indent=4)

    def load(self, filepath):
        try:
            filename = self.name + '.json'
            with open(filepath / filename, 'r') as jf:
                data = json.load(jf)
                for attr, value in data.items():
                    setattr(self, attr, value)
        except Exception as e:
            print(f' error restoring {self.name}, {str(e)}')

    def generate_state(self):
        """ generate a state to track, derived from basic drives """
        prompt=[UserMessage(content="""A basic Drive is provided below. 
Your task is to create an instantiated state to track your current state wrt this drive.
Create this instantiated state description given the basic Drive, current Situation, your Character, Memory, and recent History as given below.

<Drive>
{{$drive}}
</Drive>

<Situation>
{{$situation}}
</Situation>

<Character>
{{$character}}
</Character>

<Memory>
{{$memory}}
</Memory>

<History>
{{$history}}
</History>

Respond using this XML format:

<State> 
  <Term>term designating the above drive</Term>
   <Assessment>a one or two word value, e.g. high, medium-high, medium, medium-low, low, etc.</Assessment>
   <Trigger>a concise, few word statement of the specific situational trigger for this drive instantiation</Trigger> 
   <Termination>testable condition for drive satisfaction</Termination>
</State>

The 'Term' must contain at most three words concisely summarizing the goal of this state element, for example 'Identify stranger's intention'.

The 'Assessment' is one word describing the current distance from goal and/or it's urgency. eg, 'high' indicates we are far from the goal or it is highly important to make progress on this drive. Reason about factors like the position of the related drive in the Drive list (higher is more important), the Character and how they might react to the trigger (below), and how unique the Trigger is as an event. Look for reasons to given an assessment other than medium. Use nuanced assessments (very high, medium-low, etc) where appropriate. 

The 'Trigger' is a concise phrase or sentence designating the Character, Situation, Memory, or History element(s) that trigger this instantion.

The 'Termination' is a phrase that will be checked against History or action consequences to see if the drive is satisfied and should be terminated.

Respond with exactly one instance of the <State> XML form above. Respond ONLY with the above XML.

Do on include any introductory, explanatory, or discursive text.
End your response with:
<END>
"""
                            )]
        self.state = {}
        for drive in self.drives:
            print(f'{self.name} generating state for drive: {drive}')
            response = self.llm.ask({"drive":drive, "situation":self.context.current_state,
                                     "memory":self.memory,
                                     "character":self.character, "history":self.format_history(4)},
                                    prompt, temp=0.3, stops=['<END>'], max_tokens=200)
            term = xml.find('<Term>', response)
            assessment = xml.find('<Assessment>', response)
            trigger=xml.find('<Trigger>', response)
            termination_check=xml.find('<Termination>', response)
            # these will be used for remainder of scenario
            self.state[term] ={"drive":drive, "state": assessment, 'trigger':trigger, 'termination_check': termination_check}
        print(f'{self.name}initial state {self.state}')

    def test_state_termination(self, state, consequences, updates=''):
        """ generate a state to track, derived from basic drives """
        prompt = [UserMessage(content="""An instantiated state for a basic drive is provided below 
Your task is to test whether the instantiated state for this Drive has been satisfied as a result of recent events.

<State>
{{$state}}
</State>

<Situation>
{{$situation}}
</Situation>

<Character>
{{$character}}
</Character>

<History>
{{$history}}
</History>
 
<Events>
{{$events}}
</Events>

<Termination_check>
{{$termination_check}}
</Termination_check>

Respond using this XML format:

<Termination> 
    <Level>value of state satisfaction, True if termination test is met in Events or History</Term>
</Termination>

the 'Level' above should be True if the termination test is met in Events or recent History, and False otherwise.  

Respond ONLY with the above XML
Do on include any introductory, explanatory, or discursive text.
End your response with:
<END>
"""
                                  )]

        response = self.llm.ask({"drive": state, "situation": self.context.current_state,
                                "memory": self.memory,
                                "character": self.character, "history": self.format_history()},
                                prompt, temp=0.3, stops=['<END>', '</Termination>'], max_tokens=60)
        satisfied = xml.find('<Level>', response)
        if satisfied or satisfied.lower().strip() == 'true':
            print(f'State {state} Satisfied!')
        return satisfied

    def map_state(self):
        """ map state for llm input """
        mapped = []
        for key, item in self.state.items():
            trigger = item['drive']
            value = item['state']
            mapped.append(f"- '{key}: {trigger}', State: '{value}'")
        return "A 'State' of 'High' means the task is important or urgent\n"+'\n'.join(mapped)

    def update_state(self):
        """ update state """
        prompt = [UserMessage(content=self.character+"""{{$character}}
When last updated, your state was:

<State>
{{$state}}
</State>

Your task is to update your state.
Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your long-term memory includes:

<Memory>
{{$memory}}
</Memory>

Recent interactions not included in memory:

<RecentHistory>
{{$history}}
</RecentHistory)>

Respond with an updated state, using this XML format, 

The updated physical state should reflect updates from the previous state based on passage of time, recent events in Memory, and recent history

<UpdatedState>
{{$stateTemplate}}
</UpdatedState>

Respond ONLY with the updated state.
Do not include any introductory or peripheral text.
limit your response to 120 words at most.
End your response with:
<STOP>""")
                  ]
        mapped_state = self.map_state()
        template = self.make_state_template()
        response = self.llm.ask({'character': self.character, 'memory':self.memory,
                                 'history': self.format_history(8),
                                 "situation": self.context.current_state,
                                 "state": mapped_state, "template": template
                                },
                               prompt, temp=0.2, stops=['<STOP>'], max_tokens=180)
        state_xml = xml.find('<UpdatedState>', response)
        print(f'\n{self.name} updating state')
        for key, item in self.state.items():
            update = xml.find(f'<{key}>', state_xml)
            if update != None:
                print(f'  setting {key} to {update}')
                item["state"] = update

    def make_state_template(self):
        """ map state for llm input """
        mapped = []
        for key, item in self.state.items():
            dscp = item['drive']
            value = 'updated state assessment'
            mapped.append(f'<{key}>{value}</{key}>')
        return '\n'.join(mapped)

    def initialize(self):
        """called from worldsim once everything is set up"""
        self.generate_state()
        self.update_priorities()

    def synonym_check(self, term, candidate):
        """ except for new tasks, we'll always have correct task_name, and new tasks are presumably new"""
        if term == candidate: return True
        else: return False
        instruction=[UserMessage(content="""Your task is to decide if Text1 and Text2 designate the same task.
Reason step-by-step:
 - Does phrase Text2 have the same meaning as Text1? That is, does it designate essentially the same task, or is one a refinement of the other?

===Examples===
Text1: Communicate with Joe.
Text2: Find water.
Response: False
--- 
Text1: Communicate with Joe.
Text2: Tell Joe how much I appreciate having him around
Response: True
---
Text1: Check disposal noise
Text2: Maintain supply stocks
Response: False
---
Text1: Maintain communication with Madam
Text2: Check disposal noise
Response: False
===END Examples===

Text1: {{$text1}}
Text2: {{$text2}}

Respond True or False. 
Do not include your reasoning in your response.
Do not include any introductory, discursive, or explanatory text.
Simply respond 'True' or 'False'.
End your response with:
</END>
""")]
        response = self.llm.ask({"text1":term, "text2":candidate}, instruction, temp=0.3, stops=['</END>'], max_tokens=100)
        if 'true' in response.lower():
            return True
        else:
            return False

    def get_task_last_acts_key(self, term):
        """ checks for name in Last_acts!"""
        for task in list(self.last_acts.keys()):
            match=self.synonym_check(task, term)
            if match:
                return task
        return None
            
    def set_task_last_act(self, term, act):
        # pbly don't need this, at set we have canonical task
        task = self.get_task_last_acts_key(term)
        if task == None:
            print(f'SET_TASK_LAST_ACT {self.name} no match found for term: {term}, {act}')
            self.last_acts[term] = act
        else:
            print(f'SET_TASK_LAST_ACT {self.name} match found: term {term}, task {task}\n  {act}')
            self.last_acts[task] = act

    def get_task_last_act(self, term):
        task = self.get_task_last_acts_key(term)
        if task == None:
            #print(f'GET_TASK_LAST_ACT {self.name} no match found: term {term}')
            return 'None'
        else:
            #print(f'GET_TASK_LAST_ACT match found {self.name} term {term} task {task}\n  last_act:{self.last_acts[task]}\n')
            return self.last_acts[task]

    def make_task_name(self, reason):
        instruction=[UserMessage(content="""Generate a concise, 2-5 word task name from the motivation to act provided below.
Respond using this XML format:

<Name>task-name</Name>

===Examples===

===End Examples===

<Motivation>
{{$reason}}
</Motivation>

Respond only with your task-name using the above XML
Do not include your reasoning in your response.
Do not include any introductory, discursive, or explanatory text.
End your response with:
</END>
""")]
        response = self.llm.ask({"reason":reason}, instruction, temp=0.3, stops=['</END>'], max_tokens=12)
        return xml.find('<Motivation', response)
                    
    def get_task_xml(self, task_name):
        for candidate in self.priorities:
            #print(f'find_or_make testing\n {candidate}\nfor name {task_name}')
            if task_name == xml.find('<Text>', candidate):
                print(f'found existing task\n  {task_name}')
                return candidate
        return None
    
    def find_or_make_task_xml(self, task_name, reason):
        candidate = self.get_task_xml(task_name)
        if candidate != None:
            return candidate
        new_task = f'<Priority><Text>{task_name}</Text><Reason>{reason}</Reason></Priority>'
        self.priorities.append(new_task)
        print(f'created new task to reflect {task_name}\n {reason}\n  {new_task}')
        return new_task

    def repetitive(self, text, last_act, history):
        """ test if content duplicates last_act or entry in history """
        if last_act == None or len(last_act) < 10:
            return False
        prompt=[UserMessage(content="""You are {{$name}}.
Analyze the following NewText for duplicative content. 

A text is considered repetitive if:
1. It duplicates previous thought, speech, or action literally or substantively.
2. It addresses the same subject or task as the earlier text.
3. It adds no new thought, speech, or action.

===Examples===
<LastAct>
None
</LastAct>

<NewText>
Kidd starts walking around the city, observing the strange phenomena and taking mental notes of anything that catches his eye, hoping to uncover some clues about Bellona's secrets.
</NewText>

False
----
<LastAct>
Annie checks the water filtration system filter to ensure it is functioning properly and replace it if necessary.
</LastAct>

<NewText>
Annie checks the water filtration system filter to ensure it is functioning properly and replace it if necessary.
</NewText>

True
-----
<LastAct>
Hey, let's stick together. We're stronger together than alone. Maybe we can help each other figure things out.
</LastAct>

<NewText>
Hey, maybe we should start by looking for any signs of civilization or other people nearby. 
That way, we can make sure we're not completely alone out here. Plus, it might help us figure out where we are and how we got here. What do you think?
</NewText>

False
----
<LastAct>
Joe decides to start exploring the area around him, looking for any clues or signs that could help him understand how he ended up in the forest with no memory.
</LastAct>

<NewText>
Joe continues exploring the forest, looking for any signs of human activity or clues about his past. 
He keeps an eye out for footprints, discarded items, or anything else that might provide insight into how he ended up here.
</NewText>

False
----
<LastAct>
I'll head back to the fields now and finish plowing the south field before the sun sets.
</LastAct>

<NewText>
After finishing plowing the south field, Jean will head over to the north field to check on the irrigation system and ensure the crops are getting enough water.
</NewText>

False

===End Examples===

<LastAct>
{{$last_act}}
</LastAct>

<NewText>
{{$text}}
</NewText>
          
Does the above NewText duplicate, literally or substantively, text appearing in LastAct?
Respond 'True' if NewText is duplicative, 'False' if it is not.
Respond ONLY with 'False' or 'True'.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</END>
"""
                            )
                ]
        response = self.llm.ask({"text":text, "last_act":last_act, "history":history, 'name':self.name},
                                prompt, temp=0.2, stops=['</END>'], max_tokens=12)
        response = response.lower()
        idxt = response.find('true')
        idxf = response.find('false')
        if idxf > -1 and idxt> -1:
            if idxf <= idxt: # sometime runon junk will have true later, ignore it!
                return False
            else:
                return True
        elif idxt> -1:
            return True
        else:
            return False

    def clear_task_if_satisfied(self, task_xml, consequences, world_updates):
        termination_check = xml.find('<Termination_check>', task_xml) if task_xml != None else None
        if termination_check is None:
            return
        satisfied = self.test_priority_termination(termination_check, consequences, world_updates)
        if satisfied:
            task_name = xml.find('<Text>', task_xml)
            if task_name == self.active_task.peek():
                self.active_task.pop()
            try:
                self.priorities.remove(task_xml)
            except Exception as e:
                # this can happen because we don't actually create a task for transient intents from Think or Say
                print(str(e))
            new_intentions = []
            for intention in self.intentions:
                if xml.find('<Source>', intention) != task_name:
                    new_intentions.append(intention)
            self.intentions = new_intentions
            if self.active_task.peek() is None and len(self.priorities) == 0:
                self.update_priorities()

    def acts(self, target, act_name, act_arg='', reason='', source=''):
        #
        ### speak to someone
        #
        self.reason = reason
        if act_name is None or act_arg is None or len(act_name) <= 0 or len(act_arg) <= 0:
            return
        self_verb = 'hear' if act_name == 'Say' else 'see'
        visible_arg = 'Thinking ...' if act_name == 'Think' else act_arg

        # update main display
        intro = f'{self.name}:' if (act_name == 'Say' or act_name == 'Think') else ''
        visible_arg = f"'{visible_arg}'" if act_name == 'Say' else visible_arg
        if act_name != 'Do':
            self.show += f"\n{intro} {visible_arg}"
        self.add_to_history(f"\nYou {act_name}: {act_arg} \n  why: {reason}")

        # update thought
        if act_name == 'Think':
            self.thought = act_arg + '\n ... ' + self.reason
        else:
            self.thought = act_arg[:64] + ' ...\n ... ' + self.reason

        if (act_name == 'Do' or act_name == 'Say') and source != 'dialog' and source != 'watcher':
            if self.active_task.peek() != source:
                self.active_task.push(source)  # dialog is peripheral to action, action task selection is sticky

        # others see your act in the world
        if act_name != 'Say':  # everyone sees it
            for actor in self.context.actors:
                if actor != self:
                    actor.add_to_history(self.show)
        else:
            for actor in self.context.actors:
                if actor != self:
                    if source != 'watcher':  # when talking to watcher, others don't hear it.
                        # create other actor response to say
                        # note this assumes only two actors for now, otherwise need to add target
                        actor.add_to_history(f'You hear {self.name} say: {act_arg}')
                        if actor == target:
                            actor.hear(self, act_arg, source) # don't act immediately on receipt

        # if you acted in the world, ask Context for consequences of act
        # should others know about it?
        if act_name == 'Do':
            consequences, world_updates = self.context.do(self, act_arg)
            if source == None:
                source = self.active_task.peek()
            task_xml = self.get_task_xml(source) if source != None else None
            for task in self.priorities.copy():
                self.clear_task_if_satisfied(task, consequences, world_updates)

            self.show += '\n  ' + consequences.strip() + '\n' + world_updates.strip()  # main window
            self.add_to_history(f"You observe {consequences}")
            print(f'{self.name} setting act_result to {world_updates}')
            self.act_result = world_updates
            if target is not None:  # targets of Do are tbd
                target.sense_input += '\n' + world_updates

        self.previous_action = act_name

        if act_name == 'Say' or act_name == 'Think':
            # make sure future intentions are consistent with what we just did
            # why don't we need this for 'Do?'
            # how does this fit with new model of choose over tasks? won't this be ignored?
            self.update_intentions_wrt_say_think(source, act_arg, reason)

    def update_priorities(self):
        self.active_task = Stack() #new scene, start over. Or maybe we shouldn't?
        print(f'\n{self.name} Updating priorities\n')
        prompt = [UserMessage(content=self.character+"""You are {{$character}}.
Your basic drives include:
<Drives>
{{$drives}}
</Drives>
 
Your task is to create a set of three short term goals given who you are, your Situation, your Stance, your Memory, and your recent RecentHistory as listed below. 
Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your memories include:

<Memory>
{{$memory}}
</Memory>

Recent conversation has been:
<RecentHistory>
{{$history}}
</RecentHistory)>

Your current Stance is:
<Stance>
{{$state}}
</Stance>

Reminder: Your task is to create a set of three short term goals, derived from your priorities, given the Situation, your Stance, your Memory, and your recent RecentHistory as listed above.
List your three most important-short term goals as instantiations from your current needs: 
           
Drives and Stances are listed in a-priori priority order, highest first. However, a short-term goal gets its importance from a combination of Stance order and Stance value (e.g., A Stance value of 'High' is more urgent than one with a value of 'Low').
A short-term goal is one that is achievable in the current situation within a modest period of time, and which advances satisfaction of one or more of the Stances listed above.
List ONLY your most important priorities as simple declarative statements, without any introductory, explanatory, or discursive text.
Your Text must be consistent with the Reason.
Priorities should be as specific as possible. 
For example, if you need sleep, say 'Sleep', not 'Physiological need'.
Similarly, if your safety is threatened by a wolf, respond "Safety from wolf", not merely "Safety"
Reason statements must be concise and limited to a few keywords or at most a single terse sentence.
limit your total response to 120 words. 

Reason step by step to determine the overall importance of each possible short-term goal. Then respond with the three with the highest overall importance. 
Use the XML format:

<Priority> 
  <Text>statement of top priority</Text> 
  <Reason>concise Situation element, state, Memory element, or RecentHistory element that motivates this</Reason>
  <State>the primary Stance from which this priority is derived</State>
  <Termination_check>termination condition</Termination_check>
</Priority>
<Priority>
  <Text>statement of second priority</Text>
  <Reason>concise Situation element, state, Memory element, or RecentHistory element that motivates this</Reason>
  <State>the primary Stance from which this priority is derived</State>
  <Termination_check>termination condition</Termination_check>
</Priority>
<Priority>
  <Text>statement of third priority</Text> 
  <Reason>concise Situation element, state, Memory element, or RecentHistory element that motivates this</Reason>
  <State>the primary Stance from which this priority is derived</State>
  <Termination_check>termination condition</Termination_check>
</Priority>

'Text' is a concise statement of the goal or target to be achieved
'Reason' is the reasoning that explains the motivation for this priority
'State' is the primary Stance element from which this priority derives
'Termination-check' is a concise, testable statement to determine whether recent events satisfy this priority Text

Respond ONLY with the above XML. Do not include any introductory, explanatory, or discursive text.
End your response with:
<STOP>
""")]
        state_map = self.map_state()
        response = self.llm.ask({'character':self.character, 'goals':'\n'.join(self.priorities),
                                 'drives':'\n'.join(self.drives),
                                 'memory':self.memory,
                                 'history':self.format_history(6),
                                 "situation":self.context.current_state,
                                 "state":state_map
                                 },
                               prompt, temp=0.6, stops=['<STOP>'], max_tokens=500)
        try:
            items = xml.findall('<Priority>', response)
            # now actualize these, assuming they are ordered, top priority first
            self.priorities = []
            # don't understand why, but all models *seem* to return priorities lowest first
            items.reverse()
            # found a better way to kick off dialog - when we assign active_task!
            for intention in self.intentions:
                intention_source = xml.find('<Source>', intention)
                if intention_source != 'watcher':
                    #watcher responses never die
                    self.intentions.remove(intention)
            for n, task in enumerate(items):
                if task is not None:
                    task_name = xml.find('<Text>', task)
                    if task_name is not None and str(task_name) != 'None':
                        self.priorities.append(task)
                    else:
                        raise ValueError(f'update priorities created None task_name! {task}')
                else:
                    raise ValueError(f'update priorities created None task_name! {task}')
                    # next will be done in sense so we have current context!
                    #self.actualize_task(n, task) 
        except Exception as e:
            traceback.print_exc()
        #print(f'\n-----Done-----\n\n\n')

    def test_priority_termination(self, termination_check, consequences, updates=''):
        """ test if consequences of recent acts (or world update) have satisfied priority """
        prompt = [UserMessage(content="""A CompletionCriterion is provided below. 
    Reason step-by-step to establish whether this CompletionCriterion has now been met as a result of recent Events,
    using the CompletionCriterion as a guide for this assessment.

    <History>
    {{$history}}
    </History>

    <Events>
    {{$events}}
    </Events>

    <Completion_criterion>
    {{$termination_check}}
    </Completion_criterion>

    Respond using this XML format:

    <Complete> 
        <Level>value of task completion, True, Unknown, or False</Level>
        <Evidence>concise statement of evidence in events to support this level of task completion</Evidence>
    </Complete>

    the 'Level' above should be True if the termination check is met in Events or recent History, 
    Unknown if the Events do not support a definitive assessment, or False if Events provide little or no evidence for task completion.  

    Respond ONLY with the above XML
    Do on include any introductory, explanatory, or discursive text.
    End your response with:
    <END>
    """
                                  )]

        print(f'{self.name} testing priority termination_check: {termination_check}')
        response = self.llm.ask({"termination_check": termination_check, "situation": self.context.current_state,
                                    "memory": self.memory, "events": consequences+'\n'+updates,
                                    "character": self.character, "history": self.format_history()},
                                prompt, temp=0.3, stops=['<END>', '</Complete>'], max_tokens=120)
        satisfied = xml.find('<Level>', response)
        if satisfied != None and satisfied.lower().strip() == 'true':
            print(f'Priority {termination_check} Satisfied!')
        return False if satisfied == None else satisfied.lower().strip() == 'true'

    def actualize_task(self, n, task_xml):
        task_name = xml.find('<Text>', task_xml)
        if task_xml is None or task_name is None:
            raise ValueError(f'Invalid task {n}, {task_xml}')
        last_act = self.get_task_last_act(task_name)
        reason = xml.find('<Reason>', task_xml)
        # crude, needs improvement
        #if reason is not None and 'Low' in reason:
        #    if self.active_task.peek() == task_name:
        #        #active task is now satisfied!
        #        self.active_task.pop() = None
        #    return

        prompt = [UserMessage(content="""You are {{$character}}.
Your task is to generate an Actionable (a 'Think', 'Say', or 'Do') to advance the following task.

<Task>
{{$task}} given {{$reason}}
</Task>

Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your state is:

<State>
{{$state}}
</State>

Your memories include:

<Memory>
{{$memory}}
</Memory>

Recent history includes:
<RecentHistory>
{{$history}}
</RecentHistory>

The Previous specific act for this Task, if any, was:

<PreviousSpecificAct>
{{$lastAct}}
</PreviousSpecificAct>

And the observed result of that was:
<ObservedResult>
{{$lastActResult}}.
</ObservedResult>

Respond with an Actionable, including its Mode and SpecificAct. 

In choosing an Actionable (see format below), you can choose from three Mode values:
- 'Think' - reason about the current situation wrt your state and the task.
- 'Say' - speak, to motivate others to act, to align or coordinate with them, to reason jointly, or to establish or maintain a bond. 
    Say is especially appropriate when there is an actor you are unsure of, you are feeling insecure or worried, or need help.
    For example, if you want to build a shelter with Samantha, it might be effective to Say 'Samantha, let's build a shelter.'
- 'Do' - perform an act with physical consequences in the world.

Review your character for Mode preference. (e.g., 'xxx is thoughtful' implies higher percentage of 'Think' Actionables.) 

A SpecificAct is one which:
- Can be described in terms of specific thoughts, words, physical movements or actions.
- Has a clear beginning and end point.
- Can be performed or acted out by a person.
- Can be easily visualized or imagined as a film clip.
- Is consistent with any action commitments made in your last statements in RecentHistory.
- Is consistent with the Situation (e.g., does not suggest as new an action described in Situation).
- Does NOT repeat, literally or substantively, the previous specific act or other acts by you in RecentHistory.
- Makes sense as the next thing to do or say as a follow-on action to the previous specific act (if any), 
    given the observed result (if any). This can include a new turn in dialog or action, especially when the observed result does not indicate progress on the Task. 
- Significantly advances the story or task at hand.
- Is stated in the appropriate person (voice):
        If a thought (mode is 'Think') or speech (mode is 'Say'), is stated in the first person.
        If an act in the world (mode is 'Do'), is stated in the third person.
 
Prioritize actions that lead to meaningful progress in the narrative.

Dialog guidance:
- If speaking (mode is 'Say'), then:
- Respond in the style of natural spoken dialog, not written text. Use short sentences, contractions, and casual language. Speak in the first person.
- If intended recipient is known (e.g., in Memory) or has been spoken to before (e.g., in RecentHistory), 
    then pronoun reference is preferred to explicit naming, or can even be omitted. Example dialog interactions follow
- Avoid repeating phrases in RecentHistory derived from the task, for example: 'to help solve the mystery'.

Respond in XML:
<Actionable>
  <Mode>'Think', 'Say' or 'Do', corresponding to whether the act is a reasoning, speech, or physical act</Mode>
  <SpecificAct>thoughts, words to speak or physical action description</SpecificAct>
</Actionable>

===Examples===

Task:
Situation: increased security measures; State: fear of losing Annie

Response:
<Actionable>
  <Mode>Do</Mode>
  <SpecificAct>Call a meeting with the building management to discuss increased security measures for Annie and the household.</SpecificAct>
</Actionable>

----

Task:
Establish connection with Joe given RecentHistory element: "Who is this guy?"

Response:
<Actionable>
  <Mode>Say</Mode>
  <SpecificAct>Hi, who are you?</SpecificAct>
</Actionable>

----

Task:
Find out where I am given Situation element: "This is very very strange. Where am I?"

Response:
<Actionable>
  <Mode>Do</Mode>
  <SpecificAct>Samantha starts to look around for any landmarks or signs of civilization, hoping to find something familiar that might give her a clue as to her whereabouts.</SpecificAct>
</Actionable>

===End Examples===

Use the XML format:

<Actionable> 
  <Mode>Think, Say, or Do<Mode>
  <SpecificAct>specific thoughts, words, or action</SpecificAct> 
</Actionable>

Respond ONLY with the above XML.
Your name is {{$name}}, phrase the statement of specific action in your voice.
Ensure you do not duplicate content of a previous specific act.
{{$duplicative}}

Again, the task to translate into an Actionable is:
<Task>
{{$task}} given {{$reason}}
</Task>

Do not include any introductory, explanatory, or discursive text.
End your response with:
<END>"""
                                        )]
        print(f'{self.name} act_result: {self.act_result}')
        act= None; tries = 0
        mapped_state=self.map_state()
        duplicative_insert = ''
        temp = 0.6
        while act is None and tries < 2:
            response = self.llm.ask({'character':self.character, 'goals':'\n'.join(self.priorities),
                                     'memory':self.memory, 'duplicative':duplicative_insert,
                                     'history':self.format_history(6), 'name':self.name,
                                     "situation":self.context.current_state,
                                     "state":mapped_state, "task":task_name, "reason":reason,
                                     "lastAct": last_act, "lastActResult": self.act_result
                                     },
                                    prompt, temp=temp, top_p=1.0,
                                    stops=['</Actionable>','<END>'], max_tokens=180)

            act = xml.find('<SpecificAct>', response)
            mode = xml.find('<Mode>', response)

            if mode is None: mode = 'Do'

            # test for dup act
            if mode =='Say':
                dup = self.repetitive(act, last_act, self.format_history(2))
                if dup:
                    print(f'\n*****Duplicate test failed*****\n  {act}\n')
                    duplicative_insert =f"\nThe following Say is duplicative of previous dialog:\n'{act}'\nWhat else could you say or how else could you say it?"
                    if tries == 0:
                        act = None # force redo
                        temp +=.3
                    else:
                        act = None # skip task, nothing interesting to do
                        pass

            elif mode =='Do':
                dup = self.repetitive(act, last_act, self.format_history(2))
                if dup:
                    print(f'\n*****Repetitive act test failed*****\n  {act}\n')
                    duplicative_insert =f"The following Do is repetitive of a previous act:\n'{act}'. What else could you do or how else could you describe it?"
                    if tries <1:
                        act = None # force redo
                        temp +=.3
                    else:
                        #act = None #skip task, nothing interesting to do
                        pass
            tries += 1
            
        if act is not None:
            print(f'actionable found: task_name {task_name}\n  {act}')  
            print(f'adding intention {mode}, {task_name}')  
            for candidate in self.intentions.copy():
                candidate_source = xml.find('<Source>', candidate)
                if candidate_source == task_name:
                    self.intentions.remove(candidate)
                elif candidate_source is None or candidate_source == 'None':
                    self.intentions.remove(candidate)

            return f'<Intent> <Mode>{mode}</Mode> <Act>{act}</Act> <Reason>{reason}</Reason> <Source>{task_name}</Source> </Intent>'
        else:
            raise UserWarning('No intention constructed')
            #ins = '\n'.join(self.intentions)
            #print(f'Intentions\n{ins}')
                                    
    def forward(self, step):

        # roll conversation history forward.
        ## update physical state
        ## update long-term dialog memory
        prompt = [UserMessage(content=self.character+"""Your name is {{$me}}.
Your task is to update your long-term memory."""\
#Your current situation is:
#
#<Situation>
#{{$situation}}
#</Situation>
+"""
Your previous long-term memory includes:
<Memory>
{{$memory}}
</Memory>

Recent interactions not included in memory:

<RecentHistory>
{{$history}}
</RecentHistory)>

Respond with an complete, concise, accurate, updated memory. 
The updated memory will replace the current long-term memory, and be a concise and accurate record.
Memory should include two types of entries:
1. Information, interactions, and events significant to one or more items in your Stance above.
2. Factually significant information. Note that factual information can change, and should be updated when necessary. 
  For example, a 'fact' about your location in Memory may no longer be valid if an action caused you to move since the last update.
  
Entries in the updated Memory should come from:
1. Information from the previous memory that has not been invalidated by events in RecentHistory
2. New information appearing in RecentHistory

Limit your response to 360 words.
Respond ONLY with the updated long-term memory.
Do not include any introductory, explanatory, discursive, or peripheral text.

End your response with:
END""")
                  ]

        filtered_history = []
        # thoughts are illusory,
        #for item in [memory.text for memory in self.history.n_most_recent(16)]:
        #    if 'Think' not in item:
        #        filtered_history.append(item)
        response = self.llm.ask({'me':self.name, 'memory':self.memory,
                                'history': self.format_history(self.new_memory_cnt),
                                "situation":self.context.current_state},
                               prompt, temp=0.4, stops=['END'], max_tokens=500)
        response = response.replace('<Memory>','')
        self.memory = response
        #self.history = self.history[-2:]
        self.new_memory_cnt = 0
        self.generate_state() # why isn't this update_state? Because update is very short term!
        self.update_priorities()
        
    def update_intentions_wrt_say_think(self, source, text, reason):
        # determine if text implies an intention to act, and create a formatted intention if so
        print(f'Update intentions from say or think\n {text}\n{reason}')

        if source == 'dialog' or source=='watcher':
            print(f' source is dialog or watcher, no intention updates')
            return
        prompt=[UserMessage(content="""Your task is to analyze the following text.

<Text>
{{$text}}
</Text>

Does it include an intention for 'I' to act? 
An action can be physical or verbal.
Thought, e.g. 'reflect on my situation', should NOT be reported as an intent to act.
Respond using the following XML form:

<Analysis>
<Act>False if there is no intention to act, True if there is an intention to act</Act>
<Intention>stated intention to say or act</Intention>
<Mode>'Say' - if intention is to say something, 'Do' - if intention is to perform a physical act/Mode>
</Analysis>

===Examples===

Text:
'Good morning Annie. I'm heading to the office for the day. Call maintenance about the disposal noise please.'

Response:
<Analysis>
<Act>True</Act>
<Intention>Head to the office for the day.</Intention>
<Mode>Do</Mode>
</Analysis>

Text:
'I really should reassure annie.'

Response:
<Analysis>
<Act>True</Act>
<Intention>Annie, you have been performing wonderfully!</Intention>
<Mode>Say</Mode>
</Analysis>

Text:
'Good morning Annie. Call maintenance about the disposal noise please.'

Response:
<Analysis>
<Act>False</Act>
<Intention>None</Intention>
<Mode>NA</Mode>
</Analysis>

Text:
'Reflect on my thoughts and feelings to gain clarity and understanding, which will ultimately guide me towards finding my place in the world.'

Response:
<Analysis>
<Act>False</Act>
<Intention>None</Intention>
<Mode>NA</Mode>
</Analysis>

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the intention analysis in XML as shown above.
End your response with:
END
""")]
        response = self.llm.ask({"text":text}, prompt, temp=0.1, stops=['END', '</Analysis>'], max_tokens=100)
        act = xml.find('<Act>', response)
        if act is None or act.strip() != 'True':
            print(f'no intention in say or think')
            return
        intention = xml.find('<Intention>', response)
        if intention is None or intention=='None':
            print(f'no intention in say or think')
            return
        mode = str(xml.find('<Mode>', response))
        print(f'{self.name} adding intention from say or think {mode}, {source}: {intention}')
        new_intentions = []
        for candidate in self.intentions:
            candidate_source = xml.find('<Source>', candidate)
            if candidate_source != source:
                new_intentions.append(candidate)
        self.intentions = new_intentions
        self.intentions.append(f'<Intent> <Mode>{mode}</Mode> <Act>{intention}</Act> <Reason>{reason}</Reason> <Source>{source}</Source><Intent>')
        if source != None and self.active_task.peek() is None: # do we really want to take a spoken intention as definitive?
            print(f'\nUpdate intention from Say setting active task to {source}')
            self.active_task.push(source)
        #ins = '\n'.join(self.intentions)
        #print(f'Intentions\n{ins}')

    def tell(self, to_actor, message, source='dialog', respond=True):
        if self.active_task.peek() != 'dialog':
            self.active_task.push('dialog')
        self.acts(to_actor,'Say', message, '', 'dialog')
        return

        #generate response intention
    def hear(self, from_actor, message, source='dialog', respond=True):
        # someone says something to you
        if self.active_task.peek() == 'dialog' and self.previous_action == 'Say':
            # assuming this means this is a response to my previous action
            source = self.active_task.peek()
        elif source == 'dialog' and self.previous_action != 'Say' and self.active_task.peek() != 'dialog':
            #not a response to my Say:
            self.active_task.push('dialog')
        elif source != 'dialog':
            print(f' non dialog tell')
        if source == 'dialog':
            self.dialog_length += 1
            if self.dialog_length > 1: # end a dialog after one turn
                self.dialog_length = 0;
                # clear all actor pending dialog tasks and intentions:
                if self.active_task.peek() == 'dialog':
                    self.active_task.pop()

                for priority in self.priorities.copy():
                    if xml.find('<Text>', priority) == 'dialog':
                        print(f'{self.name} removing dialog task!')
                        self.priorities.remove(priority)
                for intention in self.intentions.copy():
                    if xml.find('<Source>', intention) == 'dialog':
                        print(f'{self.name} removing dialog intention')
                        self.intentions.remove(intention)
                if self.active_task.peek() is None:
                    self.update_priorities()
                #ignore this tell, dialog over
                return
            elif self.active_task.peek() != None and self.active_task.peek() != 'dialog':
                self.active_task.push('dialog')

        print(f'\n{self.name} tell received from {from_actor.name}, {message} {source}\n')
        self.add_to_history(f'You hear {from_actor.name} say {message}')
        if not respond:
            return
        #generate response intention
        prompt=[UserMessage(content="""Respond to the input below as {{$name}}.

{{$character}}.

Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your state is:

<State>
{{$state}}
</State>

Your memories include:

<Memory>
{{$memory}}
</Memory>

Recent conversation has been:
<RecentHistory>
{{$history}}
</RecentHistory)>

Use the following XML template in your response:
<Answer>
<Response>response to this statement</Response>
<Reason>concise reason for this answer</Reason>
<Unique>'True' if you have something new to say, 'False' if you have nothing worth saying or your response is a repeat or rephrase of previous responses</Unique>
</Answer>

Your last response was:
<PreviousResponse>
{{$last_act}}
</PreviousResponse>

{{$activity}}

The statement you are responding to is:
<Input>
{{$statement}}
</Input>>

Reminders: 
- Your task is to generate an response to the statement using the above XML format.
- The response should be in keeping with your character's State.
- The response can include body language or facial expressions as well as speech (e.g. 'Joe nods in agreement...')
- The response is in the context of the Input, and can assume all Input information is implicitly present in the dialog.
- Respond in a way that expresses an opinion on current options or proposes a next step to solving the central conflict in the dialog.
- If the intent of the response is to agree, state agreement without repeating the feelings, opinions, motivations, or goals.
- Speak in your own voice. Do not echo the speech style of the Input. 
- Respond in the style of natural spoken dialog. Use short sentences, contractions, and casual language.
 
If intended recipient is known (e.g., in Memory) or has been spoken to before (e.g., in RecentHistory or Input), then the referent should be omitted or use pronoun reference.
===Example===
RecentHistory:
Samantha: Hi Joe. It's good to know that I'm not alone in feeling lost and confused. Maybe together we can find a way out of this forest and find out how we got here.

Response:
Yeah. I hope we can find our way out here. And yeah, good idea on shelter and water. this is so weird.

===Example===
Input:
Samantha says Great! Let's stick together and look for clues or landmarks to where we are or how to get back. 
I'm feeling disoriented, but having someone else here with me helps.

Joe: Sounds like a plan. Let's stay within sight. Keep an eye out for any paths or trails or trail markers.
===End Example===


Respond only with the above XML
Do not include any additional introductory, explanatory, or discursive text. 
End your response with:
END
""")]
        mapped_state=self.map_state()
        last_act = ''
        if source in self.last_acts: #this is 'from_source' task_name!
            last_act=self.last_acts[source]
        activity = ''
        if self.active_task.peek() != None and self.active_task.peek() != 'dialog':
            activity = f'You are currently actively engaged in {self.get_task_xml(self.active_task.peek())}'
        answer_xml = self.llm.ask({'character': self.character, 'statement': f'{from_actor.name} says {message}',
                                   "situation": self.context.current_state,"name":self.name,
                                   "state": mapped_state, "memory": self.memory, "activity": activity,
                                   'history': self.format_history(6), 'last_act': str(last_act)
                                   }, prompt, temp=0.8, stops=['END', '</Answer>'], max_tokens=180)
        response = xml.find('<response>', answer_xml)
        if response is None:
            return
        unique = xml.find('<Unique>', answer_xml)
        if unique is None or 'false' in unique.lower():
            if self.active_task.peek() == 'dialog':
                self.active_task.pop()
            return 
        reason = xml.find('<Reason>', answer_xml)
        print(f' Queueing dialog response {response}')
        if source != 'watcher': #Is this right? Does a say from another user always initiate a dialog?
            response_source='dialog'
        else:
            response_source = 'watcher'
        dup = self.repetitive(response, last_act, '')
        if dup:
            if self.active_task.peek() == 'dialog':
                self.active_task.pop()
            return # skip response, adds nothing
        self.intentions.append(f'<Intent> <Mode>Say</Mode> <Act>{response}</Act> <Target>{from_actor.name}</Target><Reason>{str(reason)}</Reason> <Source>{response_source}</Source></Intent>')
        return response+'\n '+reason

    def choose(self, sense_data, action_choices):
        prompt = [UserMessage(content=self.character + """\nYour current situation is:

        <Situation>
        {{$situation}}
        </Situation>

        Your fundamental needs / drives include:

        <Drives>
        {{$drives}}
        </Drives> 

        Your state is:

        <State>
        {{$state}}
        </State>

        Your memories include:

        <Memory>
        {{$memory}}
        </Memory>

        Recent conversation has been:
        <RecentHistory>
        {{$history}}
        </RecentHistory)>

        Your current priorities include:
        <Priorities>
        {{$priorities}}
        </Priorities>

        New Observation:
        <Input>
        {{$input}}
        </Input>

        Given who you are, your current Priorities, New Observation, and the other information listed above, think step-by-step 
        and choose your most pressing, highest need / priority action to perform from the numbered list below:

        <Actions>
        {{$actions}}
        </Actions>

        Consider the conversation history in choosing your action. 
        Respond in the context of the RecentHistory (if any) and in keeping with your character. 
        Use the number for the selected action to instantiate the following XML format:

        <Action>
        index-number of chosen action
        </Action>

        Respond with the above XML, instantiated with the selected action number from the Action list. 
        Do not include any introductory, explanatory, or discursive text, 
        End your response with:
        END
        """
                              )]

        mapped_state = self.map_state()
        action_choices = [f'{n} - {action}' for n, action in enumerate(action_choices)]
        if len(action_choices) > 1:
            response = self.llm.ask({'input': sense_data + self.sense_input, 'history': self.format_history(6),
                                    "memory": self.memory, "situation": self.context.current_state,
                                    "state": mapped_state, "drives": '\n'.join(self.drives),
                                    "priorities": '\n'.join([xml.find('<Text>', task) for task in self.priorities]),
                                    "actions": '\n'.join(action_choices)
                                    }, prompt, temp=0.7, stops=['END', '</Action>'], max_tokens=300)
        # print(f'sense\n{response}\n')
        self.sense_input = ' '
        if 'END' in response:
            idx = response.find('END')
            response = response[:idx]
        if '<|im_end|>' in response:
            idx = response.find('<|im_end|>')
            response = response[:idx]
        index = -1
        choice = xml.find('<Action>', response)
        if choice is None:
            choice = response.strip()  # llms sometimes ignore XML formatting for this prompt
        if choice is not None:
            try:
                draft = int(choice.strip())
                if draft > -1 and draft < len(action_choices):
                    index = draft  # found it!
            except Exception as e:
                try:
                    draft = find_first_digit(response)
                    draft = int(draft) if (draft != None and int(draft) < len(action_choices)) else None
                    index = draft if draft != None else -1
                except Exception as e:
                    traceback.print_exc()
        if index > -1:
            return index
        else:
            return 0

    def senses(self, sense_data='', ui_queue=None):
        print(f'\n*********senses***********\nCharacter: {self.name}, active task {self.active_task.peek()}')
        all_actions={"Act": """Act as follows: '{action}'\n  because: '{reason}'""",
                     "Answer": """ Answer the following question: '{question}'""",
                     "Say": """Say: '{text}',\n  because: '{reason}'""",
                     "Think": """Think: '{text}\n  because" '{reason}'""",
                     "Discuss": """Reason step-by-step about discussion based on current situation, your state, priorities, and RecentHistory. Respond using this template to report the discussionItem and your reasoning:
<Action> <Name>Say</Name> <Arg> ..your Priorities or Memory, or based on your observations resulting from previous Do actions.></Arg> <Reason><terse reason for bringing this up for discussion></Reason> </Action>
"""}

        # do after we have all relevant updates from context and other actors
        dialog_active = False
        my_active_task = self.active_task.peek()
        intention = None
        for candidate in self.intentions:
            source = xml.find('<Source>', candidate)
            if source == 'dialog' or source =='watcher':
                intention = candidate

        if intention is None and my_active_task != None and my_active_task != 'dialog' and my_active_task != 'watcher':
            full_task = self.get_task_xml(my_active_task)
            intention = self.actualize_task(0, full_task)
            print(f'Found active_task {my_active_task}')

        if intention is None:
            intention_choices = []
            for n, task in enumerate(self.priorities):
                choice =  f"{n}. {xml.find('<Text>', task)}, because {xml.find('<Reason>', task)}"
                intention_choices.append(choice)
                print(f'{self.name} considering action option {choice}')

            if len(intention_choices) == 0:
                print(f'{self.name} Oops, no available acts')
                return

            task_index = 0
            if len(intention_choices) > 1:
                task_index = self.choose(sense_data, intention_choices)
            intention = self.actualize_task(task_index, self.priorities[task_index])

        self.intentions = [] # start anew next time
        act_name = xml.find('<Mode>', intention)
        if act_name is not None:
            self.act_name = act_name.strip()
        act_arg = xml.find('<Act>', intention)
        self.reason = xml.find('<Reason>', intention)
        source = xml.find('<Source>', intention)
        print(f'{self.name} choose {intention}')
        task_name = xml.find('<Source>', intention)
        refresh_task = None # will be set to task intention to be refreshed if task is chosen for action

        task_xml = None
        refresh_task = None
        if not dialog_active and act_name == 'Say' or act_name == 'Do':
            task_name = xml.find('<Source>', intention)
            # for now very simple task tracking model:
            if task_name is None:
                task_name = self.make_task_name(self.reason)
                print(f'No source found, created task name: {task_name}')
            task_xml = self.find_or_make_task_xml(task_name, self.reason)
            refresh_task = task_xml # intention for task was removed about, remember to rebuild
            self.last_acts[task_name] = act_arg # this should pbly be in acts, where we actually perform
        if act_name == 'Think':
            task_name = xml.find('<Source>', intention)
            #task_name = self.make_task_name(self.reason)
            self.reason = xml.find('<Reason>', intention)
            self.thought = act_arg+'\n  '+self.reason
        target = None

        if act_name == 'Say':
            #responses, at least, explicitly name target of speech.
            target_name = xml.find('<Target>', intention)
            if target_name == None:
                target_name = self.say_target(act_arg)
            if target_name != None:
                target = self.context.get_actor_by_name(target_name)

        #this will affect selected act and determine consequences
        self.acts(target, act_name, act_arg, self.reason, source)

        # maybe we should do this at start of next sense?
        #if refresh_task is not None and task_name != 'dialog' and task_name != 'watcher':
        #    for task in self.priorities:
        #        if refresh_task == task:
        #            print(f"refresh task just before actualize_task call {xml.find('<Text>', refresh_task)}")
        #            self.actualize_task('refresh', refresh_task) # regenerate intention
