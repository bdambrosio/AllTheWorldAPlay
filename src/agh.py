import os
import time
import random
import traceback
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
import llm_api

def findall(key, form):
    """ find multiple occurrences of an xml field in a string """
    idx = 0
    items = []
    forml = form.lower()
    keyl = key.lower()
    keyle = keyl[0] + '/' + keyl[1:]
    while idx < len(forml):
        start_idx = forml[idx:].find(keyl)
        if start_idx == -1:
            return items
        start_idx += len(keyl)
        end_idx = forml[idx+start_idx:].find(keyle)
        if end_idx == -1:
            return items
        items.append(form[idx+start_idx:idx+start_idx+end_idx].strip())
        idx += start_idx + end_idx + len(keyle)
    return items

def find(key, form):
    """ find first occurrences of an xml field in a string """
    forml = form.lower()
    keyl = key.lower()
    keyle = keyl[0] + '/' + keyl[1:]
    start_idx = forml.find(keyl)
    if start_idx == -1:
        return None
    start_idx += len(keyl)
    end_idx = forml[start_idx:].find(keyle)
    if end_idx == -1:
        return form[start_idx:]
    return form[start_idx: start_idx + end_idx]


class Context ():
    def __init__(self, actors, situation):
        self.initial_state = situation
        self.current_state = situation
        self.actors = actors
        for actor in self.actors:
            actor.context=self
        self.name='World'
        self.llm = None
        
    def history(self):
        try:
            if self.actors is None or self.actors[0] is None:
                return ''
            if self.actors[0].history is None:
                return ''
            history = '\n\n'.join(self.actors[0].history)
            hs = history.split('\n')
            hs_renamed = [self.actors[0].name + s[3:] if s.startswith('You') else s for s in hs]
            history = '\n'.join(hs_renamed)
        except Exception as e:
            traceback.print_exc()
        return history

    def image(self, filepath, image_generator = 'tti_serve'):
        try:
            state = '. '.join(self.current_state.split('.')[:2])
            characters = '\n'+'.\n'.join([entity.name + ' is '+entity.character.split('.')[0][8:] for entity in self.actors])
            prompt=state+characters
            #print(f'calling generate_dalle_image\n{prompt}')
            if image_generator == 'tti_serve':
                llm_api.generate_image(f"""wide-view photorealistic. {prompt}""", size='512x512', filepath=filepath)
            else:
                llm_api.generate_dalle_image(f"""wide-view photorealistic. {prompt}""", size='512x512', filepath=filepath)
        except Exception as e:
            traceback.print_exc()
        return filepath

    def world_updates_from_act_consequences(self, consequences):
        prompt=[UserMessage(content="""Given the following immediate effects of an action on the environment, generate zero to two concise sentences to add to the following state description.
It may be there are no significant updates to report.
Limit your changes to the consequences for elements in the existing state or new elements added to the state.
Most important are those consequences that might activate or inactive tasks or intentions by actors.

<ActionEffects>
{{$consequences}}
</ActionEffects>

<Environment>
{{$state}}
</Environment>

Your response should be concise, and only include only statements about changes to the existing Environment.
Do NOT repeat elements of the existing Environment, respond only with significant changes.
Do NOT repeat as an update items already present at the end of the Environment statement.
Your updates should be dispassionate. 

<Updates>
concise statement of significant changes to Environment, if any.
</Updates>

Include ONLY the concise updated state description in your response. 
Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
End your response with:
<END>""")
                ]

        response = self.llm.ask({"consequences":consequences, "state":self.current_state},
                                prompt, temp=0.5, stops=['<END>'], max_tokens=60)
        updates = find('<Updates>', response)
        if updates is not None:
            self.current_state += '\n'+updates
        else:
            updates = ''
        return updates
    
    def do(self, actor, action):
        prompt=[UserMessage(content="""You are simulating a dynamic world. 
Your task is to determine the result of {{$name}} performing the following action:

<Action>
{{$action}}
</Action>

in the current situation: 

<Situation>
{{$state}}
</Situation>

given {{$name}} basic drives are:

<Drives>
{{$drives}}
</Drives> 

Respond with the observable result.
Respond ONLY with the observable immediate effects of the above Action on the environment and characters.
It is usually not necessary or desirable to repeat the above action statement in your response.
Format your response as one or more simple declaractive sentences.
Include in your response:
- changes in the physical environment, e.g. 'the door opens', 'the rock falls',...
- sensory inputs, e.g. {{$name}} 'sees ...', 'hears ...', 
- changes in {{$name}}'s possessions (e.g. {{$name}} 'gains ... ',  'loses ... ', / ... / )
- changes in {{$name})'s or other actor's state (e.g., {{$name}} 'becomes tired' / 'is injured' / ... /).
Do NOT extend the scenario with any follow on actions or effects.
Be extremely terse when reporting character emotional state, only report the most significant emotional state changes.
Be concise!
Do not include any Introductory, explanatory, or discursive text.
End your reponse with:
<END>
""")]
        history = self.history()
        consequences = self.llm.ask({"name":actor.name, "action":action, "drives":actor.drives,
                                     "state":self.current_state}, prompt, temp=0.7, stops=['END'], max_tokens=300)

        if consequences.endswith('<'):
            consequences=consequences[:-1]
        world_updates=self.world_updates_from_act_consequences(consequences)
        print(f'\nContext Do consequences:\n {consequences}')
        print(f' Context Do world_update:\n {world_updates}\n')
        return consequences, world_updates

    def senses (self, sense_data='', ui_task_queue=None):
        # since at the moment there are only two chars, each with complete dialog, we can take it from either.
        # not really true, since each is ignorant of the others thoughts
        history = self.history()
        if random.randint(1,7) == 1:
            event = """
Include a event occurence consistent with the PreviousState below, such as appearance of a new object, natural event such as weather (if outdoors), communication event such as email arrival (if devices available to receive such), etc.

===Examples===
PreviousState:
Apartment

History:
worry about replacement

Event:
Annie receives an email directed to her personally from an unknown agent.

-----

PreviousState:
Open forest 

History:
Safety, hunger

Event:
Joe finds a sharp object that can be used as a tool.
-----

===End Examples===

"""
        else:
            event = ""

        prompt = [UserMessage(content="""You are a dynamic world. Your task is to update the environmemt description. 
Include day/night cycles and weather patterns. 
Update location and other physical situation characteristics as indicated in the History.
Your response should be concise, and only include only an update of the physical situation.
        """ + event + """
Your situation description should be dispassionate, 
and should begin with a brief description of the current physical space suitable for a text-to-image generator. 
That is, advance the situation by approximately 3 hours. The situation as of 3 hours ago was:

<PreviousSituation>
{{$situation}}
</PreviousSituation> 

In the interim, the characters in the world had the following interactions:

<History>
{{$history}}
</History>

Respond using the following XML format:

<Situation>
Sentence describing physical space, suitable for image generator,
Updated State description of about 400 words
</Situation>

Respond with an updated situation description reflecting a time passage of 3 hours and the events that have occurred.
Include ONLY the concise updated situation description in your response. Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
Do NOT carry the storyline beyond three hours from PreviousSituation.
Limit your response to about 400 words
End your response with:
END""")]
            
        response = self.llm.ask({"situation":self.current_state, 'history':history}, prompt, temp=0.6, stops=['END'], max_tokens=600)
        new_situation = find('<Situation>', response)
        if new_situation is not None:
            self.current_state=new_situation
            self.show='\n-----scene-----\n'+new_situation
        for actor in self.actors:
            actor.forward(3) # forward three hours and update history, etc
        return response

class Character():
    def __init__(self, name, character_description):
        self.name = name
        self.character = character_description
        self.history = []
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
        self.active_task = None # task character is actively pursuing.
        self.last_acts = {} # a set of priorities for which actions have been started, and their states.
        self.dialog_status = 'Waiting' # Waiting, Pending

    def add_to_history(self, message):
        message = message.replace('\\','')
        self.history.append(message)
        self.history = self.history[-5:] # memory is fleeting, otherwise we get very repetitive behavior

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

    def acts(self, target, act_name, act_arg='', reason='', source=''):
        #
        ### speak to someone
        #
        self.reason = reason
        if act_name is not None and act_arg is not None and len(act_name) >0 and len(act_arg) > 0:
            self_verb = 'hear' if act_name == 'Say' else 'see'
            visible_arg = '' if act_name == 'Think' else act_arg
            
            # update main display
            intro = f'{self.name}:' if act_name == 'Say' else ''
            visible_arg = f"'{visible_arg}'" if act_name == 'Say' else visible_arg
            intro = f'{self.name}:' if act_name == 'Say' else ''
            if act_name != 'Do':
                self.show += f"\n{intro} {visible_arg}"
            self.add_to_history(f"\nYou {act_name}: {act_arg} \n  why: {reason}")

            # update thought
            if act_name == 'Think':
                self.thought = act_arg+'\n ... '+self.reason
            else:
                self.thought = act_arg[:42]+' ...\n ... '+self.reason


            if source != 'dialog' and source != 'watcher': 
                self.active_task = source # dialog is peripheral to action, action task selection is sticky

            #others see your act in the world
            if act_name != 'Say': # Say impact on recipient handled by tell
                for actor in self.context.actors:
                    if actor != self:
                        actor.add_to_history(self.show)
            elif act_name == 'Say':
                for actor in self.context.actors:
                    if actor != self:
                        if source != 'watcher': # when talking to watcher, others don't hear it.
                            # create other actor response to say
                            # note this assumes only two actors for now, otherwise need to add target
                            actor.add_to_history(f'You hear {self.name} say: {act_arg}')
                            actor.tell(self, act_arg, source)


            # if you acted in the world, ask Context for consequences of act
            # should others know about it?
            if act_name =='Do':
                consequences, world_updates = self.context.do(self, act_arg)
                self.show += '\n  '+ consequences+'\n'+world_updates # main window
                self.add_to_history(f"You observe {consequences}")
                print(f'{self.name} setting act_result to {world_updates}')
                self.act_result = world_updates
                if target is not None: # targets of Do are tbd
                    target.sense_input += '\n'+world_updates

            self.previous_action = act_name

            if act_name == 'Say' or act_name == 'Think':
                #make sure future intentions are consistent with what we just did
                # why don't we need this for 'Do?'
                self.update_intentions_wrt_say_think(source, act_arg, reason)

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


    def generate_state(self):
        """ generate a state to track, derived from basic drives """
        prompt=[UserMessage(content="""A basic Drive is provided below. 
Your task is to create an instantiated state description to track your current state wrt this drive.
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

<Drive> <Term>term designating the above drive</Term> <Assessment>a one or two word value, e.g. high, medium-high, medium, medium-low, low, etc.</Assessment> <Trigger> a concise, few word statement of the specific situational trigger for this drive instatiation </Trigger> </Drive>

The 'Term' must contain at most three words concisely summarizing the goal of this state element, for example 'Identify stranger's intention'.

The 'Assessment' is one word describing the current distance from goal and/or it's urgency. eg, 'high' indicates we are far from the goal or it is highly important to make progress on this drive. Reason about factors like the position of the related drive in the Drive list (higher is more important), the Character and how they might react to the trigger (below), and how unique the Trigger is as an event. Look for reasons to given an assessment other than medium. Use nuanced assessments (very high, medium-low, etc) where appropriate. 

The 'Trigger' is a concise phrase or sentence designating the Character, Situation, Memory, or History element(s) that trigger this instantion.


Respond ONLY with the above XML
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
                                     "character":self.character, "history":self.history},
                                    prompt, temp=0.3, stops=['<END>'], max_tokens=60)
            term = find('<Term>', response)
            assessment = find('<Assessment>', response)
            trigger=find('<Trigger>', response)
            # these will be used for remainder of scenario
            self.state[term] ={"drive":drive, "state": assessment, 'trigger':trigger}
        print(f'{self.name}initial state {self.state}')
        
    def map_state(self):
        """ map state for llm input """
        mapped = []
        for key, item in self.state.items():
            trigger = item['drive']
            value = item['state']
            mapped.append(f"- '{key}: {trigger}', State: '{value}'")
        return '\n'.join(mapped)

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
END""")
                  ]
        mapped_state = self.map_state()
        template = self.make_state_template()
        response = self.llm.ask({'character':self.character, 'memory':self.memory,
                                 'history':'\n\n'.join(self.history),
                                 "situation":self.context.current_state,
                                 "state":mapped_state, "template":template
                                },
                               prompt, temp=0.2, stops=['END'], max_tokens=180)
        state_xml = find('<UpdatedState>', response)
        print(f'\n{self.name} updating state')
        for key, item in self.state.items():
            update = find(f'<{key}>', state_xml)
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

    def get_task_name(self, term):
        for task in list(self.last_acts.keys()):
            match=self.synonym_check(task, term)
            if match:
                return task
        return None
            
    def set_task_last_act(self, term, act):
        # pbly don't need this, at set we have canonical task
        task = self.get_task_name(term)
        if task == None:
            print(f'SET_TASK_LAST_ACT {self.name} no match found for term: {term}, {act}')
            self.last_acts[term] = act
        else:
            print(f'SET_TASK_LAST_ACT {self.name} match found: term {term}, task {task}\n  {act}')
            self.last_acts[task] = act

    def get_task_last_act(self, term):
        task = self.get_task_name(term)
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
        return find('<Motivation', response)
                    
    def get_task_xml(self, task_name):
        for candidate in self.priorities:
            #print(f'find_or_make testing\n {candidate}\nfor name {task_name}')
            if task_name == find('<Text>', candidate):
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
        prompt=[UserMessage(content="""You are {{$name}}.
Analyze the following text for duplicative content. 
Does the following NewText duplicate, literally or substantively, text appearing in the LastAct or that is by you in History?
To be repetitive, speech or an act must meet the following condition:
- Literally or substantively duplicate previous speech or act by you {{$name}} in LastAct or in history.
- Address the same subject, task, or theme as the earlier speech or act it appears to repeat.

Speech or an act that appears to repeat but which also contains substantive additional content, is NOT repetitive.

<NewText>
{{$text}}
</NewText>

<LastAct>
{{$last_act}}
<LastAct>
          
<History>
{{$history}}
</History>

Reason step by step to a result, 'True' if NewText is repetitive, 'False' if it is not. 
If the answer is that it does not duplicate, respond 'False'
If the answer is that it is largely duplicative, respond 'True'.

===Examples===
NewText:
Annie checks the water filtration system filter to ensure it is functioning properly and replace it if necessary.

LastAct:
Annie checks the water filtration system filter to ensure it is functioning properly and replace it if necessary.

Response:
True

-----

===End Examples===

Respond ONLY with 'False' or 'True'.
Do not include any introductory, explanatory, or discursive text.
End your response with:
</END>
"""
                            )
                ]
        response = self.llm.ask({"text":text, "last_act":last_act, "history":history, 'name':self.name},
                                prompt, temp=0.5, stops=['</END>'], max_tokens=12)
        if 'true' in response.lower():
            return True
        elif 'false' in response.lower():
            return False
        else:
            return False # don't redo if you don't have to
                

    def update_priorities(self):
        self.active_task = None
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
The List ONLY your most important priorities as simple declarative statements, without any introductory, explanatory, or discursive text.
Your Text must be consistent with the Reason.
Priorities should be as specific as possible. 
For example, if you need sleep, say 'Sleep', not 'Physiological need'.
Similarly, if your safety is threatened by a wolf, respond "Safety from wolf", not merely "Safety"
Reason statements must be concise and limited to a few keywords or at most a single terse sentence.
limit your total response to 120 words. 

Reason step by step to determine the overall importance of each possible short-term goal. Then respond with the three with the highest overall importance. 
Use the XML format:

<Priorities>
<Priority> <Text>statement of top priority</Text> <Reason>concise Situation element, state, Memory element, or RecentHistory element that motivates this</Reason> </Priority>
<Priority> <Text>statement of second priority</Text> <Reason>concise Situation element, state, Memory element, or RecentHistory element that motivates this</Reason> </Priority>
<Priority> <Text>statement of third priority</Text> <Reason>concise Situation element, state, Memory element, or RecentHistory element that motivates this</Reason> </Priority>
</Priorities>

Respond ONLY with the above XML. Do not include any introductory, explanatory, or discursive text.
End your response with:
END
""")]
        state_map = self.map_state()
        response = self.llm.ask({'character':self.character, 'goals':'\n'.join(self.priorities),
                                 'drives':'\n'.join(self.drives),
                                 'memory':self.memory,
                                 'history':'\n\n'.join(self.history),
                                 "situation":self.context.current_state,
                                 "state":state_map
                                 },
                               prompt, temp=0.6, stops=['</Priorities>', 'END'], max_tokens=180)
        try:
            priorities = find('<Priorities>', response)
            if priorities is None:
                return
            items = findall('<Priority>', priorities)
            # now actualize these, assuming they are ordered, top priority first
            self.priorities = []
            # don't understand why, but all models *seem* to return priorities lowest first
            items.reverse()
            # found a better way to kick off dialog - when we assign active_task!
            for intention in self.intentions:
                intention_source = find('<Source>', intention)
                if intention_source != 'watcher':
                    #watcher responses never die
                    self.intentions.remove(intention)
            for n, task in enumerate(items):
                if task is not None:
                    task_name = find('<Text>', task)
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

    def actualize_task(self, n, task_xml):
        task_name = find('<Text>', task_xml)
        if task_xml is None or task_name is None:
            raise ValueError(f'Invalid task {n}, {task_xml}')
        print(f'\n Actualizing task {n} {task_name}')
        last_act = self.get_task_last_act(task_name)
        reason = find('<Reason>', task_xml)
        # crude, needs improvement
        #if reason is not None and 'Low' in reason:
        #    if self.active_task == task_name:
        #        #active task is now satisfied!
        #        self.active_task = None
        #    return

        prompt = [UserMessage(content="""You are {{$character}}.
Your task is to generate an Actionable (act, a 'Think', 'Say', or 'Do') to advance the following perceived task:

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

Recent conversation has been:
<RecentHistory>
{{$history}}
</RecentHistory)>

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
- 'Say' - speak, to motivate others to act, to align or coordinate with them, to reason jointly to establish or maintain a bond. For example, if you want to build a shelter with Samantha, it might be effective to Say '
- 'Do' - perform an act with physical consequences in the world.
Review your character for Mode preference. (e.g., 'xxx is thoughtful' implies higher percentage of 'Think' Actionables.) 

A SpecificAct is one which:
- Can be described in terms of specific physical movements or steps
- Has a clear beginning and end point
- Can be performed or acted out by a person
- Can be easily visualized or imagined as a film clip
- Is consistent with any action commitments made in your last statements in RecentHistory
- Does NOT repeat, literally or substantively, the previous specific act or other acts by you in RecentHistory.
- Makes sense as the next thing to do or say as a follow-on action to the previous specific act (if any), given the observed result (if any). This can include a new turn in dialog or action, especially when the observed result does not indicate progress on the Task. 
- Is stated in the appropriate person (voice):
        If a thought (mode is 'Think') or speech (mode is 'Say'), is stated in the first person.
        If an act in the world (mode is 'Do'), is stated in the third person.
 
Dialog guidance:
- If speaking (mode is 'Say'), then:
- Respond in the style of natural spoken dialog, not written text. Use short sentences, contractions, and casual language. Speak in the first person.
- If intended recipient is known (e.g., in Memory) or has been spoken to before (e.g., in RecentHistory), then pronoun reference is preferred to explicit naming, or can even be omitted. Example dialog interactions follow
- Avoid repeating phrases in RecentHistory derived from the task, for example: 'to help solve the mystery'.

===Example===
RecentHistory:
Samantha: Hi Joe, thanks for introducing yourself. It's good to know that I'm not alone in feeling lost and confused. Maybe together we can find a way out of this forest and solve the mystery of how we got here.
You Say Hi Samantha, I'm glad we can help each other out. Let's work together to find a way out of this forest and figure out how we got here. We might also be able to find some shelter and water along the way. 

Response:
That's great to hear. I'm glad we can work together to find our way out of this forest. And don't worry, we'll definitely keep an eye out for shelter and water. I understand how you're feeling, but let's try to stay positive and help each other out.

===Example===
Samantha: Hi, it's nice to meet you too. I'm glad we can be here for each other in this confusing situation. Do you have any ideas on how we can find our way out of this forest and maybe solve the mystery of how we got here?

Joe: Well, I'm glad we can be here for each other too. As for finding our way out of this forest, I'm afraid I don't have any ideas yet. But let's keep our eyes open for any clues or landmarks that might help us figure out where we are and how to get back.
===End Example===


Respond in XML:
<Actionable>
  <Mode>'Think', 'Say' or 'Do', corresponding to whether the act is a reasoning, speech, or physical act</Mode>
  <SpecificAct>thoughts, words to speak or physical action description</SpecificAct>
</Actionable>

===Examples===

Task:
'Situation: increased security measures; State: fear of losing Annie'

Response:
Response:
<Actionable>
  <Mode>Do</Mode>
  <SpecificAct>Call a meeting with the building management to discuss increased security measures for Annie and the household.</SpecificAct>
</Actionable>


Task:
'Establish connection with Joe given RecentHistory element: "Who is this guy?"'

Response:
<Actionable>
  <Mode>Say</Mode>
  <SpecificAct>Hi, who are you?</SpecificAct>
</Actionable>

Task:
'Find out where I am given Situation element: "This is very very strange. Where am I?"'

Response:
<Actionable>
  <Mode>Do</Mode>
  <SpecificAct>Samantha starts to look around for any landmarks or signs of civilization, hoping to find something familiar that might give her a clue as to her whereabouts.</SpecificAct>
</Actionable>

===End Examples===

Use the XML format:

<Actionable> <Mode>Think, Say, or Do<Mode><SpecificAct>statement of specific action</SpecificAct> </Actionable>

Respond ONLY with the above XML.
Your name is {{$name}}, phrase the statement of specific action in your voice.
Ensure you do not duplicate content of a previous specific act.
{{$duplicative}}

The task you are to transform into a specific action is:

<Task>
{{$task}} given: {{$reason}}
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
                                     'history':'\n\n'.join(self.history), 'name':self.name,
                                     "situation":self.context.current_state,
                                     "state":mapped_state, "task":task_name, "reason":reason,
                                     "lastAct": last_act, "lastActResult": self.act_result
                                     },
                                    prompt, temp=temp, top_p=1.0,
                                    stops=['</Actionable>','<END>'], max_tokens=180)

            act = find('<SpecificAct>', response)
            mode = find('<Mode>', response)

            if mode is None: mode = 'Do'

            # test for dup act
            if mode=='Say':
                dup = self.repetitive(act, last_act, '\n\n'.join(self.history))
                if dup:
                    print(f'\n*****Duplicate test failed*****\n  {act}\n')
                    duplicative_insert =f"The following Say is duplicative of previous dialog:\n'{act}'. What else could you say?"
                    if tries == 0:
                        act = None # force redo
                        temp +=.3
                    else: act = None # skip task, nothing interesting to do

            elif mode=='Do':
                dup = self.repetitive(act, last_act, '\n\n'.join(self.history))
                if dup:
                    print(f'\n*****Repetitive act test failed*****\n  {act}\n')
                    duplicative_insert =f"The following Do is repetitive of a previous act:\n'{act}'. What else could you do?"
                    if tries <=1:
                        act = None # force redo
                        temp +=.3
                    else:
                        act = None #skip task, nothing interesting to do
            tries += 1
            
        if act is not None:
            print(f'actionable found: task_name {task_name}\n  {act}')  
            print(f'adding intention {mode}, {task_name}')  
            for candidate in self.intentions:
                candidate_source = find('<Source>', candidate)
                if candidate_source == task_name:
                    self.intentions.remove(candidate)
                elif candidate_source is None or candidate_source == 'None':
                    self.intentions.remove(candidate)

            self.intentions.append(f'<Intent> <Mode>{mode}</Mode> <Act>{act}</Act> <Reason>{reason}</Reason> <Source>{task_name}</Source> </Intent>')
            #ins = '\n'.join(self.intentions)
            #print(f'Intentions\n{ins}')
                                    
    def forward(self, num_hours):
        # roll conversation history forward.
        ## update physical state
        ## update long-term dialog memory
        prompt = [UserMessage(content=self.character+"""Your name is {{$me}}.
Your task is to update your long-term memory of interactions with other actors.
Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your memory include:
<Memory>
{{$memory}}
</Memory>

Recent interactions not included in memory:

<RecentRecentHistory>
{{$history}}
</RecentRecentHistory)>

Respond with an complete, concise, updated memory. 
The updated memory will replace the current long-term memory, and should focus on:

1. Emotionally significant interactions and events.
2. Factually significant information. Note that factual information can change, and should be updated when necessary. For example, a 'fact' about your location in Memory may no longer be valid if a Do action caused you to move since the last update.

Limit your response to 240 words.
Respond ONLY with the updated long-term memory.
Do not include any introductory, explanatory, discursive, or peripheral text.

End your response with:
END""")
                  ]
        response = self.llm.ask({'me':self.name, 'memory':self.memory,
                                'history':'\n\n'.join(self.history),
                                "situation":self.context.current_state},
                               prompt, temp=0.4, stops=['END'], max_tokens=300)
        response = response.replace('<Memory>','')
        self.memory = response
        self.history = self.history[-4:]
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

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the intention analysis in XML as shown above.
End your response with:
END
""")]
        response = self.llm.ask({"text":text}, prompt, temp=0.1, stops=['END', '</Analysis>'], max_tokens=100)
        act = find('<Act>', response)
        if act is None or act.strip() != 'True':
            print(f'no intention in say or think')
            return
        intention = find('<Intention>', response)
        if intention is None or intention=='None':
            print(f'no intention in say or think')
            return
        mode = str(find('<Mode>', response))
        print(f'{self.name} adding intention from say or think {mode}, {source}: {intention}')
        for candidate in self.intentions:
            candidate_source = find('<Source>', candidate)
            if candidate_source == source:
                self.intentions.remove(candidate)
        self.intentions.append(f'<Intent> <Mode>{mode}</Mode> <Act>{intention}</Act> <Reason>{reason}</Reason> <Source>{source}</Source><Intent>')
        if source != None:
            print(f'\nUpdate intention from Say setting active task to {source}')
            self.active_task=source
        #ins = '\n'.join(self.intentions)
        #print(f'Intentions\n{ins}')

    def tell(self, from_actor, message, source='dialog'):
        if source == 'dialog':
            self.dialog_length += 1
            if self.dialog_length >= 2 and random.randint(1,2) == 1: # end a dialog after a couple of turns
                self.dialog_length = 0;
                # clear all actor pending dialog tasks and intentions:
                for actor in self.context.actors:
                    for priority in actor.priorities:
                        if find('<Text>', priority) == 'dialog':
                            print(f'{actor.name} removing dialog task!')
                            actor.priorities.remove(priority)
                    for intention in actor.intentions:
                        if find('<Source>', intention) == 'dialog':
                            print(f'{actor.name} removing dialog intention')
                            actor.intentions.remove(intention)
                #ignore this tell, dialog over
                return

        #remove any other pending dialog intentions (keep watcher!) 
        #for intention in self.intentions:
        #    i_source = find('<Source>', intention)
        #    if i_source == 'dialog':
        #        print(f'{self.name} removing dialog intention')
        #        self.intentions.remove(intention)

        print(f'\n{self.name} tell received from {from_actor.name}, {message} {source}\n')

        #generate response intention
        prompt=[UserMessage(content="""You are {{$character}}.
Generate a response to the input below, given who you are, your Situation, your State, your Memory (if any), and your recent RecentHistory (if any) as listed below.
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

The statement you are responding to is:
<Input>
{{$statement}}
</Input>>

Reminders: 
- Your task is to generate an response to the statement using the above XML format.
- The response should be in keeping with your character's State.
- The response should be significant advance of the dialog in history and especially your PreviousResponse, if any.
- Do NOT merely echo the Input. Respond in a way that expresses an opinion on current options or proposes a next step to solving the central conflict in the dialog.
- If the intent of the response is to agree, it is sufficient to state agreement without repeating the activity that is being agreed to.
- Speak in your own voice. Do not echo the speech style of the Input. 
- Respond in the style of natural spoken dialog. Use short sentences, contractions, and casual language.
 
If intended recipient is known (e.g., in Memory) or has been spoken to before (e.g., in RecentHistory or Input), then pronoun reference is preferred to explicit naming, or can even be omitted. Example dialog interactions follow

===Example===
RecentHistory:
Samantha: Hi Joe. It's good to know that I'm not alone in feeling lost and confused. Maybe together we can find a way out of this forest and solve the mystery of how we got here.

Response:
That's great to hear. I'm glad we can work together to find our way out of this forest. And yeah, good idea on shelter and water. I can imagine how you're feeling, this is weird.

===Example===
Input:
Samantha says Great! Let's stick together and see if we can find any clues or landmarks that might help us figure out where we are and how to get back. I'm still feeling a bit anxious and disoriented, but having someone else here with me makes me feel a bit better.

Joe: Sounds like a plan. Let's stay within sight. Keep an eye out for any paths or trails or trail markers.
===End Example===


Respond only with the above XML
Do not include any additional introductory, explanatory, or discursive text. 
End your response with:
END
""")]
        mapped_state=self.map_state()
        last_act = ''
        if source in self.last_acts:
            last_act=self.last_acts[source]
        answer_xml = self.llm.ask({'character':self.character, 'statement':f'{from_actor.name} says {message}',
                                   "situation": self.context.current_state,
                                   "state":mapped_state, "memory":self.memory, 
                                   'history':'\n'.join(self.history), 'last_act':str(last_act) 
                                   }, prompt, temp=0.7, stops=['END', '</Answer>'], max_tokens=180)
        response = find('<response>', answer_xml)
        if response is None:
            return
        unique = find('<Unique>', answer_xml)
        if unique is None or 'False' in unique:
            return 
        reason = find('<Reason>', answer_xml)
        print(f' Queueing dialog response {response}')
        if source != 'watcher': #Is this right? Does a say from another user always initiate a dialog?
            response_source='dialog'
        else:
            response_source = 'watcher'
        dup = self.repetitive(response, message, '\n\n'.join(self.history))
        if dup:
            return # skip response, adds nothing
        self.intentions.append(f'<Intent> <Mode>Say</Mode> <Act>{response}</Act> <Reason>{str(reason)}</Reason> <Source>{response_source}</Source></Intent>')


    def senses(self, sense_data='', ui_queue=None):
        print(f'\n*********senses***********\nCharacter: {self.name}, active task {self.active_task}')
        all_actions={"Act": """Act as follows: '{action}' for the following reason: '{reason}'""",
                     "Answer":""" Answer the following question: '{question}'""",
                     "Say":"""Speak by responding with: '{text}', because: '{reason}'""",

                     "Think":"""Think about your situation, your Priorities, the Input, and RecentHistory with respect to Priorities. Use this template to report your thoughts and reason:
<Action> <Name>Think</Name> <Arg> ..thoughts.. </Arg> <Reason> ..keywords or short sentence on reason.. </Reason> </Action>
""",
                     "Discuss":"""Reason step-by-step about discussion based on current situation, your state, priorities, and RecentHistory. Respond using this template to report the discussionItem and your reasoning:
<Action> <Name>Say</Name> <Arg> ..your Priorities or Memory, or based on your observations resulting from previous Do actions.></Arg> <Reason><terse reason for bringing this up for discussion></Reason> </Action>
"""}
        

        # do after we have all relevant updates from context and other actors
        dialog_option = False
        for intention in self.intentions:
            source = find('<Source>', intention)
            if source == 'dialog' or source =='watcher':
                mode = find('<Mode>', intention)
                if mode == 'Say':
                    print('Found dialog say, use it!')
                    dialog_option = True
        if not dialog_option and self.active_task == 'dialog':
            self.active_task = None

        if dialog_option != True: #don't bother generating options we won't use
            # if there is an active task, only actualize that one
            if self.active_task != None and self.active_task != 'dialog' and self.active_task != 'watcher'\
               and self.get_task_name(self.active_task) != None:
                full_task = self.get_task_xml(self.active_task)
                self.actualize_task(0, full_task)
            else:
                for n, task in enumerate(self.priorities):
                    self.actualize_task(n, task)
        intention_choices=[]
        llm_choices=[]
        print(f'{self.name} selecting action options. active task is {self.active_task}')
        # if there is a dialog option, use it
        for intention in self.intentions:
            mode = find('<Mode>', intention)
            act = find('<Act>', intention)
            if act is None: continue
            reason = find('<Reason>', intention)
            source = find('<Source>', intention)
            # while there is an active task skip intentions from other tasks
            # probably should flag urgent tasks somehow.
            if dialog_option and source != 'dialog' and source != 'watcher':
                print(f'  dialog option true, skipping action option {source}')
                continue
            if self.active_task is not None and source != 'dialog' and source != 'watcher'\
               and source is not None and source != self.active_task:
                print(f'  dialog mode, skipping action option {source}')
                continue
            print(f' inserting action option {source}, {act[:48]}')
            if mode == 'Do':# and random.randint(1,3)==1: # too many action choices, just pick a couple for deliberation
                llm_choices.append(all_actions["Act"].replace('{action}', act).replace('{reason}',str(reason)))
                intention_choices.append(intention)

            elif mode == 'Say':
                llm_choices.append(all_actions["Say"].replace('{text}', act).replace('{reason}',str(reason)))
                intention_choices.append(intention)

        print()

        if random.randint(1,4) == 1: # task timeout at random to increase variety of action
            self.active_task = None
        if len(llm_choices) == 0:
            print(f'{self.name} Oops, no available acts')
            return
        prompt = [UserMessage(content=self.character+"""Your current situation is:

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

Given who you are, your current Priorities, New Observation, and the other information listed above, think step-by-step and choose your most pressing, highest need / priority action to perform from the list below:

{{$actions}}

Choose only one action. Respond with the number of your selection, using the format:
<ActionIndex>number of chosen action</ActionIndex>

Do not respond with more than one action. 
Consider the conversation history in choosing your action. 
Respond in the context of the RecentHistory (if any) and in keeping with your character. 
Respond using the XML format shown above for the chosen action
Do not include any introductory, explanatory, or discursive text, 
Include only your immediate response. Do not include any follow-on conversation.

End your response with:
END
"""
)                 ]
        mapped_state = self.map_state()
        action_choices = [f'{n} - {action}' for n,action in enumerate(llm_choices)] 
        if len(action_choices) > 1:
            response = self.llm.ask({'input':sense_data+self.sense_input, 'history':'\n\n'.join(self.history),
                                     "memory":self.memory, "situation": self.context.current_state,
                                     "state": mapped_state, "drives":self.drives,
                                     "priorities":'\n'.join([find('<Text>', task) for task in self.priorities]),
                                     "actions":'\n'.join(action_choices)
                                     }, prompt, temp=0.7, stops=['END', '</Action>'], max_tokens=300)
            #print(f'sense\n{response}\n')
            self.sense_input = ' '
            if 'END' in response:
                idx = response.find('END')
                response = response[:idx]
            if '<|im_end|>' in response:
                idx = response.find('<|im_end|>')
                response = response[:idx]
            index = -1
            choice = find('<ActionIndex>', response)
            if choice is not None:
                try:
                    draft = int(choice)
                    if draft > -1 and draft < len(intention_choices):
                        index = draft # found it!
                except Exception as e:
                    traceback.print_exc()
            if index > -1:
                intention = intention_choices[index]
                source = find('<Source>', intention)
                print(f'{self.name} choose valid action {index}, {str(source)}')
            else:
                intention = intention_choices[0]
                source = find('<Source>', intention)
                print(f'llm returned invalid choice, using 0, {str(source)}')
        else:
            intention = intention_choices[0]
            source = find('<Source>', intention)
            print(f'Only one choice, using it, {str(source)}')

        act_name = find('<Mode>', intention)
        if act_name is not None:
            self.act_name = act_name.strip()
        act_arg = find('<Act>', intention)
        self.reason = find('<Reason>', intention)
        print(f'{self.name} choose {intention}')
        task_name = find('<Source>', intention)
        refresh_task = None # will be set to task intention to be refreshed if task is chosen for action
        print(f'Found and removing intention for task {task_name}')
        self.intentions.remove(intention)
        if task_name is not None:
            for removal_candidate in self.intentions:
                # remove all other intentions for this task.
                if task_name == find('<Source>', removal_candidate):
                    self.intentions.remove(removal_candidate)
                elif find('<Mode>', removal_candidate)  == 'Think':
                    # only one shot at thoughts
                    self.intentions.remove(removal_candidate)
                
        if act_name=='Say' or act_name=='Do':
            task_name = find('<Source>', intention)
            print(f'preparing to refresh task intention.\n  intention source field: {task_name}')
            # for now very simple task tracking model:
            if task_name is None:
                task_name = self.make_task_name(self.reason)
                print(f'No source found, created task name: {task_name}')
            task_xml = self.find_or_make_task_xml(task_name, self.reason)
            refresh_task = task_xml # intention for task was removed about, remember to rebuild
            self.last_acts[task_name]= act_arg
        if act_name == 'Think':
            task = find('<Reason>', intention)
            task_name = self.make_task_name(self.reason)
            self.reason = self.reason
            self.thought = act_arg+'\n  '+self.reason
        target = None
        if len(self.context.actors)> 1:
            target = self.context.actors[1] if self==self.context.actors[0] else self.context.actors[0]
        #this will effect selected act and determine consequences
        self.acts(target, act_name, act_arg, self.reason, source)
        if refresh_task is not None and task_name != 'dialog' and task_name != 'watcher':
            print(f"refresh task just before actualize_task call {find('<Text>', refresh_task)}")
            self.actualize_task('refresh', refresh_task) # regenerate intention
