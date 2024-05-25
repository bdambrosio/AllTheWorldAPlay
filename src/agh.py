import os
import time
import random
import traceback
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
import llm_api


def findall(key,  form):
    """ find multiple occurences of an xml field in a string """
    idx = 0
    items = []
    forml = form.lower()
    keyl = key.lower()
    keyle = keyl[0]+'/'+keyl[1:]
    while idx < len(forml):
        start_idx = forml[idx:].find(keyl)+len(keyl)
        if start_idx < 0:
            return items
        end_idx = forml[idx+start_idx:].find(keyle)
        if end_idx < 0:
            return items
        items.append(form[idx+start_idx:idx+start_idx+end_idx].strip())
        idx += start_idx+end_idx
    return items

def find(key,  form):
    """ find first occurences of an xml field in a string """
    idx = 0
    forml = form.lower()
    keyl = key.lower()
    keyle = keyl[0]+'/'+keyl[1:]
    if keyl not in forml:
        return None
    start_idx = forml[idx:].find(keyl)+len(keyl)
    if start_idx < 0:
        return None
    end_idx = forml[idx+start_idx:].find(keyle)
    if end_idx < 0:
        return form[start_idx:] 
    idx += start_idx+end_idx
    return form[start_idx: start_idx+end_idx]



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
        
    def do(self, actor, action):
        prompt=[SystemMessage(content="""You are a dynamic world. Your task is to determine the result of {{$name}} performing

<Action>
{{$action}}
</Action>

in the current situation: 

<Situation>
{{$state}}
</Situation>

Respond with the observable result for {{$name}}.
Respond ONLY with the observable result as a simple declarative statement.
Include in the statement any available sensory input (e.g. {{$name}} sees/hears/feels,...) as well as any change is {{$name})'s or other actor's state (e.g., {{$name}) now has...).
Respond ONLY with the immediate effect of the specified action. Do NOT extend the scenario with any follow on actions or effects.
Do not include any Introductory, explanatory, or discursive text.
End your reponse with:
<END>
""")]
        history = self.history()
        response = self.llm.ask({"name":actor.name, "action":action,
                                "state":self.current_state}, prompt, temp=0.7, stops=['<END'], max_tokens=200)
        return response

    def senses (self, input='', ui_task_queue=None):
        # since at the moment there are only two chars, each with complete dialog, we can take it from either.
        # not really true, since each is ignorant of the others thoughts
        history = self.history()
        if random.randint(1,7) == 1:
            event = """
Include a random event such as the malfunction of a device or bot or arrival of an email to process. 
Examples:
- the bot operating the vacuum cleaner unexpectedly stops moving. 
- annie receives an email directed to her personally from an unknown agent.
            """
        else:
            event = ""

        prompt = [SystemMessage(content="""You are a dynamic world. Your task is to update your state description. 
Include day/night cycles and weather patterns. 
Update location and other physical environment characteristics as indicated in the History.
Your response should be concise, and only include only an update of the physical environment.
        """ + event + """
Your state description should be dispassionate, 
and should begin with a brief description of the current physical space suitable for a text-to-image generator. 
That is, advance the environment by approximately 3 hours. The state as of 3 hours ago was:

<PreviousState>
{{$state}}
</PreviousState> 

In the interim, the characters in the world had the following interactions:

<History>
{{$history}}
</History>

Respond using the following XML format:

<State>
Sentence describing physical space, suitable for image generator,
Updated State description of about 200 words
</State>

Respond with an updated environment description reflecting a time passage of 3 hours and the events that have occurred.
Include ONLY the concise updated state description in your response. Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
Do NOT carry the storyline beyond three hours from PreviousState.
Limit your response to about 200 words
End your response with:
END""")]
            
        response = self.llm.ask({"state":self.current_state, 'history':history}, prompt, temp=0.6, stops=['END'], max_tokens=360)
        new_situation = find('<State>', response)
        if new_situation is not None:
            self.current_state=new_situation
        for actor in self.actors:
            actor.forward(3) # forward three hours and update history, etc
        return response

class Character():
    def __init__ (self, name, character_description):
        self.name = name
        self.speaker = None
        self.character = character_description
        self.history = []
        self.memory = 'None'
        self.context=None
        self.priorities = [''] # will be displayed by main thread at top in character intentions text widget
        self.reason='None' # to be displayed by main thread in character main text widget
        self.show='' # to be displayed by main thread in UI public (main) text widget
        self.reflect_elapsed = 99
        self.reflect_interval = 3
        self.physical_state = "Fear: Low, Thirst: Low, Hunger: Low, Fatigue: Low, Illness: Low, MentalState: alert"
        self.intentions = []
        self.previous_action = ''
        self.sense_input = ''
        self.drives = """
 - immediate physiological needs: survival, water, food, clothing, shelter, rest.  
 - safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events. 
 - assurance of short-term future physiological needs (e.g. adequate water and food supplies, shelter maintenance). 
 - love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friendship, intimacy, trust, acceptance.
"""
        self.widget = None
        self.active_task = None # task character is actively pursuing.
        self.act_result = ''
        self.last_acts = {} # a set of priorities for which actions have been started, and their states.
        # Waiting - Waiting for input, InputPending, OutputPending - say intention pending
        self.dialog_status = 'Waiting' # Waiting, Pending

    def initialize(self):
        """called from worldsim once everything is set up"""
        self.update_priorities()

    def add_to_history(self, role, act, message):
        message = message.replace('\\','')
        self.history.append(f"{role}: {act} {message}")
        self.history = self.history[-3:] # memory is fleeting, otherwise we get very repetitive behavior

    def synonym_check(self, term, candidate):
        """ except for new tasks, we'll always have correct task_name, and new tasks are presumably new"""
        if term == candidate: return True
        else: return False
        instruction=[SystemMessage(content="""Your task is to decide if Text1 and Text2 designate the same task.
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
        response = self.llm.ask({"text1":term, "text2":candidate}, instruction, temp=0.3, stops=['</END>'], max_tokens=200)
        if 'true' in response.lower():
            return True
        else:
            return False

    def get_task(self, term):
        for task in list(self.last_acts.keys()):
            match=self.synonym_check(task, term)
            if match:
                return task
        return None
            
    def set_task_last_act(self, term, act):
        # pbly don't need this, at set we have canonical task
        task = self.get_task(term)
        if task == None:
            print(f'\nset_task_last_act {self.name} no match found for term: {term}, {act}\n')
            self.last_acts[term] = act
        else:
            print(f'\nset_task_last_act {self.name} match found: term {term}, task {task}\n  {act}')
            self.last_acts[task] = act

    def get_task_last_act(self, term):
        task = self.get_task(term)
        if task == None:
            print(f'task_last_act {self.name} no match found: term {term}')
            return 'None'
        else:
            print(f'task_last_act match found {self.name} term {term} task {task}\n   act:{self.last_acts[task]}')
            return self.last_acts[task]

    def make_task_name(self, reason):
        instruction=[SystemMessage(content="""Generate a concise, 2-5 word task name from the motivation to act provided below.
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
                    
    def find_or_make_task_xml(self, task_name, reason):
        for candidate in self.priorities:
            #print(f'find_or_make testing\n {candidate}\nfor name {task_name}')
            if task_name == find('<Text>', candidate):
                print(f'found existing task\n   {candidate}')
                return candidate
        new_task = f'<Priority><Text>{task_name}</Text><Reason>{reason}</Reason></Priority>'
        self.priorities.append(new_task)
        print(f'created new task to reflect {task_name}\n {reason}\n  {new_task}')
        return new_task

    def update_priorities(self):
        self.active_task = None
        prompt = [SystemMessage(content=self.character+"""You are {{$character}}.
Your basic drives include:
<Drives>
{{$drives}}
</Drives>
 
Your task is to create a set of three short term goals given who you are, your Situation, your PhysicalState, your Memory, and your recent RecentHistory as listed below.
Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your physical state is:

<PhysicalState>
{{$physState}}
</PhysicalState>

Your memories include:

<Memory>
{{$memory}}
</Memory>

Recent conversation has been:
<RecentHistory>
{{$history}}
</RecentHistory)>

The mapping between the above and short-term goals is complex, but follows natural behavior. For example:
- if situation mentions cold, dark, or stormy, then the physiological need for 'shelter' would likely generate shelter-seeking goals.
- if physicalState Fear is High, then the drive for 'safety from threats' might generate weapon, shelter, or companionship goals.
- if RecentHistory includes thoughts of being alone or loneliness or fear, then drive for love or belonging might generate communication, friendship, intimacy, or acceptance related goals.
- if physicalState Thirst is High, then the physiological need for water is active and would likely generate a water drink goal, if water is available in the Situation, or a water search goal otherwise.

Reminder: Your task is to create a set of three short term goals, derived from your priorities, given the Situation, your PhysicalState, your Memory, and your recent RecentHistory as listed above.

List your three most important short term goals as instantiations from: 

{{$drives}}
            
A short term goal is important when the drive it corresponds to is unmet. 
List ONLY your most important priorities as simple declarative statements, without any introductory, explanatory, or discursive text.
Your priorities should be as specific as possible. 
For example, if you need sleep, say 'Sleep', not 'Physiological need'.
Similarly, if your safety is threatened by a wolf, respond "Safety from wolf", not merely "Safety"
limit your response to 120 words. 

Use the XML format:
<Priorities>
<Priority> <Text>statement of top priority</Text> <Reason>Situation element, physical state, Memory element, or RecentHistory element that motivates this</Reason> </Priority>
<Priority> <Text>statement of second priority</Text> <Reason>Situation element, physical state, Memory element, or RecentHistory element that motivates this</Reason> </Priority>
<Priority> <Text>statement of third priority</Text> <Reason>Situation element, physical state, Memory element, or RecentHistory element that motivates this</Reason> </Priority>
</Priorities>

Respond ONLY with the above XML. Do not include any introductory, explanatory, or discursive text.
End your response with:
END
""")]
        response = self.llm.ask({'character':self.character, 'goals':'\n'.join(self.priorities),
                                 'drives':self.drives, 'memory':self.memory,
                                'history':'\n\n'.join(self.history),
                                "situation":self.context.current_state,
                                "physState":self.physical_state,
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
            if len(items) > 0:
                # old intentions need to go away, except for pending answers!
                for intention in self.intentions:
                    source = find('<Source>', intention)
                    if source != 'dialog':
                        self.intentions.remove(intention)
            for n, task in enumerate(items):
                self.priorities.append(task)
                self.actualize_task(n, task)
        except Exception as e:
            traceback.print_exc()
        #print(f'\n-----Done-----\n\n\n')

    def actualize_task(self, n, task_xml):
        task_name = find('<Text>', task_xml)
        print(f'\n Actualizing task {n} {task_name}')
        last_act = self.get_task_last_act(task_name)
        reason = find('<Reason>', task_xml)
        if reason is not None and ': Low' in reason:
            if self.active_task == task_name:
                #active task is now satisfied!
                self.active_task = None
            return

        prompt = [SystemMessage(content="""You are {{$character}}.
Your task is to act in response to the following perceived task:

<Task>
{{$task}} given {{$reason}}
</Task>

Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your physical state is:

<PhysicalState>
{{$physState}}
</PhysicalState>

Your memories include:

<Memory>
{{$memory}}
</Memory>

Recent conversation has been:
<RecentHistory>
{{$history}}
</RecentHistory)>

The Previous specific act for this Task, if any, was:
{{$lastAct}}

And the observed result of that was:
<ObservedResult>
{{$lastActResult}}.
</ObservedResult>

Respond with a specific act that can advance the task listed earlier. 
A specific action is one which:

- Can be described in terms of specific physical movements or steps
- Has a clear beginning and end point
- Can be performed or acted out by a person
- Can be easily visualized or imagined as a film clip
- Is either a natural follow-on action to the previous specific act (if any), given the observed result (if any), or an alternate line of action. Especially consider this second alternative when the observed result does not indicate progress on the Task. 


Respond in XML:
<Actionable>
  <Mode>'Say' or 'Do', corresponding to whether the act is a speech act or a physical act</Mode>
  <SpecificAct>words to speak or specific action description</SpecificAct>
</Actionable>

===Examples===

Task:
'Establish connection with Joe given RecentHistory element: "Who is this guy?"'

Response:
<Actionable>
  <Mode>Say</Mode>
  <SpecificAct>Say hello to Joe.</SpecificAct>
</Actionable>

Task:
'Find out where I am given Situation element: "This is very very strange. Where am I?"'

Response:
<Actionable>
  <Mode>Do</Mode>
  <SpecificAct>You start to look around for any landmarks or signs of civilization, hoping to find something familiar that might give you a clue as to your whereabouts.</SpecificAct>
</Actionable>

===End Examples===

Use the XML format:

<Actionable> <SpecificAct>statement of specific action</SpecificAct> </Actionable>

Respond ONLY with the above XML.
The task you are to transform into a specific action is:

<Task>
{{$task}} given {{$reason}}
</Task>

Ensure you do not duplicate previous specific act.
Do not include any introductory, explanatory, or discursive text.
End your response with:
<END>"""
                                        )]
        print(f'{self.name} act_result: {self.act_result}')
        act= None; tries=0
        while act is None and tries < 2:
            response = self.llm.ask({'character':self.character, 'goals':'\n'.join(self.priorities), 'memory':self.memory,
                                     'history':'\n\n'.join(self.history),
                                     "situation":self.context.current_state,
                                     "physState":self.physical_state, "task":task_name, "reason":reason,
                                     "lastAct": last_act, "lastActResult": self.act_result
                                     },
                                    prompt, temp=0.6, stops=['</Actionable>','<END>'], max_tokens=180)
            act = find('<SpecificAct>', response)
            mode = find('<Mode>', response)
            if mode is None:
                mode = 'Do'
            tries += 1
            
        if act is not None:
            print(f'actionable found: task_name {task_name}\n  {act}')  
            self.intentions.append(f'<Intent> <Mode>{mode}</Mode> <Act>{act}</Act> <Reason>{reason}</Reason> <Source>{task_name}</Source> </Intent>')
            #ins = '\n'.join(self.intentions)
            #print(f'Intentions\n{ins}')
                                    
    def update_physical_state(self, key, response):
        new_state = find('<'+key+'>', response)
        if new_state != None and len(new_state)> 0:
            #state values can't have commas
            new_state = new_state.replace(',',' ')
            # Split the string by commas to get individual key-value pairs
            parts = self.physical_state.split(', ')
    
            # Iterate through the parts and update the value for the given key
            updated_parts = []
            for part in parts:
                if part.startswith(key + ':'):
                    updated_parts.append(f"{key}: {new_state}")
                else:
                    updated_parts.append(part)
    
            # Join the parts back together with commas
            self.physical_state = ', '.join(updated_parts)

    def forward(self, num_hours):
        # roll conversation history forward.
        ## update physical state
        prompt = [SystemMessage(content=self.character+"""Your name is {{$me}}.
When last updated, your physical state was:

<PhysicalState>
{{$physState}}
</PhysicalState>

Your task is to update your physical state.
3 Hours have past since you last updated your physical state.
Your current situation is:

<Situation>
{{$situation}}
</Situation>

Recent interactions not included in memory:

<RecentHistory>
{{$history}}
</RecentHistory)>

Respond with an updated physical state, using this XML format:

<PhysicalState>
<Fear>Low, Moderate, Medium, High, Urget</Fear>
<Thirst>Low, Moderate, Medium, High, Urgent</Thirst>
<Hunger>Low, Moderate Medium, High, Urgent</Hunger>
<Fatigue>Low, Moderate, Medium, High, Urgent</Fatigue>
<Illness>Low, Moderate, Medium, High, Urgent</Illness>
<MentalState>2-4 world describing mental state</MentalState>
</PhysicalState>

The updated physical state should reflect updates from the previous state based on passage of time, recent events in Memory, and recent history

- Fear - increases with perception of threat, decreases with removal of threat.
- Thirst - increases as time passes since last drink, increases with heat and exertion.
- Hunger - increases as time passes since last food, increases with exertion.
- Fatigue - increases as time passes since last rest, increases with heat and exertion.
- Illness  - increases with injury or illness, decreases with rest and healing.
- Mental State - one to four words on mental state, e.g. 'groggy', 'alert', 'confused and lost', etc.

Respond ONLY with the updated state.
Do not include any introductory or peripheral text.
limit your response to 120 words at most.
End your response with:
END""")
                  ]
        response = self.llm.ask({'me':self.name, 'memory':self.memory,
                                'history':'\n\n'.join(self.history),
                                "situation":self.context.current_state,
                                "physState":self.physical_state
                                },
                               prompt, temp=0.2, stops=['END'], max_tokens=180)
        self.update_physical_state('Fear', response)
        self.update_physical_state('Thirst', response)
        self.update_physical_state('Hunger', response)
        self.update_physical_state('Fatigue', response)
        self.update_physical_state('Illness', response)
        self.update_physical_state('MentalState', response)

        ## update long-term dialog memory
        prompt = [SystemMessage(content=self.character+"""Your name is {{$me}}.
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
        self.update_priorities()
        
    def update_intentions_wrt_say_think(self, source, text, reason):
        # determine if text implies an intention to act, and create a formatted intention if so
        print(f'Update intentions from say or think\n {text}\n{reason}')

        prompt=[SystemMessage(content="""Your task is to analyze the following text.

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
        if act is None: return
        intention = find('<Intention>', response)
        if intention is None: return
        mode = str(find('<Mode>', response))
        print(f'{self.name} adding intention from say or think {mode}, {source}: {intention}')
        self.intentions.append(f'<Intent> <Mode>{mode}</Mode> <Act>{intention}</Act> <Reason>{reason}</Reason> <Source>{source}</Source><Intent>')
        ins = '\n'.join(self.intentions)
        print(f'Intentions\n{ins}')

    def tell(self, actor, message):
        actor.sense_input += self.name + ' says '+message
        if '?' in message:
            #question, formulate response
            prompt=[SystemMessage(content="""You are {{$character}}.
Generate a response to the question below, given who you are, your Situation, your PhysicalState, your Memory, and your recent RecentHistory as listed below.
Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your physical state is:

<PhysicalState>
{{$physState}}
</PhysicalState>

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
<Response>answer to question</Response>
<Reason>reason for this answer</Reason>
</Answer>

The question is:
<Question>
{{$question}}
</Question>>

Reminder: Your task is to generate an answer to the question using the above XML format.
Respond only with the above XML
Do not include any additional introductory, explanatory, or discursive text. 
End your response with:
END
""")]
            answer_xml = self.llm.ask({'character':self.character, 'question':message, "situation": self.context.current_state,
                                       "physState":self.physical_state, "memory":self.memory, 
                                       'history':'\n'.join(self.history) 
                                       }, prompt, temp=0.7, stops=['END', '</Answer>'], max_tokens=180)
            print(f'tell {answer_xml}')
            response = find('<response>', answer_xml)
            if response is None:
                return
            reason = find('<reason>', answer_xml)
            print(f'Question received {message}\n  response queued {response}')
            self.intentions.append(f'<Intent> <Mode>Say</Mode> <Act>{response}</Act> <Reason>{str(reason)}</Reason> <Source>dialog</Source></Intent>')

    def acts(self, target, act_name, act_arg='', reason='', source=''):
        #
        ### speak to someone
        #
        show = '' # widget window
        self.show = ''
        self.reason = reason
        if act_name is not None and act_arg is not None and len(act_name) >0 and len(act_arg) > 0:
            verb = 'says' if act_name == 'Say' else ''
            #self.ui.display('\n**********************\n')
            if act_name=='Say' or act_name == 'Do':
                self.add_to_history('You', act_name , act_arg+f'\n  why: {reason}')
                for actor in self.context.actors:
                    if actor != self: # everyone else sees/hears your act!
                        actor.add_to_history(self.name, '' , act_arg)
                        #print(f'adding to {actor.name} history: {act_arg}')
                # target has special opportunity to respond - tbd
                if target is not None:
                    target.sense_input += '\n'+self.name+' '+act_name+': '+act_arg
                else:
                    for actor in self.context.actors:
                        if actor != self:
                            actor.sense_input = '\n'+self.name+' says '+act_arg
                            if act_name == 'Say':
                                actor.tell(self.name, act_arg)
                                
                #self.show goes in actor 'value' pane
                self.show = '\n'+self.name+' '+verb + ": "+act_arg
                if act_name =='Do':
                    #can we link back to task?
                    result = self.context.do(self, act_arg)
                    self.show += '\n  '+result # main window
                    self.add_to_history('You', 'observe', result)
                    print(f'{self.name} setting act_result to {result}')
                    self.act_result = result
                    if target is not None: # this is wrong, world should update who sees do
                        target.sense_input += '\n'+result
            else:
                self.show = 'Seems to be thinking ...'
                text = str(act_arg)
                self.add_to_history('You', 'think', text+'\n  '+reason)
                self.show = '\n'+self.name+': Thinking'
                for actor in self.context.actors:
                    if actor != self: # everyone else sees/hears your act!
                        actor.add_to_history('You', 'see', f'{self.name} thinking')
            self.previous_action = act_name

            #if random.randint(1,3) == 1: # not too often, makes action to jumpy
            #    self.priorities = []
            #    self.update_priorities() # or should we do this at sense input? 
            if act_name == 'Say' or act_name == 'Think':
                self.update_intentions_wrt_say_think(act_arg, reason, )
                ins = '\n'.join(self.intentions)
                print(f'Intentions\n{ins}')
            

    def senses(self, input='', ui_queue=None):
        print(f'\n*********senses***********\nCharacter: {self.name}, active task {self.active_task}')
        all_actions={"Act": """Act as follows: '{action}' for the following reason: '{reason}'""",
                     "Answer":""" Answer the following question: '{question}'""",
                     "Say":"""Speak by responding with: '{text}', because: '{reason}'""",

                     "Think":"""Think about your situation, your Priorities, the Input, and RecentHistory with respect to Priorities. Use this template to report your thoughts and reason:
<Action> <Name>Think</Name> <Arg> ..thoughts.. </Arg> <Reason> ..reason.. </Reason> </Action>
""",
                     "Discuss":"""Reason step-by-step about discussion based on current situation, your PhysicalState, priorities, and RecentHistory. Respond using this template to report the discussionItem and your reasoning:
<Action> <Name>Say</Name> <Arg> ..your Priorities or Memory, or based on your observations resulting from previous Do actions.></Arg> <Reason><reasons for bringing this up for discussion></Reason> </Action>
"""}
        

        intention_choices=[]
        llm_choices=[]
        print(f'{self.name} selecting action options. active task is {self.active_task}')
        dialog_option = False
        # if there is a dialog option, use it
        for intention in self.intentions:
            source = find('<Source>', intention)
            if source == 'dialog':
                dialog_option = True
        for intention in self.intentions:
            mode = find('<Mode>', intention)
            act = find('<Act>', intention)
            if act is None: continue
            reason = find('<Reason>', intention)
            source = find('<Source>', intention)
            # while there is an active task skip intentions from other tasks
            # probably should flag urgent tasks somehow.
            if dialog_option and source != 'dialog':
                print(f'  skipping action option {source}')
                continue
            if self.active_task is not None and source != 'dialog' and source is not None and source != self.active_task:
                print(f'  skipping action option {source}')
                continue
            print(f' inserting action option {source}, {act[:48]}')
            if mode == 'Do':# and random.randint(1,3)==1: # too many action choices, just pick a couple for deliberation
                llm_choices.append(all_actions["Act"].replace('{action}', act).replace('{reason}',str(reason)))
                intention_choices.append(intention)

            elif mode == 'Say':
                llm_choices.append(all_actions["Say"].replace('{text}', act).replace('{reason}',str(reason)))
                intention_choices.append(intention)

        #if self.previous_action != 'Think' or random.randint(1,3) == 1: # you think too much
            #    llm_choices.append(all_actions['Think'])
            #intention_choices.append(intention)
            #if len(llm_choices) < 3 or random.randint(1,2) == 1: # you talk too much
            #llm_choices.append(all_actions['Discuss'])
            #intention_choices.append(intention)
        #if len(llm_choices) == 0:
        #    print(f' inserting action option Think, nothing else available')
        #    llm_choices.append(all_actions['Think'])
        #intention_choices.append(intention)
        print()

        #if random.randint(1,4) == 1: # task timeout
        #    self.active_task = None
        if len(llm_choices) == 0:
            print(f'{self.name} Oops, no available acts')
            return
        prompt = [SystemMessage(content=self.character+"""Your current situation is:

<Situation>
{{$situation}}
</Situation>

Your physical state is:

<PhysicalState>
{{$physState}}
</PhysicalState>

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

Given your current Priorities, New Observation, and the other information listed above, think step-by-step and choose one action to perform from the list below:

{{$actions}}

Choose only one action. Respond with the number of your selection, using the format:
<ActionIndex>number of chosen action</ActionIndex>

Do not respond with more than one action 
Consider the conversation history in choosing your action. 
Respond in the context of the RecentHistory (if any) and in keeping with your character. 
Respond using the XML format shown above for the chosen action
Do not include any introductory, explanatory, or discursive text, 
Include only your immediate response. Do not include any follow-on conversation.

End your response with:
END
"""
)                 ]

        action_choices = [f'{n} - {action}' for n,action in enumerate(llm_choices)] 
        if len(action_choices) > 1:
            response = self.llm.ask({'input':input+self.sense_input, 'history':'\n\n'.join(self.history),
                                     "memory":self.memory, "situation": self.context.current_state,
                                     "physState":self.physical_state,
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
                print(f'{self.name} choose valid action')
                intention = intention_choices[index]
            else:
                print(f'llm returned invalid choice, using 0')
                intention = intention_choices[0]
        else:
            intention = intention_choices[0]

        act_name = find('<Mode>', intention)
        if act_name is not None:
            self.act_name = act_name.strip()
        act_arg = find('<Act>', intention)
        self.reason = find('<Reason>', intention)
        print(f'{self.name} choose {intention}')
        refresh_task = None # will be set to task intention to be refreshed if task is chosen for action
        #print(f'Found and removing intention for act {intention}')
        self.intentions.remove(intention)
        
        ins = '\n'.join(self.intentions)
        #print(f'Remaining Intentions\n{ins}\n')
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
            if task_name != 'dialog':
                self.active_task = task_name
        if act_name == 'Think':
            task = find('<Reason>', intention)
            task_name = self.make_task_name(self.reason)
            self.reason = act_arg+'\n  '+self.reason
        target = None
        if len(self.context.actors)> 1:
            target = self.context.actors[1] if self==self.context.actors[0] else self.context.actors[0]
        #this will effect selected act and determine consequences
        self.acts(target, act_name, act_arg, self.reason, source)
        if refresh_task is not None:
            print(f"refresh task just before actualize_task call {find('<Text>', refresh_task)}")
            self.actualize_task('refresh', refresh_task) # regenerate intention
