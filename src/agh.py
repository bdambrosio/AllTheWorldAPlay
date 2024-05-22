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
    """ find multiple occurences of an xml field in a string """
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
Do not include any Introductory, explanatory, or discursive text.
Respond ONLY with the immediate effect of the specified action. Do NOT extend the scenario with any follow on actions or effects.
End your reponse with:
<END>
""")]
        history = self.history()
        response = self.llm.ask({"name":actor.name, "action":action,
                                "state":self.current_state}, prompt, temp=0.7, stops=['<END'], max_tokens=200)
        #self.current_state += '\n'+response
        return response

    def senses (self, input='', ui_task_queue=None):
        #first step is to compute current conversation history.
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
        self.reasoning='None' # to be displayed by main thread in character main text widget
        self.show='' # to be displayed by main thread in UI public (main) text widget
        self.reflect_elapsed = 99
        self.reflect_interval = 3
        self.physical_state = "Fear: Low, Thirst:Low, Hunger: Low, Fatigue: Low, Health: High, MentalState: alert"
        self.intentions = []
        self.previous_action = ''
        self.sense_input = ''
        self.drives = """
 - immediate physiological needs including survival, water, food, clothing, shelter, sleep  
 - safety from threats including ill-health or physical threats from unknown or adversarial actors or adverse events. 
 - assurance of short-term future physiological needs (e.g. adequate water and food supplies, shelter maintenance). 
 - love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friedship, intimacy, trust, acceptance.
"""
        self.widget = None
        
    def add_to_history(self, role, act, message):
        message = message.replace('\\','')
        self.history.append(f"{role}: {act} {message}")
        self.history = self.history[-3:] # memory is fleeting, otherwise we get very repetitive behavior

    def suggest_priorities(self):
        prompt = [SystemMessage(content=self.character+"""You are {{$character}}.
Your basic drives include:
<Drives>
{{$drives}}
</Drives>
 
Your task is to create a set of three short term goals given who you are,  your Situation, your PhysicalState, your Memory, and your recent RecentHistory as listed below.
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

Reminder: Your task is to create a set of three short term goals, derived from your priorities, given the Situation, your PhysicalState, your Memory, and your recent RecentHistory as listed above.

List your three most important short term goals as instantiations from: 

{{$drives}}
            
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
            print(f'\nSuggested priorities:')
            self.priorities = []
            for n, priority in enumerate(items):
                print(f'\n Actualizing priority (n) {priority}')
                task = find('<Text>', priority)
                reason = find('<Reason>', priority)
                self.priorities.append(task)
                prompt = [SystemMessage(content="""You are {{$character}}.
Your task is to act in response to the following perceived priority:

<Priority>
{{$task}} given {{$reason}}
</Priority>

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

Respond with a specific act that will respond to the priority listed earlier. 
A specific action is one which:

- Can be described in terms of specific physical movements or steps
- Has a clear beginning and end point
- Can be performed or acted out by a person
- Can be easily visualized or imagined as a film clip

Respond in XML:
<Actionable>
  <Mode>'Say' or 'Do', corresponding to whether the act is a speech act or a physical act</Mode>
  <SpecificAct>words to speak or specific action description</SpecificAct>
</Actionable>

===Examples===

Priority:
'Establish connection with Joe given RecentHistory element: "Who is this guy?"'

Response:
<Actionable>
  <Mode>Say</Mode>
  <SpecificAct>Say hello to Joe.</SpecificAct>
</Actionable>

Priority:
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

Do not include any introductory, explanatory, or discursive text.
End your response with:
<END>"""
                                        )]
                response = self.llm.ask({'character':self.character, 'goals':'\n'.join(self.priorities), 'memory':self.memory,
                                         'history':'\n\n'.join(self.history),
                                         "situation":self.context.current_state,
                                         "physState":self.physical_state, "task":task, "reason":reason
                                         },
                                        prompt, temp=0.6, stops=['</Actionable>','<END>'], max_tokens=180)
                act = find('<SpecificAct>', response)
                mode = find('<Mode>', response)
                if mode is not None:
                    print(f' actionable found: {act}')
                    self.intentions.append(f'<Intent> <Mode>{mode}</Mode> <Act>{act}</Act> <Reasoning>{reason}</Reasoning> <Intent>')
                    
        except Exception as e:
            traceback.print_exc()
        print(f'\n-----Done-----\n\n\n')
                                    
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

<RecentRecentHistory>
{{$history}}
</RecentRecentHistory)>

Respond with an updated physical state, using this XML format:

<PhysicalState>
<Fear>Low, Medium, High</Fear>
<Thirst>Low, Medium, High</Thirst>
<Hunger>Low, Medium, High</Hunger>
<Fatigue>Low, Medium, High</Fatigue>
<Health>Low, Medium, High</Health>
<MentalState>2-6 world describing mental state</MentalState>
</PhysicalState>

The updated physical state should focus on:

- Level of Fear - increases with perception of threat, decreases with removal of threat.
- Level of Thirst - increases as time passes since last drink, increases with heat and exertion.
- Level of Hunger - increases as time passes since last food, increases with exertion.
- Level of Fatigue - increases as time passes since last rest, increases with heat and exertion.
- Level of Health  - decreases with injury or illness, increases with rest and healing.
- Mental State - one or two words on mental state, e.g. 'groggy', 'alert', 'confused and lost', etc.

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
        self.update_physical_state('Health', response)
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

    def update_intentions_wrt_say_think(self, text, reasoning):
        # determine if text implies an intention to act, and create a formatted intention if so
        print(f'Update intentions from say or think\n {text}\n{reasoning}')

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
        print(f'{self.name} adding intention {mode}: {intention}')
        self.intentions.append(f'<Intent> <Mode>{mode}</Mode> <Act>{intention}</Act> <Reasoning>{reasoning}</Reasoning> <Intent>')
            
    def acts(self, target, act_name, act_arg='', reasoning=''):
        #
        ### speak to someone
        #
        show = '' # widget window
        self.show = ''
        self.reasoning = reasoning
        if act_name is not None and act_arg is not None and len(act_name) >0 and len(act_arg) > 0:
            #self.ui.display('\n**********************\n')
            if act_name=='Say' or act_name == 'Do':
                self.add_to_history('You', act_name , act_arg+f'\n  why: {reasoning}')
                for actor in self.context.actors:
                    if actor != self: # everyone else sees/hears your act!
                        verb = 'says' if act_name == 'Say' else ''
                        actor.add_to_history(self.name, '' , act_arg)
                        #print(f'adding to {actor.name} history: {act_arg}')
                # target has special opportunity to respond - tbd
                if target is not None:
                    target.sense_input = '\n'+self.name+' '+act_name+': '+act_arg

                #self.show goes in actor 'value' pane
                self.show = '\n'+self.name+' '+verb + ": "+act_arg
                if act_name =='Do':
                    self.intentions = [] # maybe we should clear regardless of act?
                    result = self.context.do(self, act_arg)
                    self.show += '\n  observes: '+result # main window
                    self.add_to_history('You', 'observe', result)
                    if target is not None: # this is wrong, world should update who sees do
                        target.sense_input += '\n'+result
                 
            else:
                self.show = 'Seems to be thinking ...'
                text = str(act_arg)
                self.add_to_history('You', 'think', text+'\n  '+reasoning)
                self.show = '\n'+self.name+': Thinking'
                for actor in self.context.actors:
                    if actor != self: # everyone else sees/hears your act!
                        actor.add_to_history('You', 'see', f'{self.name} thinking')
            self.previous_action = act_name

            self.priorities = []
            self.suggest_priorities() # or should we do this at sense input? 
            if act_name == 'Say' or act_name == 'Think':
                self.update_intentions_wrt_say_think(act_arg, reasoning)
            

    def senses(self, input='', ui_queue=None):
        #print(f'\n********************ask*********************\nSpeaker: {speaker}, Input: {input}')
        all_actions={"Act": """Do in the world by responding with:
<Action> <Name>Do</Name> <Arg>{action}</Arg> <Reasoning>{reason}</Reasoning> </Action>
""",
                     "Answer":f"""If the new Observation contains a question, answer it using this form:
<Action> <Name>Say</Name> <Arg><answer to question from other actor></Arg> <Reasoning><reasons for this answer></Reasoning> </Action>
""",
                     "Say":"""Speak by responding with:
<Action> <Name>Say</Name> <Arg>{text}</Arg> <Reasoning>{reasson{</Reasoning> </Action>
""",
                     "Think":"""Think about your situation, your Priorities, the Input, and RecentHistory with respect to Priorities. Use this template to report your thoughts:
<Action> <Name>Think</Name> <Arg><thoughts on situation></Arg> <Reasoning><reasons for these thoughts></Reasoning> </Action>
""",
                     "Discuss":"""Reason step-by-step about something to say based on current situation, your PhysicalState and priorities, and RecentHistory. Respond using this template:
<Action> <Name>Say</Name> <Arg><item of concern you want to discuss, based on the current Situation, your PhysicalState, your emotional needs as reflected in your Priorities or Memory, or based on your observations resulting from previous Do actions.></Arg> <Reasoning><reasons for bringing this up for discussion></Reasoning> </Action>"""}

        allowed_actions=[]
        for intention in self.intentions:
            print(f'act selection intentions\n{intention}')
            mode = find('<Mode>', intention)
            act = find('<Act>', intention)
            reason = find('<Reason>', intention)
            if act is None: continue
            if mode == 'Do':
                allowed_actions.append(all_actions["Act"].replace('{action}', act).replace('{reasoning}',str(reason)))
            elif mode == 'Say':
                allowed_actions.append(all_actions["Say"].replace('{text}', act).replace('{reasoning}',str(reason)))
        if input.endswith('?') or '?' in str(self.sense_input):
            allowed_actions.append(all_actions['Answer'])
        if self.previous_action != 'Think' or random.randint(1,3) == 1: # you think too much
            allowed_actions.append(all_actions['Think'])
        #if len(allowed_actions) < 3 or random.randint(1,2) == 1: # you talk too much
        allowed_actions.append(all_actions['Discuss'])
        if len(allowed_actions) == 0:
            allowed_actions.append(all_actions['Think'])
            
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

Your current short-term goals/intentions:
{{$intentions}}

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

Choose only one action. Respond with your chosen action.
Do not include any introductory, explanatory, or discursive text.
Do not respond with more than one action 
Respond using the XML format shown for the chosen action

Consider the conversation history in choosing your action. 
Do not use the same action repeatedly, perhaps it is time to try another act.
Respond in the context of the RecentHistory (if any) and in keeping with your character. 
Speak only for yourself. Respond only with Action Name, Arg, and Reasoning. 
Do not include any introductory, explanatory, or discursive text, 
Include only your immediate response. Do not include any follow-on conversation.

End your response with:
END
"""
)                 ]
        response = self.llm.ask({'input':input+self.sense_input, 'history':'\n\n'.join(self.history),
                                "memory":self.memory, "situation": self.context.current_state,
                                "physState":self.physical_state, "priorities":'\n'.join(self.priorities),
                                "actions":'- '+'\n- '.join(allowed_actions), "intentions":'\n'.join(self.intentions)
                                }, prompt, temp=0.7, stops=['END', '</Action>'], max_tokens=300)
        #print(f'sense\n{response}\n')
        self.sense_input = ' '
        if 'END' in response:
            idx = response.find('END')
            response = response[:idx]
        if '<|im_end|>' in response:
            idx = response.find('<|im_end|>')
            response = response[:idx]
        #self.add_to_history(speaker, '', input)

        act_name = find('<Name>', response)
        if act_name is not None:
            self.act_name = act_name.strip()
        act_arg = find('<Arg>', response)
        self.reasoning = find('<Reasoning>', response)
        if act_name == 'Think':
            self.reasoning = act_arg+'\n  '+self.reasoning
        self.acts(self.context.actors[1] if self==self.context.actors[0] else self.context.actors[0],
                  act_name, act_arg, self.reasoning)

