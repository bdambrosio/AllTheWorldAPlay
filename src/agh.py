import os
import time
import random
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
import llm_api

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
        
    def history(self):
        history = '\n\n'.join(self.actors[0].history)
        hs = history.split('\n')
        hs_renamed = [self.actors[0].name + s[3:] if s.startswith('You') else s for s in hs]
        history = '\n'.join(hs_renamed)
        return history

    def image(self, filepath):
        state = '. '.join(self.current_state.split('.')[:2])
        characters = '\n'+'.\n'.join([entity.name + ' is '+entity.character.split('.')[0][8:] for entity in self.actors])
        prompt=state+characters
        #print(f'calling generate_dalle_image\n{prompt}')
        llm_api.generate_image(f"""wide-view photorealistic. {prompt}""", size='640x480', filepath=filepath)
        return filepath
    
    def do(self, actor, action):
        prompt=[SystemMessage(content="""You are a dynamic world. Your task is to determine the result of {{$name}} performing {{$action}} in the current situation. 

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
        response = llm.ask({"name":actor.name, "action":action,
                                "state":self.current_state}, prompt, temp=0.7, stops=['END'], max_tokens=200)
        self.current_state += '\n'+response
        return response

    def senses (self, ui_task_queue):
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
That is, advance the environment by approximately 3 hours. The state as of 3 hours ago was:

<PreviousState>
{{$state}}
</PreviousState> 

In the interim, the characters in the world had the following interactions:

<History>
{{$history}}
</History>

Include day/night cycles and weather patterns. Update location and other physical environment characteristics as indicated in the History.
Your response should be concise, and only include only an update of the physical environment.
        """ + event + """
Your state description should be dispassionate, and should begin with a brief description of the current physical space suitable for a text-to-image generator. 

Respond with an updated environment description reflecting a time passage of 3 hours and the events that have occurred.
Include ONLY the concise updated state description in your response. Do not include any introductory, explanatory, or discursive text, or any markdown or other formatting. 
Do NOT carry the storyline beyond three hours from PreviousState.
Limit your response to about 200 words
End your response with:
END""")]
            
        response = llm.ask({"state":self.current_state, 'history':history}, prompt, temp=0.3, stops=['END'], max_tokens=360)
        self.current_state=response
        for actor in self.actors:
            actor.forward(3) # forward three hours and update history
        return response

class Character():
    def __init__ (self, name, character_description):
        self.name = name
        self.speaker = None
        self.character = character_description
        self.history = []
        self.memory = ''
        self.context=None
        self.priorities = ['']
        self.reflect_elapsed = 99
        self.reflect_interval = 3
        self.physical_state = ""
        self.intention = False
        self.previous_action = ''
        self.sense_input = ''
        self.needs = """
 - immediate physiological needs including survival, water, food, clothing, shelter, sleep  
 - safety from threats including ill-health or physical threats from adversarial actors or adverse events. 
 - assurance of short-term future physiological needs (e.g. adequate water and food supplies, shelter maintenance). 
 - love and belonging, including mutual physical contact, comfort with knowing one's place in the world, friedship, intimacy, trust, acceptance.
"""
        self.widget = None
        
    def add_to_history(self, role, act, message):
        message = message.replace('\\','')
        self.history.append(f"{role}: {act} {message}")

    def reflect(self):
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
<ConversationHistory>
{{$history}}
</ConversationHistory)>

Your current priorities are:

<CurrentPriorities>
{{$goals}}
</CurrentPriorities>

Your task is to update your list of priorities given the Situation, your PhysicalState, your Memory, and your recent ConversationHistory as listed above.

List your three most important priorites as instantiations from: 

{{$needs}}
            
List ONLY your most important priorities as simple declarative statements, without any introductory, explanatory, or discursive text.
Your priorities should be as specific as possible. 
For example, if you need sleep, say 'Sleep', not 'Physiological need'.
Similarly, if your safety is threatened by a wolf, respond "Safety from wolf", not merely "Safety"
limit your response to 120 words. 

Use the XML format:
<Priorities>
priority one
priority ...
priority...
</Priorities>

End your response with:
END
""")]
        response = llm.ask({'goals':'\n'.join(self.priorities), 'memory':self.memory,
                                'history':'\n\n'.join(self.history),
                                "situation":self.context.current_state,
                                "physState":self.physical_state, "needs":self.needs
                                },
                               prompt, temp=0.7, stops=['END'], max_tokens=180)
        #self.widget.display(f'-----Memory update-----\n{response}\n\n')
        priorities = find('<Priorities>', response)
        self.priorities = priorities.split('\n')
                                    
    def forward(self, num_hours):
        # roll conversation history forward.
        ## update physical state
        self.reflect()
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

<RecentConversationHistory>
{{$history}}
</RecentConversationHistory)>

Respond with an updated physical state, using this XML format:

<PhysicalState>
<Fear>Low, Medium, High</Fear>
<Thirst>Low, Medium, High</Thirst>
<Hunger>Low, Medium, High</Hunger>
<Fatigue>Low, Medium, High</Fatigue>
<Health>Low, Medium, High</Health>
</PhysicalState>

The updated physical state should focus on:

- Level of Fear - increases with perception of threat, decreases with removal of threat.
- Level of Thirst - increases as time passes since last drink, increases with heat and exertion.
- Level of Hunger - increases as time passes since last food, increases with exertion.
- Level of Fatigue - increases as time passes since last rest, increases with heat and exertion.
- Level of Health  - decreases with injury or illness, increases with rest and healing.

Respond ONLY with the updated state.
Do not include any introductory or peripheral text.
limit your response to 120 words at most.
End your response with:
END""")
                  ]
        response = llm.ask({'me':self.name, 'memory':self.memory,
                                'history':'\n\n'.join(self.history),
                                "situation":self.context.current_state,
                                "physState":self.physical_state
                                },
                               prompt, temp=0.1, stops=['END'], max_tokens=180)
        fear = find('<Fear>', response)
        thirst = find('<Thirst>', response)
        hunger = find('<Hunger>', response)
        fatigue = find('<Fatigue>', response)
        health = find('<Health>', response)
        self.physical_state = '\n'.join([name+': '+value for name, value in zip(['fear', 'thirst', 'hunger', 'fatigue', 'health'], [fear, thirst, hunger, fatigue, health])])
        print(f'Physical State update:\n{self.name}: {self.physical_state}')

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

<RecentConversationHistory>
{{$history}}
</RecentConversationHistory)>

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
        response = llm.ask({'me':self.name, 'memory':self.memory,
                                'history':'\n\n'.join(self.history[:-2]),
                                "situation":self.context.current_state},
                               prompt, temp=0.4, stops=['END'], max_tokens=300)
        response = response.replace('<Memory>','')
        self.memory = response
        #conv = '\n\n'.join(self.history[:-2])
        #print(f'history to memory formation\n{conv}\n')
        self.history = self.history[-2:]
        #conv = '\n\n'.join(self.history)
        #print(f'history after memory formation \n{conv}\n')

    def acts(self, target, act_name, act_arg='', reasoning=''):
        #
        ### speak to someone
        #
        show = ''
        if act_name is not None and act_arg is not None and len(act_name) >0 and len(act_arg) > 0:
            #self.ui.display('\n**********************\n')
            if act_name=='Say' or act_name == 'Do':
                self.add_to_history('You', act_name , act_arg)
                self.add_to_history('You', 'think',f'   reasoning: {reasoning}')
                for actor in self.context.actors:
                    if actor != self: # everyone else sees/hears your act!
                        #print(f'{self.name} sharing {act_name} with {actor.name}')
                        actor.add_to_history(self.name, act_name , act_arg)
                if target is not None:
                    target.sense_input = '\n'+self.name+' '+act_name+': '+act_arg
                self.previous_action = act_name
                show = act_name + ": "+act_arg
                self.ui.display(f'\n{self.name}: {show}')
                if act_name =='Do':
                    self.intention = False

                    result = self.context.do(self, act_arg)
                    show += '\nthen'+result
                    self.add_to_history('You', 'observe', result)
                    if target is not None: # this is wrong, world should update who sees do
                        target.sense_input += '\n'+result
                    self.ui.display(f'  {result}')
            else:
                show = 'Seems to be thinking ...'
                self.previous_action='Think'
                self.add_to_history('You', 'think', act_arg)
                self.add_to_history('You', 'think',f'   reasoning: {reasoning}')
                self.ui.display(f'\n{self.name}: {show}\n')
                for actor in self.context.actors:
                    if actor != self: # everyone else sees/hears your act!
                        #print(f'{self.name} showing thinking with {actor.name}')
                        actor.add_to_history('You', 'see', f'{self.name} thinking')

            if act_name == 'Say' or act_name == 'Think':
                prompt=[SystemMessage(content="""Your task is to analyze the following text.

<Text>
{{$text}}
</Text>

Does it include an intention for 'I' to act? 
Respond using the following XML form:

<Analysis>
<Act>False if there is no intention to act, True if there is an intention to act</Act>
<Intention>stated intention - action or goal</Intention>
</Analysis>

===Examples===

Text:
'Good morning Annie. I'm heading to the office for the day. Call maintenance about the disposal noise please.'

Response:
<Analysis>
<Act>True</Act>
<Intention>Head to the office for the day.</Intention>
</Analysis>

Text:
'Good morning Annie. Call maintenance about the disposal noise please.'

Response:
<Analysis>
<Act>False</Act>
<Intention>None</Intention>
</Analysis>

===End Examples===

Do NOT include any introductory, explanatory, or discursive text.
Respond only with the intention analysis in XML as shown above.
End your response with:
END
""")]
                response = llm.ask({"text":act_arg}, prompt, temp=0.5, stops=['END'], max_tokens=100)
                act = find('<Act>', response)
                self.intention = find('<Intention>', response)
                print(f'{self.name} intends {self.intention}')
        else:
            print(f'\n\nstuff missing\n  act {act_name} act_arg {act_arg}\n\n')
        return show

    def senses(self, input=''):
        #print(f'\n********************ask*********************\nSpeaker: {speaker}, Input:{input}')
        all_actions={"Act":f"Act in the world using the Do action, with '{self.intention}' as the action to perform (value for <Arg>).",
                     "Answer":"If Input is a question, respond with your answer using the Say action.", 
                     "Say":f"If a previous Think expressed intention to say something and is not followed by a Say, then use the Say action with '{self.intention}' as words to say (value for <Arg>).",
                     "Think":"Think about Input and ConversationHistory with respect to Priorities, using the Think action. Limit response to 140 words.",
                     "Discuss":"Start a new topic to discuss using the Say action, based on the current Situation, your PhysicalState, your emotional needs as reflected in your Priorities or Memory, or based on your observations resulting from previous Do actions."}

        allowed_actions=[]
        if self.intention is not None and self.intention != False and 'False' not in self.intention:
            allowed_actions.append(all_actions["Act"])
        if input.endswith('?'):
            allowed_actions.append(all_actions['Answer'])
        if self.previous_action == 'Think':
            allowed_actions.append(all_actions['Say'])
        if self.previous_action != 'Think' and random.randint(1,2) == 1: # you think too much
            allowed_actions.append(all_actions['Think'])
        if len(allowed_actions) < 3 and  random.randint(1,2) == 1: # you talk too much
            allowed_actions.append(all_actions['Discuss'])
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
<ConversationHistory>
{{$history}}
</ConversationHistory)>

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
Respond using the following XML format:

<Action>
  <Name>name of action to perform: 'Say' or 'Do' or 'Think'</Name>
  <Arg> words to say / action to perform / thought </Arg>
  <Reasoning>reason for action chosen, including Priority it addresses</Reasoning>
</Action>

Consider the conversation history in choosing your action. 
Do not use the same action repeatedly, perhaps it is time to try another act.
Respond in the context of the ConversationHistory (if any) and in keeping with your character. 
Speak only for yourself. Respond only with Action Name, Arg, and Reasoning. 
Do not include any introductory, explanatory, or discursive text, 
Include only your immediate response. Do not include any follow-on conversation.

End your response with:
END
"""
)                 ]
        response = llm.ask({'input':input+self.sense_input, 'history':'\n\n'.join(self.history),
                                "memory":self.memory, "situation": self.context.current_state,
                                "physState":self.physical_state, "priorities":'\n'.join(self.priorities),
                                "actions":'- '+'\n- '.join(allowed_actions), "intention":self.intention
                                }, prompt, temp=0.9, stops=['END'], max_tokens=500)
        #print(f'action choice response \n{response}')
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
            act_name = act_name.strip()
        act_arg = find('<Arg>', response)
        reasoning = find('<Reasoning>', response)
        #self.ui.display(f'\n{self.name}: {act_name}, {act_arg}, \n')
        self.reasoning = reasoning
        if act_name == 'Think':
            self.reasoning = act_arg+'\n  '+self.reasoning
        self.acts(self.context.actors[1] if self==self.context.actors[0] else self.context.actors[0],
                  act_name, act_arg, reasoning)

llm = llm_api.LLM()
