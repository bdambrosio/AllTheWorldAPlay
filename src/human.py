import os, re
import time
import random
import traceback
import readline
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
import llm_api
import agh

class Human (agh.Character):
    def __init__ (self, name, ui, character_description=''):
        super().__init__(name, character_description)
        self.ui = ui
        self.context = ui.context
        
    def initialize(self):
        """called from worldsim once everything is set up"""

    def add_to_history(self, role, act, message):
        message = message.replace('\\','')
        self.history.append(f"{role}: {act} {message}".strip())
        self.history = self.history[-8:] # memory is fleeting, otherwise we get very repetitive behavior

    def update_physical_state(self, key, response):
        pass

    def forward(self, num_hours):
        pass
    
    def tell(self, actor, message, source='dialog'):
        print(f"{actor.name} says: {message}")

    def inject(self, message):
        split_chars = ',;:\n'
        pattern = f"[{re.escape(split_chars)}]"
        print(f'Human inject called w msg {message}')
        result = re.split(pattern, message)
        parts = [x for x in result if x]
        if parts is None or len(parts) < 2:
            return
        if len(parts[0].strip()) >0:
            who = parts[0].strip()
        else:
            who = parts[1].strip()
        print(f'Inject target {who}')
        for actor in self.context.actors:
            if actor.name == who.strip():
                actor.tell(self, message[len(who):], source='watcher')
                actor.add_to_history('You hear watcher say '+message)
    
    def update_intentions_wrt_say_think(self, source, text, reason):
        # determine if text implies an intention to act, and create a formatted intention if so
        print(f'Update intentions from say or think\n {text}\n{reason}')
        pass

    def senses(self, sense_data='', ui_queue=None):
        print(f'\n*********senses***********\nCharacter: {self.name}')
        try:
            if self.intentions is None or len(self.intentions) ==0:
                return
            intention = self.intentions[0]
            act_name = agh.find('<Mode>', intention)
            act_dscp = agh.find('<Act>', intention)
            act_reason = agh.find('<Reason>', intention)
            task_name = agh.find('<Source>', intention)
            if act_name=='Say' or act_name=='Do':
                self.last_acts[task_name]= act_dscp
                if task_name != 'dialog' and task_name != 'watcher':
                    self.active_task.push(task_name)
                self.reason = act_reason
                #this will effect selected act and determine consequences
            self.acts(None, act_name, act_dscp, act_reason, task_name)

        except Exception as e:
            traceback.print_exc()
