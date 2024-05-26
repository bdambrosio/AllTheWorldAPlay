import os
import time
import random
import traceback
import readline
from utils.Messages import SystemMessage, UserMessage, AssistantMessage
import llm_api
import agh

class Human (agh.Character):
    def __init__ (self, name, character_description):
        super().__init__(name, character_description)

    def initialize(self):
        """called from worldsim once everything is set up"""

    def add_to_history(self, role, act, message):
        message = message.replace('\\','')
        self.history.append(f"{role}: {act} {message}")
        self.history = self.history[-3:] # memory is fleeting, otherwise we get very repetitive behavior

    def update_physical_state(self, key, response):
        pass

    def forward(self, num_hours):
        pass
    
    def tell(self, actor, message, source='dialog'):
        print(f"{actor.name} says: {message}")
        self.senses()
        
    def update_intentions_wrt_say_think(self, source, text, reason):
        # determine if text implies an intention to act, and create a formatted intention if so
        print(f'Update intentions from say or think\n {text}\n{reason}')
        pass

    def senses(self, sense_data='', ui_queue=None):
        print(f'\n*********senses***********\nCharacter: {self.name}')
        target = None
        try:
            parts = []
            while len(parts) != 2:
                text = input('Enter act_name; act_description:\n')
                parts = text.split(';')
                
            act_name = parts[0].strip().capitalize()
            act_dscp = parts[1].strip()
            self.reason = ''
            task_name='dialog'
            if act_name=='Say' or act_name=='Do':
                self.last_acts[task_name]= act_dscp
                self.active_task = task_name
                #this will effect selected act and determine consequences
            self.acts(target, act_name, act_dscp, self.reason, task_name)

        except Exception as e:
            traceback.print_exc()
