import re
import traceback
import agh
import utils.xml_utils as xml

class Human (agh.Character):
    def __init__ (self, name, character_description=''):
        super().__init__(name, character_description)
        self.name = name
        self.character = character_description

    def initialize(self):
        """called from worldsim once everything is set up"""

    def add_to_history(self, message):
        message = message.replace('\\','')
        self.history.add_to_history(message)

    def update_physical_state(self, key, response):
        pass

    def forward(self, num_hours):
        pass
    
    def tell(self, actor, message, source='dialog', respond=True):
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
        self.inject(input("Watcher says: "))
        print(f'\n*********senses***********\nCharacter: {self.name}')
        try:
            if self.intentions is None or len(self.intentions) ==0:
                return
            intention = self.intentions[0]
            act_name = xml.find('<Mode>', intention)
            act_dscp = xml.find('<Act>', intention)
            act_reason = xml.find('<Reason>', intention)
            task_name = xml.find('<Source>', intention)
            if act_name=='Say' or act_name=='Do':
                self.last_acts[task_name]= act_dscp
                if task_name != 'dialog' and task_name != 'watcher':
                    self.active_task.push(task_name)
                self.reason = act_reason
                #this will effect selected act and determine consequences
            self.acts(None, act_name, act_dscp, act_reason, task_name)

        except Exception as e:
            traceback.print_exc()
