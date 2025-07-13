import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import sys
import os
import threading
import time
import random
import traceback
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QLineEdit,QLabel,QPushButton,QScrollArea,QSizePolicy,QHBoxLayout
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QPixmap
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import Qt, QTimer, QEvent, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor, QFont
from utils.llm_api import LLM
import sim.agh as agh
import sim.context as context
import xml.etree.ElementTree as xml
import sim.human as human
import utils.llm_api as llm_api
import json
import utils.xml_utils as xml



IMAGEGENERATOR = 'tti_serve'
UPDATE_LOCK = threading.Lock()
APP = None # pyQt5 app
main_window = None
WATCHER = None # the person watching, in-world representative created on the fly from Inject
home_dir = os.path.expanduser("~")
WORLDS_DIR = Path(home_dir, ".local/share/AllTheWorld/worlds/")

def add_text_ns(widget, text):
    """add text to a TextEdit without losing scroll location"""
    scroll_position = widget.verticalScrollBar().value()  # Save the current scroll position   
    widget.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
    widget.insertPlainText(text)  # Insert the text at the cursor position   
    widget.verticalScrollBar().setValue(scroll_position)  # Restore the scroll position

def map_state(state):
    mapped_state = [f"{key}:  {item['state']}" for key, item in state.items()]
    return ', '.join(mapped_state)

class HoverWidget(QWidget):
    def __init__(self, character):
        super().__init__()
        self.character = character
        
        # Create the main layout
        self.layout = QVBoxLayout()
        
        # Create the image label
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(192, 192)
        self.layout.addWidget(self.image_label)
        
        # Create the text widget (QTextEdit or QLabel)
        self.text_widget = QTextEdit(self)
        self.text_widget.setText("Customizable text that appears on hover")
        self.text_widget.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        self.text_widget.setVisible(False)  # Initially hidden
        self.layout.addWidget(self.text_widget)
        
        # Set the layout
        self.setLayout(self.layout)
        
        # Load an image
        #self.set_image("path/to/your/image.jpg")

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def enterEvent(self, event):
        print(f'event {event}')
        if event.type() == QEvent.Enter:
            print(f' enter under mouse! {event}')
            self.text_widget.clear()
            recent_memories = self.character.structured_memory.get_recent(5)
            memory_text = '\n'.join(memory.to_string() for memory in recent_memories)
            add_text_ns(self.text_widget, memory_text)
            self.text_widget.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if event.type() == QEvent.Leave:
            self.text_widget.setVisible(False)
        super().leaveEvent(event)

active_qthreads = 0 # whenever this is zero we will step if RUN is True
RUN = False
STEP_CTR=0

class InputWidget(QDialog):
    def __init__(self, prompt):
        super().__init__()
        self.user_input = None
        self.setWindowTitle("User Input")
        self.setGeometry(100, 100, 300, 150)

        layout = QVBoxLayout()

        self.label = QLabel(prompt)
        layout.addWidget(self.label)

        self.input_field = QLineEdit()
        layout.addWidget(self.input_field)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit_input)
        layout.addWidget(self.submit_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_input)
        layout.addWidget(self.cancel_button)


        self.setLayout(layout)

    def submit_input(self):
        self.user_input = self.input_field.text()
        self.accept()

    def cancel_input(self):
        self.user_input = None
        self.close()

    def get_user_input(self):
        return self.user_input

class BackgroundSense(QThread):
    #class BackgroundSense():
    taskCompleted = pyqtSignal()
    def __init__(self, character):
        super().__init__()
        self.character = character

    def run(self):
        global UPDATE_LOCK
        with UPDATE_LOCK:
            try:
                print(f'calling {self.character.name} senses')
                # other source of effective input is assignment to self.character.sense_input, e.g. from other say
                result = self.character.senses(sense_data = '')
                self.taskCompleted.emit()
            except Exception as e:
                traceback.print_exc()


agh_threads = []

class WrappingLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWordWrap(True)

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

class CustomWidget(QWidget):
    def __init__(self, context, character, parent=None):
        super().__init__(parent)
        self.context = context
        self.character = character
        self.ui=parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.top_bar = QHBoxLayout()
        name = QLabel(self.context.name, self)
        name.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        name.adjustSize()
        self.top_bar.addWidget(name)
        if self.character != self.context:
            self.active_task = WrappingLabel(self.character.active_task.peek(), self)
            self.active_task.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.active_task.setWordWrap(True)
            self.active_task.adjustSize()
            self.top_bar.addWidget(self.active_task)
            self.state = WrappingLabel(map_state(self.character.state), self)
            self.state.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.state.setWordWrap(True)
            self.state.adjustSize()
            self.top_bar.addWidget(self.state)
            self.top_bar.setSpacing(5)  # Add this line
            self.top_bar.addStretch()  # Add this line
            layout.addLayout(self.top_bar)
            h_layout = QHBoxLayout()
            self.image_label = HoverWidget(self.character)
            self.image_label.set_image('images/'+self.character.name+'.png')
            h_layout.addWidget(self.image_label)
            self.intentions = QTextEdit()
            self.intentions.setLineWrapMode(QTextEdit.WidgetWidth)
            self.intentions.setStyleSheet("background-color: #333333; color: #FFFFFF;")
            h_layout.addWidget(self.intentions)
            self.priorities = QTextEdit()
            self.priorities.setLineWrapMode(QTextEdit.WidgetWidth)
            self.priorities.setStyleSheet("background-color: #333333; color: #FFFFFF;")
            h_layout.addWidget(self.priorities)
            layout.addLayout(h_layout)
        self.thoughts = QTextEdit()
        self.thoughts.setReadOnly(True)
        self.thoughts.setLineWrapMode(QTextEdit.WidgetWidth)  # Enable text wrapping
        self.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        self.text_area_scroll = QScrollArea()
        self.text_area_scroll.setWidget(self.thoughts)
        self.text_area_scroll.setWidgetResizable(True)
        font = QFont()
        font.setPointSize(13)
        self.thoughts.setFont(font)

        layout.addWidget(self.text_area_scroll)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        self.cycle = 0
        #if self.context.name != 'World': # can't do this before entity is fully initialized
        #    self.update_actor_image()
        if self.character != self.context:
            self.character.update_actor_image()
        #self.show()
        print(f'{self.character.name} ui widget inititalized')
        
    def update_actor_image(self):
        try:
            #add first two sentences of initial context for background
            context = self.context.current_state.split('.')
            if len(context[0].strip()) > 0:
                context = context[0].strip()
                rem_context = context[1:]
            elif len(context) > 1:
                context = context[1].strip()
                rem_context = context[2:]
            else:
                context = self.context.current_state[:84]
                context = ''
                i = 0
                candidates = self.context.current_state.split('.')
                while len(context) < 84 and i < len(candidates):
                    context += candidates[i]+'. '
                    i +=1
                context = context[:96]
                description = self.character.name + ', '+'. '.join(self.character.character.split('.')[:2])[6:] +', '+\
                    self.character.show.replace(self.character.name, '')[-128:].strip()
                description = description[:192-min(len(context), 48)] + '. '+context
                prompt = description
                print(f' actor image prompt len {len(prompt)}')
                image_path = llm_api.generate_image(self.context.llm, prompt, size='192x192', filepath=self.character.name + '.png')
                self.set_image(str(image_path))
        except Exception as e:
            traceback.print_exc()

    def update_world_image(self):
        # assumes this is being executed on context widget
        image_path = self.context.image(filepath='worldsim.png')
        self.ui.set_image(str(image_path))

    def start_sense(self):
        result = self.character.senses(sense_data = '')
        #self.background_task = BackgroundSense(self.context)
        agh_threads.append(self) # keep track of how many threads are running, even if serially
        #self.background_task.run()
        #self.handle_sense_completed()
        #self.background_task.taskCompleted.connect(self.handle_sense_completed)
        #self.background_task.start()
        #time.sleep(0.1) # ensure entities run in listed order
        print(f'{self.character.name} started sense')
        self.handle_sense_completed()   

    def format_intentions(self):
        return '<br>'.join(["<b>"+str(xml.find('<mode>', intention))
                          +':</b>('+str(xml.find('<source>', intention))+') '+str(xml.find('<act>', intention))[:32]
                          +'<br> Why: '+str(xml.find('<reason>', intention))[:32]
                          +'<br>'
                          for intention in self.character.intentions])
            
    def format_tasks(self):
        return '<br>'.join(["<b>"+str(xml.find('<name>', task))
                          +':</b> '+str(xml.find('<reason>', task))
                          +'<br>'
                          for task in self.character.priorities])
            
    def handle_sense_completed(self):
        if self.character != self.context:
            self.update_entity_state_display()
            self.character.update_actor_image()
            self.ui.display('\n')
            self.ui.display(self.character.show)
            self.character.show='' # need to not erase till image update!
        else:
            self.thoughts.insertPlainText('\n------time passes-----\n')
            self.thoughts.insertPlainText(str(self.context.current_state))
            if self.context.show is not None and len(self.context.show) > 0:
                self.ui.display(self.context.show)
                self.context.show=''
            for entity in self.context.ui.actors:
                if entity.name != 'World' and type(entity) != context.Context:
                    entity.widget.update_entity_state_display()
            path = self.context.image('worldsim.png')
            self.ui.set_image(path)
            #self.ui.display('\n----- context updated -----\n')
            self.ui.display('\n')

        # initiate another cycle?
        print(f'{self.context.name} sense completed')
        
            
    def set_image(self, image_path):
        pixmap = QPixmap(str(image_path))
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.set_image(scaled_pixmap)
        

    def display(self, text):
        self.thoughts.moveCursor(QTextCursor.End)
        self.thoughts.insertPlainText('\n'+text)
        #add_text_ns(self.value, '\n'+text)

    def update_value(self, new_value):
        self.start_sense()

    def update_entity_state_display(self):
        self.active_task.setText(self.character.active_task.peek())
        self.active_task.adjustSize()
        
        print(f'updating entity state display')
        self.state.setText(map_state(self.character.state))
        self.state.adjustSize()

        self.priorities.clear()
        self.priorities.insertHtml(self.format_tasks())
        self.intentions.clear()
        self.intentions.insertHtml(self.format_intentions())
        # display show, not reason, because show includes thought if act was think
        if self.character.show is not None and len(self.character.show) > 4:
            self.thoughts.moveCursor(QTextCursor.End)
            self.thoughts.insertPlainText('\n-------------\n')
            self.thoughts.insertPlainText(self.character.thought)
            self.thoughts.moveCursor(QTextCursor.End)
        #if type(self.context) == agh.Agh:
        #    print(f'\n----\n{self.context.name} last_acts:\n{self.context.last_acts}\n')
        
class WorldSim(QMainWindow):
    def __init__(self, context, server='deepseek', world_name='Lost'):
        super().__init__()
        self.context = context
        self.server = server
        self.world_name = world_name
        self.character_widgets = {}  # Track character widgets here
        self.llm = LLM(server)
        self.context.set_llm(self.llm)
        self.steps_since_last_update = 0
        
        # Add listener for state updates
        self.context.add_state_listener(self.handle_state_update)
        self.init_ui()
        for actor in self.context.actors:
           actor.set_llm(self.llm)
        for actor in self.context.actors:
           print(f'calling {actor.name} initialize')
           actor.initialize(self)
        for actor in self.context.actors:
            print(f'calling {actor.name} greet')
            #actor.greet()
            actor.see()

        for widget in self.custom_widgets:
           #
           widget.update_entity_state_display()

    def handle_state_update(self, update_type, data):
        """Handle context state updates"""
        if update_type == 'output':
            self.text_display.append(data['text'])
        elif update_type == 'state_update':
            # Update character display
            self.update_character_display(data['characters'])
            # Update other UI elements as needed
            self.text_display.update()
            
    def update_character_display(self, characters):
        """Update character panel with current states"""
        text = ""
        for char in characters:
            text += f"{char.name}:\n"
            for priority in char.priorities:
                text += f"  {priority.name}: {priority.value}\n"
        self.character_display.setText(text)

    def init_ui(self):
        # Main central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        
        # Left panel with custom widgets
        left_panel = QVBoxLayout()
        self.custom_widgets = [
            self.create_character_widget(actor, parent=self)
            for actor in self.context.actors
        ]
       
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel)
        left_panel_widget.setFixedWidth(800)
        self.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        main_layout.addWidget(left_panel_widget)

        # Center panel with image display and text area
        center_panel = QVBoxLayout()
        
        self.image_label = QLabel()
        #self.context.widget.update_world_image()
        self.image_label.setAlignment(Qt.AlignCenter)
        center_panel.addWidget(self.image_label)
        
        font = QFont()
        font.setPointSize(14)  # Set font size to 14 points
        self.activity= QTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setLineWrapMode(QTextEdit.WidgetWidth)  # Enable text wrapping
        self.activity.setFont(font)
        self.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        #self.text_area.setWidgetResizable(True)
        center_panel.addWidget(self.activity)
        #text_area_scroll = QScrollArea()
        #text_area_scroll.setWidget(self.text_area)
        #text_area_scroll.setWidgetResizable(True)
        #center_panel.addWidget(text_area_scroll)

        center_panel_widget = QWidget()
        center_panel_widget.setLayout(center_panel)
        main_layout.addWidget(center_panel_widget)
        
        # Right panel with control buttons
        right_panel = QVBoxLayout()
        
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run)
        right_panel.addWidget(self.run_button)

        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.step)
        right_panel.addWidget(self.step_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause)
        right_panel.addWidget(self.pause_button)

        self.inject_button = QPushButton("Inject")
        self.inject_button.clicked.connect(self.inject)
        right_panel.addWidget(self.inject_button)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        right_panel.addWidget(self.refresh_button)

        self.load_button = QPushButton("Load World")
        self.load_button.clicked.connect(self.load)
        right_panel.addWidget(self.load_button)

        self.save_button = QPushButton("Save World")
        self.save_button.clicked.connect(self.save)
        right_panel.addWidget(self.save_button)

        right_panel.addStretch(1)
        
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel)
        main_layout.addWidget(right_panel_widget)

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('PyQt5 Application')
        agh.ui = self
        #self.show()       
        for widget in self.custom_widgets:
            if widget.character == widget.context:
                widget.thoughts.clear()
                widget.thoughts.insertPlainText(str(widget.character.current_state))
                widget.update_world_image()
            else:
                widget.update_entity_state_display()
                widget.character.update_actor_image()
            left_panel.addWidget(widget)

        self.context.image()
        self.show()

    def display(self, r):
        add_text_ns(self.activity, r)
        #if self.tts:
        #    try:
        #        self.speech_service(decoded)
        #    except:
        #        traceback.print_exc()
      
    def set_image(self, image_path):
        pixmap = QPixmap(str(image_path))
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def run(self):
        global RUN # declare because doing assign
        self.run_button.setEnabled(False)
        self.step_button.setEnabled(False)
        RUN = True
        self.step()

    def step(self):
        self.run_button.setEnabled(False)
        self.step_button.setEnabled(False)
        self.update_parameters()
        # Process events to keep UI responsive
        QApplication.processEvents()
        

    def step_completed(self):
        self.run_button.setEnabled(True)
        self.step_button.setEnabled(True)
        QApplication.processEvents()
        if RUN:
            self.step()
            
    def pause(self):
        global RUN
        RUN = False
        

    def inject(self):
        global APP, WATCHER, main_window
        if WATCHER is None:
            WATCHER = human.Human('Watcher', "Human user representative", main_window)
            WATCHER.context=self.context
        input_widget = InputWidget("Character name, message:")
        if input_widget.exec_() == QDialog.Accepted:
            user_input = input_widget.get_user_input()
            WATCHER.inject(user_input)

    def refresh(self):
        for widget in self.custom_widgets:
            if type(widget.context) != context.Context:
                widget.update_entity_state_display()


    def load(self):
        worlds = [d for d in os.listdir(WORLDS_DIR) if os.path.isdir(os.path.join(WORLDS_DIR, d))]
        # print(models)
        world_number = -1

        while world_number < 0 or world_number > len(worlds) - 1:
            print(f'Available worlds:')
            for i in range(len(worlds)):
                world_name = worlds[i]
                try:
                    with open(WORLDS_DIR / world_name / 'scenario.json', 'r') as s:
                        scenario = json.load(s)
                        if 'name' in scenario.keys():
                            print(f" {i}. {scenario['name']}")
                except FileNotFoundError as e:
                    print(str(e))
            number = input('\ninput world # to load: ')
            try:
                world_number = int(number)
            except:
                print(f'Enter a number between 0 and {len(worlds) - 1}')
        self.world_name = worlds[world_number]
        RUN = False
        self.context.load(WORLDS_DIR / world_name)

    def save(self):
        #get filename
        if self.world_name is None:
            input_widget = InputWidget("World name for save:")
            if input_widget.exec_() == QDialog.Accepted:
                self.world_name = input_widget.get_user_input()
        if self.world_name == None:
            return
        input_widget = InputWidget(f'Saving {self.world_name}, ok?')
        if input_widget.exec_() == QDialog.Accepted:
            worlds_path = WORLDS_DIR / self.world_name
            if not worlds_path.exists():
                worlds_path.mkdir(parents=True, exist_ok=True)
                print(f"Directory '{worlds_path}' created.")
            self.context.save(worlds_path, self.world_name)

    def update_parameters(self):
        # Advance simulation time
        new_time = self.context.advance_time()
        
        for widget in self.custom_widgets:
            if type(widget.character) != context.Context:
                # Pass simulation time to forward
                widget.update_value(new_time)
                # Process events periodically to check for pause
                QApplication.processEvents()
            else:
                if self.steps_since_last_update > 5:    
                    widget.context.senses('')
                    widget.update_world_image()
                    self.steps_since_last_update = 0
                else:
                    self.steps_since_last_update += 1
                    widget.thoughts.clear()
                    widget.thoughts.insertPlainText(str(widget.context.current_state))
        self.step_completed()

    def create_character_widget(self, character, parent=None):
        """Create and track widget for character"""
        widget = CustomWidget(self.context, character, self)
        self.character_widgets[character] = widget
        return widget

def main(context, server='local', world_name=None):
    global APP, main_window, WORLDS_DIR
    worlds_path = Path(WORLDS_DIR)
    if not worlds_path.exists():
        worlds_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{worlds_path}' created.")
    else:
        print(f"Directory '{worlds_path}' already exists.")
    APP = QApplication(sys.argv)

    main_window = WorldSim(context, server, world_name)
    main_window.world_name = world_name
    main_window.llm = LLM(server)
    main_window.context = context
    #set refs to llm
    main_window.context.set_llm(main_window.llm)
    main_window.actors = context.actors
    main_window.server = server
    for actor in main_window.actors:
       actor.set_llm(main_window.llm)
    for actor in main_window.actors:
       print(f'calling {actor.name} initialize')
       actor.initialize(main_window)
    for actor in main_window.actors:
        print(f'calling {actor.name} greet')
        #actor.greet()
        actor.see()

    #for actor in main_window.actors:
    #   actor.widget.update_entity_state_display()

    # Update server setting for characters
    for actor in context.actors:
        if isinstance(actor, agh.Agh):
            actor.llm = LLM(server)
            
    sys.exit(APP.exec_())

if __name__ == '__main__':
    S = agh.Agh("Samantha", """You are a pretty young Sicilian woman. 
You love the outdoors and hiking.
You are intelligent, introspective, philosophical and a bit of a romantic, but keep this mostly to yourself. 
You have a painful history, maybe it is just as well you don't remember it.
You are very informal, chatty, and think and speak in teen slang, and are a playful and flirty when relaxed. 
You are comfortable on long treks, and are unafraid of hard work. 
You are suspicious by nature, and wary of strangers. 
Your name is Samanatha""")

    #Specifying for this scenario, otherwise all they do is hunt for water, berries, and grubs
    S.set_drives([
    #"evaluation of Joe. Can I trust him?",
    #"safety from threats including accident, illness, or physical threats from unknown or adversarial actors or adverse events.",
    #"finding a way out of the forest.",
    "solving the mystery of how they ended up in the forest with no memory.",
    "love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "immediate physiological needs: survival, shelter, water, food, rest."
    ])
    S.add_to_history("You think This is very very strange. Where am i? I'm near panic. Who is this guy? How did I get here? Why can't I remember anything?")

    #
    ## Now Joe, the other character in this 'survivor' scenario
    #

    J = agh.Agh("Joe", """You are a young Sicilian male, intelligent and self-sufficient. 
You are informal and somewhat impulsive. 
You are strong, and think you love the outdoors, but are basically a nerd.
You yearn for something more, but don't know what it is.
You are socially awkward, especially around strangers. 
You speak in informal teen style.
Your name is Joe.""")

    J.set_drives([
    "communication and coordination with Samantha, gaining Samantha's trust.",
    #"safety from threats including accident, illness, or physical threats from unknown or adversarial actors or adverse events.",
    #"finding a way out of the forest.",
    "solving the mystery of how they ended up in the forest with no memory.",
    #"love and belonging, including home, acceptance, friendship, trust, intimacy.",
    "immediate physiological needs: survival, shelter, water, food, rest."
    ])

    J.add_to_history("You think Ugh. Where am I?. How did I get here? Why can't I remember anything? Who is this woman?")
    # add a romantic thread. Doesn't work very well yet. One of my test drivers of agh, actually.
    J.add_to_history("You think Whoever she is, she is pretty!")


    #    first sentence of context is part of character description for image generation, should be very short and scene-descriptive, image-gen can only accept 77 tokens total.
    W = context.Context([S, J],
                """A temperate, mixed forest-open landscape with no buildings, roads, or other signs of humananity. It is a early morning on what seems like it will be a warm, sunny day.
""")

    main(W, server='deepseek', world_name='Lost')
