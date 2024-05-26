import sys
import traceback
import threading
from queue import Queue
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QMetaObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QScrollArea, QFrame, QSizePolicy
from PyQt5.QtCore import QTimer, pyqtSlot, QSize
import PyQt5.QtGui as QtGui
from PyQt5.QtGui import QPixmap, QImage, QFont, QTextCursor
import numpy as np
import agh
import llm_api

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QEvent

IMAGEGENERATOR = 'tti_serve'
UPDATE_LOCK = threading.Lock()

def add_text_ns(widget, text):
    """add text to a TextEdit without losing scroll location"""
    scroll_position = widget.verticalScrollBar().value()  # Save the current scroll position   
    widget.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
    widget.insertPlainText(text)  # Insert the text at the cursor position   
    widget.verticalScrollBar().setValue(scroll_position)  # Restore the scroll position


class HoverWidget(QWidget):
    def __init__(self, entity):
        super().__init__()
        self.entity = entity
        
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
        self.set_image("path/to/your/image.jpg")

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def enterEvent(self, event):
        print(f'event {event}')
        if event.type() == QEvent.Enter:
            print(f' enter under mouse! {event}')
            self.text_widget.clear()
            add_text_ns(self.text_widget, self.entity.memory)
            #self.text_widget.insertPlainText(self.entity.memory)
            self.text_widget.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if event.type() == QEvent.Leave:
            self.text_widget.setVisible(False)
        super().leaveEvent(event)

active_qthreads = 0 # whenever this is zero we will step if RUN is True
RUN = False
STEP_CTR=0

class BackgroundSense(QThread):
    taskCompleted = pyqtSignal()
    def __init__(self, entity):
        super().__init__()
        self.entity = entity

    def run(self):
        global UPDATE_LOCK
        with UPDATE_LOCK:
            try:
                #print(f'calling {self.entity} senses')
                #other source of effective input is assignment to self.entity.sense_input, e.g. from other say
                result = self.entity.senses(sense_data = '')
            except Exception as e:
                traceback.print_exc()
        self.taskCompleted.emit()

agh_threads = []

class WrappingLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWordWrap(True)

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

class CustomWidget(QWidget):
    def __init__(self, entity, parent=None):
        super(CustomWidget, self).__init__(parent)
        self.ui = parent
        self.entity = entity
        self.entity.widget = self
        layout = QVBoxLayout()
        self.top_bar = QHBoxLayout()
        name = QLabel(self.entity.name, self)
        name.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        name.adjustSize()
        self.top_bar.addWidget(name)
        if type(self.entity) != agh.Context:
            self.active_task = WrappingLabel(self.entity.active_task, self)
            self.active_task.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.active_task.setWordWrap(True)
            self.active_task.adjustSize()
            self.top_bar.addWidget(self.active_task)
            self.physical_state = WrappingLabel(self.entity.physical_state, self)
            self.physical_state.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.physical_state.setWordWrap(True)
            self.physical_state.adjustSize()
            self.top_bar.addWidget(self.physical_state)
            self.top_bar.setSpacing(5)  # Add this line
            self.top_bar.addStretch()  # Add this line
            layout.addLayout(self.top_bar)
        if self.entity.name !='World':
            h_layout = QHBoxLayout()
            self.image_label = HoverWidget(self.entity)
            self.image_label.set_image('../images/'+self.entity.name+'.png')
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
        if self.entity.name != 'World':
            self.update_actor_image()
            
        print(f'{self.entity.name} ui widget inititalized')
        
    def update_actor_image(self):
        try:
            #add first two sentences of initial context for background
            context = self.entity.context.initial_state.split('.')
            if len(context[0].strip()) > 0:
                context = context[0].strip()
                rem_context = context[1:]
            elif len(context) > 1:
                context = context[1].strip()
                rem_context = context[2:]
            else:
                context = self.entity.context.current_state[:84]
            if IMAGEGENERATOR == 'dall-e-2':
                # can take a longer dscp than tti_serve
                description = self.entity.name + ', '+'. '.join(self.entity.character.split('.')[:2])[8:] +', '+\
                    self.entity.show + '. Location: '+context
                prompt = "photorealistic style. "+description+rem_context
                llm_api.generate_dalle_image(prompt, size='192x192', filepath="../images/"+self.entity.name+'.png')
                self.set_image("../images/"+self.entity.name+'.png')
            elif IMAGEGENERATOR == 'tti_serve':
                description = self.entity.name + ', '+'. '.join(self.entity.character.split('.')[:2])[8:] +', '+\
                    self.entity.show.replace(self.entity.name, '')[-72:] + '. Location: '+context
                prompt = "photorealistic style. "+description
                print(f' actor image prompt len {len(prompt)}')
                llm_api.generate_image(prompt, size='192x192', filepath="../images/"+self.entity.name+'.png')
        except Exception as e:
            traceback.print_exc()
        self.set_image("../images/"+self.entity.name+'.png')

    def update_world_image(self):
        image_path =  self.ui.context.image(filepath='../images/worldsim.png', image_generator=IMAGEGENERATOR)
        self.ui.set_image(image_path)

    def start_sense(self):
        self.background_task = BackgroundSense(self.entity)
        agh_threads.append(self.background_task)
        self.background_task.taskCompleted.connect(self.handle_sense_completed)
        self.background_task.start()
        print(f'{self.entity.name} started sense')

    def format_intentions(self):
        return '<br>'.join(["<b>"+str(agh.find('<Mode>', intention))
                          +':</b>('+str(agh.find('<Source>', intention))+') '+str(agh.find('<Act>', intention))[:32]
                          +'<br> Why: '+str(agh.find('<Reason>', intention))[:32]
                          +'<br>'
                          for intention in self.entity.intentions])
            
    def format_tasks(self):
        return '<br>'.join(["<b>"+str(agh.find('<Text>', task))
                          +':</b> '+str(agh.find('<Reason>', task))
                          +'<br>'
                          for task in self.entity.priorities])
            
    def handle_sense_completed(self):
        global agh_threads, UPDATE_LOCK
        try:
            if self.background_task in agh_threads:
                agh_threads.remove(self.background_task)
        except Exception as e:
            traceback.print_exc()
        if self.entity.name != 'World':
            self.ui.display('\n')
            self.ui.display(self.entity.show)
            self.update_entity_state_display()
            self.update_actor_image()
        else:
            self.thoughts.clear()
            self.thoughts.insertPlainText(str(self.entity.current_state))
            for entity in self.entity.ui.actors:
                if entity.name != 'World' and type(entity) != agh.Context:
                    entity.widget.update_entity_state_display()
            path = self.entity.image('../images/worldsim.png')
            self.ui.set_image('../images/worldsim.png')
            self.ui.display('\n----- context updated -----\n')

        # initiate another cycle?
        print(f'{self.entity.name} sense completed {len(agh_threads)}')
        if len(agh_threads) == 0:
            self.ui.step_completed()
        
            
    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.set_image(scaled_pixmap)
        

    def display(self, text):
        self.thoughts.moveCursor(QTextCursor.End)
        self.thoughts.insertPlainText('\n'+text)
        #add_text_ns(self.value, '\n'+text)

    def update_value(self, new_value):
        self.start_sense()

    def update_entity_state_display(self):
        self.active_task.setText(self.entity.active_task)
        self.active_task.adjustSize()
        self.physical_state.setText(self.entity.physical_state)
        self.physical_state.adjustSize()
        self.priorities.clear()
        self.priorities.insertHtml(self.format_tasks())
        self.intentions.clear()
        self.intentions.insertHtml(self.format_intentions())
        # display show, not reason, because show includes thought if act was think
        if self.entity.show is not None and len(self.entity.show) > 4:
            self.thoughts.moveCursor(QTextCursor.End)
            self.thoughts.insertPlainText('\n-------------\n')
            self.thoughts.insertPlainText(self.entity.thought)
            self.thoughts.moveCursor(QTextCursor.End)
        #if type(self.entity) == agh.Agh:
        #    print(f'\n----\n{self.entity.name} last_acts:\n{self.entity.last_acts}\n')
        
class MainWindow(QMainWindow):
    def __init__(self, context, server):
        super().__init__()
        self.llm = llm_api.LLM(server)
        self.context = context
        #set refs to llm
        self.context.llm = self.llm
        self.actors = context.actors
        self.init_ui()
        self.internal_time = 0
        self.agh = agh
        self.server=server
        for actor in self.actors:
           actor.llm = self.llm
        for actor in self.actors:
           print(f'calling {actor.name} initialize')
           actor.initialize()
        for actor in self.actors:
           print(f'calling {actor.name} greet')
           actor.greet()

        for actor in self.actors:
           actor.widget.update_entity_state_display()

        #print(f'initial tells')
        #self.actors[0].tell(self.actors[1], 'Where are we, Who are you?')
        #self.actors[1].tell(self.actors[0], "What's going on?")
        #self.actors[0].widget.update_entity_state_display()
        #self.actors[1].widget.update_entity_state_display()

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
        self.custom_widgets = [CustomWidget(self.context, parent=self)]
        self.custom_widgets.extend([ CustomWidget(actor, parent=self) for actor in self.actors])
        for widget in self.custom_widgets:
            left_panel.addWidget(widget)
            widget.entity.ui = self # let entity know about UI for display
        
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel)
        left_panel_widget.setFixedWidth(800)
        self.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        main_layout.addWidget(left_panel_widget)

        # Center panel with image display and text area
        center_panel = QVBoxLayout()
        
        self.image_label = QLabel()
        self.context.widget.update_world_image()
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

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        right_panel.addWidget(self.refresh_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        right_panel.addWidget(self.reset_button)
        
        right_panel.addStretch(1)
        
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel)
        main_layout.addWidget(right_panel_widget)

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('PyQt5 Application')
        agh.ui = self
        for widget in self.custom_widgets:
            if widget.entity.name == 'World':
                widget.thoughts.clear()
                widget.thoughts.insertPlainText(str(widget.entity.current_state))
            else:
                widget.update_entity_state_display()
        
        self.show()

    def display(self, r):
        add_text_ns(self.activity, r)
        #self.text_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
        #self.text_area.insertPlainText(r)  # Insert the text at the cursor position
        self.activity.repaint()
        #if self.tts:
        #    try:
        #        self.speech_service(decoded)
        #    except:
        #        traceback.print_exc()
      
    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def run(self):
        global RUN # declare because doing assign
        self.run_button.setEnabled(False)
        self.step_button.setEnabled(False)
        RUN = True
        self.step()

    def step(self):
        print('Step')
        self.run_button.setEnabled(False)
        self.step_button.setEnabled(False)
        self.update_parameters()
        self.append_text(f"\nStep {self.internal_time}\n")

    def step_completed(self):
        self.internal_time += 1
        self.run_button.setEnabled(True)
        self.step_button.setEnabled(True)
        if RUN:
            self.step()
            
    def pause(self):
        global RUN
        RUN = False
        
    def refresh(self):
        for widget in self.custom_widgets:
            if type(widget.entity) != agh.Context:
                widget.update_entity_state_display()

    def reset(self):
        global RUN
        self.update_parameters()
        self.append_text("Reset")
        RUN = False

    def update_parameters(self):
        # Example: update custom widgets with internal_time
        for widget in self.custom_widgets:
            #print(f'{widget.entity.name} sensing')
            if type(widget.entity) != agh.Context:
                widget.update_value(self.internal_time)
            else:
                if self.internal_time % 5 == 4:
                    widget.update_value(self.internal_time)

    def append_text(self, text):
        self.activity.append(text)

def main(context, server='local'):
    app = QApplication(sys.argv)
    main_window = MainWindow(context, server=server)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
