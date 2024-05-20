import sys
import traceback
from queue import Queue
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QMetaObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QScrollArea, QFrame, QSizePolicy
from PyQt5.QtCore import QTimer, pyqtSlot
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
            self.text_widget.insertPlainText(self.entity.memory)
            self.text_widget.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if event.type() == QEvent.Leave:
            self.text_widget.setVisible(False)
        super().leaveEvent(event)

active_qthreads = 0 # whenever this is zero we will step if RUN is True
RUN = False

class BackgroundSense(QThread):
    taskCompleted = pyqtSignal()
    def __init__(self, entity):
        super().__init__()
        self.entity = entity

    def run(self):
        try:
            #print(f'calling {self.entity} senses')
            result = self.entity.senses(input = '')
        except Exception as e:
            traceback.print_exc()
        self.taskCompleted.emit()

agh_threads = []

class CustomWidget(QWidget):
    def __init__(self, entity, parent=None):
        super(CustomWidget, self).__init__(parent)
        self.ui = parent
        self.entity = entity
        self.entity.widget = self
        layout = QVBoxLayout()
        self.label = QLabel(self.entity.name, self)
        layout.addWidget(self.label)
        if self.entity.name !='World':
            h_layout = QHBoxLayout()
            self.image_label = HoverWidget(self.entity)
            self.image_label.set_image('../images/'+self.entity.name+'.png')
            #self.image_label = QLabel()
            #self.image_label.setFixedSize(192, 192)
            #self.image_label.setAlignment(Qt.AlignCenter)
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
        self.value = QTextEdit()
        self.value.setReadOnly(True)
        self.value.setLineWrapMode(QTextEdit.WidgetWidth)  # Enable text wrapping
        self.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        self.text_area_scroll = QScrollArea()
        self.text_area_scroll.setWidget(self.value)
        self.text_area_scroll.setWidgetResizable(True)
        font = QFont()
        font.setPointSize(13)
        self.value.setFont(font)

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
                description = self.entity.name + ', '+'. '.join(self.entity.character.split('.')[:3])[8:] +', '+self.entity.physical_state+\
                    '. Location: '+context
                prompt = "photorealistic style. "+description+rem_context
                llm_api.generate_dalle_image(prompt, size='192x192', filepath="../images/"+self.entity.name+'.png')
                self.set_image("../images/"+self.entity.name+'.png')
            elif IMAGEGENERATOR == 'tti_serve':
                description = self.entity.name + ', '+'. '.join(self.entity.character.split('.')[:2])[8:] +', '+self.entity.physical_state+\
                    '. Location: '+context
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
        global agh_threads
        self.background_task = BackgroundSense(self.entity)
        agh_threads.append(self.background_task)
        self.background_task.taskCompleted.connect(self.handle_sense_completed)
        self.background_task.start()
        print(f'{self.entity.name} started sense')
        
    def handle_sense_completed(self):
        global agh_threads
        try:
            if self.background_task in agh_threads:
                agh_threads.remove(self.background_task)
        except Exception as e:
            traceback.print_exc()
        if self.entity.name != 'World':
            self.ui.display(self.entity.show)
            self.value.moveCursor(QTextCursor.End)
            self.value.insertPlainText('\n-------------\n')
            self.priorities.clear()
            self.priorities.insertPlainText('\n'.join(self.entity.priorities))
            if self.entity.intention is not None and self.entity.intention != 'None':
                self.intentions.insertPlainText('\n----------\n'+self.entity.intention)
            self.value.insertPlainText(self.entity.reasoning)
            self.value.moveCursor(QTextCursor.End)
            self.value.insertPlainText('\n-------------\n')
            #self.value.insertPlainText(str(self.entity.memory))
            self.update_actor_image()
                
        else:
            self.value.clear()
            self.value.insertPlainText(str(self.entity.current_state))
            for entity in self.entity.ui.actors:
                if entity != 'World':
                    entity.widget.priorities.clear()
                    entity.widget.priorities.insertPlainText('\n'.join(entity.priorities))
                    entity.widget.intentions.clear()
                    entity.widget.intentions.insertPlainText(entity.physical_state)
                    entity.widget.value.insertPlainText(entity.reasoning)
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
        #self.image_label.setPixmap(scaled_pixmap)
        self.image_label.set_image(scaled_pixmap)
        

    def display(self, text):
        self.value.moveCursor(QTextCursor.End)
        self.value.insertPlainText('\n'+text)

    def update_value(self, new_value):
        self.cycle += 1
        if self.entity.name != 'World': # actor
            self.start_sense()
        else: # update world less often, too expensive
            if self.cycle %4 == 3:
                self.start_sense()

class MainWindow(QMainWindow):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.actors = context.actors
        self.init_ui()
        self.internal_time = 0
        self.agh = agh

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
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setLineWrapMode(QTextEdit.WidgetWidth)  # Enable text wrapping
        self.text_area.setFont(font)
        self.setStyleSheet("background-color: #333333; color: #FFFFFF;")
        text_area_scroll = QScrollArea()
        text_area_scroll.setWidget(self.text_area)
        text_area_scroll.setWidgetResizable(True)
        center_panel.addWidget(text_area_scroll)

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
                widget.value.clear()
                widget.value.insertPlainText(str(widget.entity.current_state))
            else:
                widget.priorities.clear()
                widget.priorities.insertPlainText('\n'.join(widget.entity.priorities)+'\n')
                widget.intentions.clear()
                widget.intentions.insertPlainText(widget.entity.physical_state+'\n')
        
        self.show()

    def display(self, r):
        self.text_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
        r = str(r)
        self.text_area.insertPlainText(r)  # Insert the text at the cursor position
        #if self.tts:
        #    try:
        #        self.speech_service(decoded)
        #    except:
        #        traceback.print_exc()
        self.text_area.repaint()
      
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
        #self.append_text(f"\nStep {self.internal_time}\n")

    def step_completed(self):
        self.run_button.setEnabled(True)
        self.step_button.setEnabled(True)
        if RUN:
            self.step()
            
    def pause(self):
        global RUN
        RUN = False
        
    def reset(self):
        global RUN
        self.internal_time = 0
        self.update_parameters()
        self.append_text("Reset")
        RUN = False

    def update_parameters(self):
        # Example: update custom widgets with internal_time
        for widget in self.custom_widgets:
            print(f'{widget.entity.name} sensing')
            widget.update_value(self.internal_time)

    def append_text(self, text):
        self.text_area.append(text)

def main(context):
    app = QApplication(sys.argv)
    main_window = MainWindow(context)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
