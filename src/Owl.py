from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QTimer, QTextCodec, QRect
from PyQt5.QtWidgets import QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton
import signal
#from PyQt5 import QApplication
import random
import re
import traceback
import time
import requests # for web search service
import subprocess
import os

from utils.Messages import UserMessage
#from pyexts import Openbook as op
#from utils import OSClient
#import ipinfo
#import nyt
from chat.OwlCoT import OwlInnerVoice
#from Planner import Planner
#from Interpreter import Interpreter
import chat.react as react


NYT_API_KEY = os.getenv("NYT_API_KEY")

city = 'Berkeley'
state = 'California'
print(f"My city and state is: {city}, {state}")
print("Owl loaded")
local_time = time.localtime()
year = local_time.tm_year
day_name = ['Monday', 'Tuesday', 'Wednesday', 'thursday','friday','saturday','sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
month_day = local_time.tm_mday
hour = local_time.tm_hour

global news, news_details


FORMAT=True
PREV_LEN=0


max_tokens = 7144

class ImageDisplay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_files = ["../images/Owl2.png", "../images/Owl.png"]
        self.current_image_index=0
        # Create layout manager
        layout = QtWidgets.QVBoxLayout()
        self.setWindowTitle('Owl')
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        self.label = QtWidgets.QLabel("")
        layout.addWidget(self.label)
        # Load image file
        self.update_image()
        self.resize(240,240)
        self.show()

    def update_image(self):
        """Updates the displayed image."""
        img_path = self.image_files[self.current_image_index]
        pixmap = QtGui.QPixmap(img_path).scaled(360, 360, Qt.KeepAspectRatio)
        rect = QRect(60, 60, 240, 240)
        pixmap = pixmap.copy(rect)
        self.label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        """Handle mouse press events to change the image on left click."""
        if event.button() == Qt.LeftButton:
            # Increment the index. If at the end of the list, go back to the start.
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            # Update the image displayed
            self.update_image()

class ChatApp(QtWidgets.QWidget):
   def __init__(self):
      super().__init__()
      self.tts = False
      self.owlCoT = OwlInnerVoice(self)
      self.memory_display = None
      self.windowCloseEvent = self.closeEvent
      signal.signal(signal.SIGINT, self.controlC)
      # Set the background color for the entire window
      self.setAutoFillBackground(True)
      palette = self.palette()
      palette.setColor(self.backgroundRole(), QtGui.QColor("#202020"))  # Use any hex color code
      self.setPalette(palette)
      self.codec = QTextCodec.codecForName("UTF-8")
      self.widgetFont = QFont(); self.widgetFont.setPointSize(14)
      self.reflect = True

      #self.setStyleSheet("background-color: #101820; color")
      # Main Layout
      main_layout = QHBoxLayout()
      # Text Area
      text_layout = QVBoxLayout()
      main_layout.addLayout(text_layout)
      
      class MyTextEdit(QTextEdit):
         def __init__(self, app):
            super().__init__()
            self.app = app
            self.textChanged.connect(self.on_text_changed)

         def on_text_changed(self):
            self.app.timer.stop()
            self.app.timer.start(600000)
            
         def keyPressEvent(self, event):
            if event.matches(QKeySequence.Paste):
               clipboard = QApplication.clipboard()
               self.insertPlainText(clipboard.text())
            else:
               super().keyPressEvent(event)
            
      self.input_area = MyTextEdit(self)
      self.input_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
      #self.input_area.setAcceptRichText(True)
      
      self.mainFont = QFont("Noto Color Emoji", 14)
      self.input_area.setFont(self.widgetFont)
      self.input_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
      text_layout.addWidget(self.input_area)
      
      self.msg_area = MyTextEdit(self)
      self.msg_area.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
      self.msg_area.setFont(self.widgetFont)
      self.msg_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
      text_layout.addWidget(self.msg_area)
      
      # Control Panel
      control_layout = QVBoxLayout()
      control_layout2 = QVBoxLayout()

      # Buttons and Comboboxes
      self.submit_button = QPushButton("Submit")
      self.submit_button.setFont(self.widgetFont)
      self.submit_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.submit_button.setToolTip('submit new text to LLM')
      self.submit_button.clicked.connect(self.submit)
      control_layout.addWidget(self.submit_button)
      
      self.clear_button = QPushButton("Clear")
      self.clear_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.clear_button.setFont(self.widgetFont)
      self.clear_button.clicked.connect(self.clear)
      self.clear_button.setToolTip('clear display')
      control_layout.addWidget(self.clear_button)
      
      self.clear_mem_button = QPushButton("Clear\nStream")
      self.clear_mem_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.clear_mem_button.setFont(self.widgetFont)
      self.clear_mem_button.clicked.connect(self.clear_mem)
      self.clear_mem_button.setToolTip('clear memory stream. Memory stream is the record of a reactive thought loop, and a "recent+RAG" subset is included in llm context. This action includes both in-memory and the persistant record.')
      control_layout.addWidget(self.clear_mem_button)
      
      self.temp_combo = self.make_combo(control_layout, '   Temp', [".01", ".1", ".2", ".4", ".5", ".7", ".9", "1.0"])
      self.temp_combo.setCurrentText('.1')
      self.temp_combo.setToolTip('Temperature parameter for most LLM calls.')
      
      self.top_p_combo = self.make_combo(control_layout, '  Top_P', [".01", ".1", ".2", ".4", ".5", ".7", ".9", "1.0"])
      self.top_p_combo.setToolTip('Top_p parameter for most LLM calls.')
      self.top_p_combo.setCurrentText('1.0')
      
      self.max_tokens_combo = self.make_combo(control_layout, 'MaxTkns', ["25", "50", "100", "150", "250", "400", "600", "1000", "2000", "4000"])
      self.max_tokens_combo.setToolTip('max_tokens for most LLM calls. For multiple-call actions like research, controls overall response, and proportionally, internal calls')
      self.max_tokens_combo.setCurrentText('600')
      
      self.tts_button = QPushButton("Speak") # launch working memory editor
      self.tts_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.tts_button.setFont(self.widgetFont)
      self.tts_button.setToolTip('toggles text to speech if you have the TTS server running.')
      self.tts_button.clicked.connect(self.speak)

      #label = QLabel(" Planner")
      #label.setStyleSheet("QLabel {background-color: #202020; color: #AAAAAA; }")
      #label.setFont(self.widgetFont)
      #label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      #control_layout.addWidget(label)
      
      #self.plan_button = QPushButton("Plan")
      #self.plan_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      #self.plan_button.setFont(self.widgetFont)
      #self.plan_button.clicked.connect(self.plan)
      #control_layout.addWidget(self.plan_button)
      
      #self.run_plan_button = QPushButton("Run Plan")
      #self.run_plan_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      #self.run_plan_button.setFont(self.widgetFont)
      #self.run_plan_button.clicked.connect(self.run_plan)
      #control_layout.addWidget(self.run_plan_button)
      
      #self.step_plan_button = QPushButton("Step Plan")
      #self.step_plan_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      #self.step_plan_button.setFont(self.widgetFont)
      #self.step_plan_button.clicked.connect(self.step_plan)
      #control_layout.addWidget(self.step_plan_button)

      #spacer = QSpacerItem(0, 20)  # vertical spacer with 20 pixels height
      #control_layout.addItem(spacer)  # Add spacer to the layout

      self.recall_button = QPushButton("Recall")
      self.recall_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.recall_button.setFont(self.widgetFont)
      self.recall_button.clicked.connect(self.recall)
      self.recall_button.setToolTip('uses new text in the input window as a google search query.')
      control_layout.addWidget(self.recall_button)

      ext_label = QLabel("External")
      ext_label.setStyleSheet("QLabel {background-color: #202020; color: #AAAAAA; }")
      ext_label.setFont(self.widgetFont)
      ext_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      control_layout.addWidget(ext_label)
      
      self.google_button = QPushButton("Google")
      self.google_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.google_button.setFont(self.widgetFont)
      self.google_button.clicked.connect(self.google_search)
      self.google_button.setToolTip('uses new text in the input window as a google search query.')
      control_layout.addWidget(self.google_button)


      label = QLabel("Library")
      label.setStyleSheet("QLabel {background-color: #202020; color: #AAAAAA; }")
      label.setFont(self.widgetFont)
      label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      control_layout.addWidget(label)
      
      self.index_button = QPushButton("Index")
      self.index_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.index_button.setFont(self.widgetFont)
      self.index_button.clicked.connect(self.index_url)
      self.index_button.setToolTip('Assumes the new text in the input window is a uri for a pdf. Also assumes the indexing server is running. Queues the uri for indexing in the research library. ')
      control_layout.addWidget(self.index_button)

      self.s2_button = QPushButton("Online\nSearch")
      self.s2_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.s2_button.setFont(self.widgetFont)
      self.s2_button.clicked.connect(self.s2_search)
      self.s2_button.setToolTip("Uses the new text in the input window as a query to SemanticScholar. Launches a semanticScholar search, after mangling the query in various ways, and steps through with the user which found papers to index. Papers will be queued for indexing even if the indexing server isn't running at the moment.")
      control_layout.addWidget(self.s2_button)
      
      self.library_button = QPushButton("Library\nSearch")
      self.library_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.library_button.setFont(self.widgetFont)
      self.library_button.clicked.connect(self.library_search)
      self.library_button.setToolTip("Uses the new text in the input window as a query to the library. Displays the found papers.")
      control_layout.addWidget(self.library_button)
      
      self.browse_button = QPushButton("Library\nBrowse")
      self.browse_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.browse_button.setFont(self.widgetFont)
      self.browse_button.clicked.connect(self.library_browse)
      self.browse_button.setToolTip("launch library browse app. Supports title/abstract search, document display, chat with selected doc.")
      control_layout.addWidget(self.browse_button)
      
      self.research_button = QPushButton("Research")
      self.research_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.research_button.setFont(self.widgetFont)
      self.research_button.clicked.connect(self.research)
      self.research_button.setToolTip("Uses the new text in the input window to launch a captive research task. This can take a minute or two, found relevant research library entries will be massaged into a coherent short report. Options will be provided for followon sections. Citations for library items used (both internal faiss ppr id and formal citation if available) will be listed in lower window. (make sure to scroll down if you don't see them!")
      control_layout.addWidget(self.research_button)
      
      self.report_button = QPushButton("Report")
      self.report_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.report_button.setFont(self.widgetFont)
      self.report_button.clicked.connect(self.generate_report)
      self.report_button.setToolTip("Uses the new text in the input window to launch a captive report-writer task. This can take a bit. Report writer will first work with you to develop an expanded description of the task, an outline, the depth of search for background material (online or local library only), then draft the report. At the moment (will change) the report is text, and is written to 'section<n>.txt' locally. lilbrary items used (both internal faiss ppr id and formal citation if available) will be listed in lower window. (make sure to scroll down if you don't see them!")
      control_layout.addWidget(self.report_button)
      
      
      control_layout.addStretch(1)  # Add stretch to fill the remaining space
      self.owl = ImageDisplay()
      
      # Add control layout to main layout
      main_layout.addLayout(control_layout)
      main_layout.addLayout(control_layout2)
      self.setLayout(main_layout)
      self.timer = QTimer()
      self.timer.setSingleShot(True)  # Make it a single-shot timer
      self.timer.timeout.connect(self.on_timer_timeout)
      self.owlCoT.init_Owl_Doc()
      #greeting = self.owlCoT.wakeup_routine()
      #self.display_response(greeting)

   def make_combo(self, control_layout, label, choices, callback=None):
      spacer = QSpacerItem(0, 10)  # Vertical spacer with 20 pixels width
      control_layout.addItem(spacer)  # Add spacer to the layout
      
      label = QLabel(label)
      label.setStyleSheet("QLabel {background-color: #202020; color: #AAAAAA; }")
      label.setFont(self.widgetFont)
      label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      control_layout.addWidget(label)
      
      combo = QComboBox()
      combo.setFont(self.widgetFont)
      combo.addItems(choices)
      combo.setStyleSheet("""
QComboBox { background-color: #101820; color: #FAEBD7; }
QComboBox QAbstractItemView { background-color: #101820; color: #FAEBD7; }  # Set the background color of the list view (dropdown)
      """)        
      #combo.clicked.connect(callback)
      control_layout.addWidget(combo)

      return combo


   def get_profile(self, profile, theme):
      global profile_contexts
      if profile in profile_contexts.keys():
         profile_dict = profile_contexts[profile]
         if theme in profile_dict.keys(): 
            choice = random.choice(profile_dict[theme])
            #print(choice)
            return choice
         else:
            print(f'{theme} not found in {profile}: {list(profile_dict.keys())}')
      else:
         print(f'{profile} not found {profile_contexts.keys()}')

   CURRENT_PROFILE_PROMPT_TEXT = ''

   def get_current_profile_prompt_text(self):
      global CURRENT_PROFILE_PROMPT_TEXT
      return CURRENT_PROFILE_PROMPT_TEXT


   def closeEvent(self, event):
      if self.owl is not None:
          self.owl.close()
      event.accept()  # Allow the window to close

   def controlC(self, signum, frame):
      if self.owl is not None:
          self.owl.close()
      QApplication.exit()

   def display_response(self, r):
      global PREV_LEN
      self.input_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
      r = str(r)
      encoded = self.codec.fromUnicode(r)
      # Decode bytes back to string
      decoded = encoded.data().decode('utf-8')
      decoded = decoded.replace('\\n','\n')
      if not decoded.endswith('\n'):
         decoded += '\n'
      self.input_area.insertPlainText(decoded)  # Insert the text at the cursor position
      if self.tts:
         try:
            self.speech_service(decoded)
         except:
            traceback.print_exc()
      self.input_area.repaint()
      PREV_LEN=len(self.input_area.toPlainText())-1
      
   def display_msg(self, r):
      self.msg_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
      r = str(r)
      # presumably helps handle extended chars
      encoded = self.codec.fromUnicode(r)
      decoded = encoded.data().decode('utf-8')+'\n'
      self.msg_area.insertPlainText(decoded)  # Insert the text at the cursor position
      if self.tts:
         try:
            self.speech_service(decoded)
         except:
            traceback.print_exc()
      self.msg_area.repaint()
      
   def submit(self):
      global PREV_LEN
      self.timer.stop()
      print('timer reset')
      self.timer.start(600000)
      new_text = self.input_area.toPlainText()[PREV_LEN:].strip()
      response = ''
      print(f'submit {new_text}')
      response = self.owlCoT.doc.tell(self.owlCoT.owl, new_text, source='dialog')
      self.owlCoT.owl.show = ''
      self.owlCoT.owl.senses()
      #response = self.owlCoT.invoke_react_loop(new_text, self) # this last for async display
      print(f'submit response\n{self.owlCoT.owl.show}')
      self.display_response(self.owlCoT.owl.show)
      return

   def clear(self):
      global PREV_POS, PREV_LEN
      self.input_area.clear()
      PREV_POS="1.0"
      PREV_LEN=0
   
   def clear_mem(self):
      global PREV_POS, PREV_LEN
      self.input_area.clear()
      PREV_POS="1.0"
      PREV_LEN=0
      print(f'clear_mem calling {type(self.owlCoT.owl.memory_stream)}')
      self.owlCoT.owl.memory_stream.clear()
   
   def recall(self):
      """ recall from memory stream """
      global PREV_LEN, op
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText()) + 2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]

      selectedText = selectedText.strip()
      memories = self.owlCoT.memory_stream_recall(selectedText)
      self.display_response(memories)

   ## External tools
   #
   def google_search(self): # extended search of s2 and ingest reports
       global PREV_LEN, op
       selectedText = ''
       cursor = self.input_area.textCursor()
       if cursor.hasSelection():
           selectedText = cursor.selectedText()
       elif PREV_LEN < len(self.input_area.toPlainText())+2:
           selectedText = self.input_area.toPlainText()[PREV_LEN:]
           selectedText = selectedText.strip()
       response = self.owlCoT.web(query=selectedText, widget=self)
       self.display_response('\n\n'+response)
    #
    ## Planner interface
    #

   def plan(self): # select or create a plan
      self.planner.select_plan()
         
   def run_plan(self): # ask planner to run a plan
      self.planner.run_plan()
         
   def step_plan(self): # release a working memory item from active memory
      self.planner.step_plan()

   #
   ## Semantic memory interface
   #

   def research(self): 
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
      response,paper_ids = self.owlCoT.library_search(selectedText)
      citations = []
      for paper_id in paper_ids:
          citations.append(str(paper_id)+': '+s2.cite(paper_id))
      self.display_response('\n'+response)
      self.display_msg('\nReferences:\n'+'\n\n'.join(citations))

   def is_valid_uri(self, uri):
      # Improved regex pattern for matching most URIs
      pattern = re.compile(
         r'^https?://'  # http:// or https://
         r'(?:(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,6}'  # Domain name
         r'|'  # or
         r'localhost'  # localhost
         r'|'  # or
         r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
         r'(?::\d+)?'  # Optional port
         r'(?:/?|[/?]\S+)$', re.IGNORECASE)  # Optional path and query
      return re.match(pattern, uri) is not None
   

   def index_url(self): # index a url in S2 faiss
      global PREV_LEN, op#, vmem, vmem_clock
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
         print(f'cursor has selected, len {len(selectedText)}')
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
         print(f'cursor has selected {len(selectedText)} chars')
      start = selectedText.find('http')
      if start >= 0 and self.is_valid_uri(selectedText[start:]):
         selectedText = selectedText[start:]
      elif start < 0:
         start = selectedText.find('file:///')
         if start < 0:
            self.display_msg(f'not uri: {selectedText}')
            return
      selectedText = selectedText[start:]
      self.s2.queue_url_for_indexing(selectedText)
      self.display_response("\n")
      self.display_msg("Indexing request submitted.")

   def s2_search(self): # extended search of s2 and ingest reports
      global PREV_LEN, op
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
      s2.online_search(selectedText)
         
   def library_search(self): # extended search of s2 and ingest reports
       global PREV_LEN, op
       selectedText = ''
       cursor = self.input_area.textCursor()
       if cursor.hasSelection():
           selectedText = cursor.selectedText()
       elif PREV_LEN < len(self.input_area.toPlainText())+2:
           selectedText = self.input_area.toPlainText()[PREV_LEN:]
           selectedText = selectedText.strip()
       papers = s2.search(selectedText, interactive=True)
       titles = set(papers.keys())
       paper_ids = []
       for title in titles:
           if title is not None:
               paper = s2.paper_from_title(title)
               if paper is not None:
                   paper_ids.append(paper['faiss_id'])
       chooser = s2.PaperSelect(list(titles), paper_ids, '')
       chooser.show()
       app.exec()

   def library_browse(self): # extended search of s2 and ingest reports
        global PREV_LEN, op
        selectedText = ''
        cursor = self.input_area.textCursor()
        if cursor.hasSelection():
            selectedText = cursor.selectedText()
        elif PREV_LEN < len(self.input_area.toPlainText())+2:
            selectedText = self.input_area.toPlainText()[PREV_LEN:]
            selectedText = selectedText.strip()
        s2.browse(selectedText)
        #rr = subprocess.Popen(['python3', 'semanticScholar3.py', '-browse', selectedText, '-template', f"{self.owlCoT.llm.template}"])
         
   def generate_report(self): # index a url in S2 faiss
      global PREV_LEN, op#, vmem, vmem_clock
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
      rr = subprocess.Popen(['python3', 'paper_writer.py', '-report', selectedText, '-template', f"{self.owlCoT.llm.template}"])
      self.display_msg("report writer spawned.")
         
   def workingMem(self): # lauching working memory editor
      self.owlCoT.save_workingMemory() # save current working memory so we can edit it
      he = subprocess.run(['python3', 'memoryEditor.py'])
      if he.returncode == 0:
         try:
            self.workingMemory = self.owlCoT.load_workingMemory()
         except Exception as e:
            self.display_msg(f'Failure to reload working memory\n  {str(e)}')

   def speak(self): # lauching working memory editor
      prompt = [UserMessage(content="""Generate a question to add details to the task the user wants to accomplish.\nRespond using the following JSON template:

{"question":\'<question to ask>\'}

Respond only with the above JSON, without any commentary or explanatory text
"""),
                UserMessage(content="""What specific types of planning, problem-solving, or goal-directed behaviors are you interested in using LLMs for?"""),
                UserMessage(content="""I am interested in using LLMs for researching scientific articles, processing the found pdfs, and using the found information to construct solutions for disease. These solutions will often involve custom miRNA and will require underestanding genomics and proteomics. Again, to focus, the goal at the moment is a system that can create such solutions, not the biological solution itself."""),
                UserMessage(content="""Generate a followup question about any additional requirements of the task.
Respond using the following JSON template:
{"question":\'<question to ask>\'}

Respond only with the above JSON, without any commentary or explanatory text""")
                ]
      self.owlCoT.llm.ask({}, prompt, client= self.owlCoT.llm.anthropicClient)
      

      if self.tts:
         self.tts = False
         self.display_msg('Speech off')
      else:
         self.display_msg('Speech on')
         self.tts = True

   def speech_service(self, text):
      #self.display_msg('speaking...')
      try:
         r = requests.post("http://bruce-linux:5004/", json={"text":text})
      except Exception as e:
         print('\nspeech attempt failed {str(e)}\n')

   def history(self):
      self.owlCoT.historyEditor() # save and display Conversation history for editting

   def on_timer_timeout(self):
      global profile, profile_text
      if not self.reflect:
         return
      response = self.owlCoT.reflect()
      #print(f'Reflection response {response}')
      if response is not None and type(response) == dict:
         if 'tell' in response.keys():
            self.display_response(response['tell'])
            react.remember('Owl reflects '+ response['tell'])
      self.timer.start(600000) # longer timeout when nothing happening
      #print('timer start')

    
app = QtWidgets.QApplication([])
window = ChatApp()
window.show()

if __name__== '__main__':
   import library.semanticScholar3 as s2
   window.s2=s2
   s2.ui=window
   s2.cot=window.owlCoT
   app.exec_()


