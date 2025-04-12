from ast import literal_eval
from PyQt5.QtWidgets import (QApplication, QTextEdit, QVBoxLayout, QWidget, 
                            QPushButton, QDialog, QProgressDialog, QMessageBox,
                            QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from pathlib import Path
import json, requests
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def import_file():
    file_path, _ = QFileDialog.getOpenFileName(
        window,
        "Select Text File",
        "",
        "Text Files (*.txt *.log);;All Files (*)"
    )
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                text_edit.setPlainText(content)
        except Exception as e:
            QMessageBox.critical(window, "Error", f"Failed to read file: {str(e)}")

def format_text():
    raw_text = text_edit.toPlainText()
    formatted_text = raw_text.replace("\\t", "\t").replace("\\n", "\n")
    text_edit.setPlainText(formatted_text)

def clear_text():
    text_edit.clear()

def submit_to_llm():
    prompt = text_edit.toPlainText()
    if not prompt.strip():
        QMessageBox.warning(window, "Error", "Please enter text before submitting")
        return
    
    # Convert only unescaped single quotes to double quotes
    prompt = prompt.replace('"', '\\"')
    prompt = re.sub(r"(?<!\\)'", '"', prompt)
    
    try:
        prompt = prompt.replace("\n", "\\n").replace("\t", "\\t")
        prompt = prompt.replace("\\\\\\n", "\\n")
        prompt = prompt.replace("\\\\\\t", "\\t")
        prompt = literal_eval(prompt)
    except Exception as e:
        QMessageBox.warning(window, "Error", f"Failed to parse prompt: {str(e)}")
        return
    progress = QProgressDialog("Waiting for LLM response...", None, 0, 0, window)
    progress.setWindowModality(Qt.WindowModal)
    progress.show()
    
    try:
        url = 'http://localhost:5000/v1/chat/completions'
        response =  requests.post(url, headers={"Content-Type":"application/json"}, json=prompt)
        text = response.content.decode('utf-8').strip()
        dialog = ResponseDialog(text, window)
        progress.close()
        dialog.exec_()
    except Exception as e:
        progress.close()
        QMessageBox.critical(window, "Error", f"Failed to get LLM response: {str(e)}")

class ResponseDialog(QDialog):
    def __init__(self, response_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LLM Response")
        layout = QVBoxLayout()
        
        response_edit = QTextEdit()
        response_edit.setPlainText(response_text)
        response_edit.setReadOnly(True)
        response_edit.setStyleSheet("background-color: #2E2E2E; color: ivory;")
        layout.addWidget(response_edit)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
        self.resize(600, 400)

# Create the PyQt application
app = QApplication([])

# Create the main window and layout
window = QWidget()
layout = QVBoxLayout()

# Create the QTextEdit widget for text input and output
text_edit = QTextEdit()
text_edit.setStyleSheet("background-color: #2E2E2E; color: ivory;")  # Dark blue-grey background and ivory text
text_edit.setPlaceholderText("Paste your text here...")
text_edit.setTabStopDistance(40)  # Set tab stop distance (in pixels)
textFont = QFont(); textFont.setPointSize(13)
text_edit.setFont(textFont)  
layout.addWidget(text_edit)

# Create a QPushButton to import files
import_button = QPushButton("Import File")
import_button.clicked.connect(import_file)
layout.addWidget(import_button)

# Create a QPushButton to trigger text formatting
format_button = QPushButton("Format Text")
layout.addWidget(format_button)

# Connect the button's clicked signal to the function
format_button.clicked.connect(format_text)

clear_button = QPushButton("Clear Text")
layout.addWidget(clear_button)

# Connect the button's clicked signal to the function
clear_button.clicked.connect(clear_text)

submit_button = QPushButton("Submit to LLM")
submit_button.clicked.connect(submit_to_llm)
layout.addWidget(submit_button)

# Set up the window
window.setLayout(layout)
window.setWindowTitle("Text Formatter")
window.show()

# Run the app
app.exec_()
