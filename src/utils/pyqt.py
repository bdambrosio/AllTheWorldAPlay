import random
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec
import concurrent.futures
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal
import json
import os, json, math, time, requests, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.xml_utils as xml

def generate_faiss_id(allocated_p):
    faiss_id = random.randint(1, 333333333)
    while allocated_p(faiss_id):
        faiss_id = random.randint(1, 333333333)
    return faiss_id

def confirmation_popup(action, argument, modal=True):
    if type(action) is str:
        action = action.strip()
        actions = action
        if action.startswith('{'):
            try:
                action = json.loads(action)
                actions = json.dumps(action, indent=4)
            except:
                pass
    dialog = TextEditDialog(actions, argument, modal=modal)
    result = dialog.exec_()
    if result == QDialog.Accepted:
        return dialog.text_edit.toPlainText()
    else:
        return False

class TextEditDialog(QDialog):
    @staticmethod
    def ensure_app():
        """Ensure QApplication exists"""
        if not QApplication.instance():
            app = QApplication(sys.argv)
            return app
        return None

    def __init__(self, static_text, editable_text, parent=None, modal=True):
        self.app = self.ensure_app()
        super(TextEditDialog, self).__init__(parent)
        
        layout = QVBoxLayout(self)
        
        self.static_label = QLabel(static_text, self)
        layout.addWidget(self.static_label)
        
        self.text_edit = QTextEdit(self)
        self.text_edit.setText(editable_text)
        layout.addWidget(self.text_edit)
        
        self.yes_button = QPushButton('Yes', self)
        self.yes_button.clicked.connect(self.accept)
        layout.addWidget(self.yes_button)
        
        self.no_button = QPushButton('No', self)
        self.no_button.clicked.connect(self.reject)
        layout.addWidget(self.no_button)
        if not modal:
            self.setModal(False)
        
class ListDialog(QDialog):
    @staticmethod
    def ensure_app():
        """Ensure QApplication exists"""
        if not QApplication.instance():
            app = QApplication(sys.argv)
            return app
        return None
    
    def __init__(self, items, parent=None):
        self.app = self.ensure_app()
        super(ListDialog, self).__init__(parent)
        
        self.setWindowTitle('Choose an Item')
        
        self.list_widget = QListWidget(self)
        for item in items:
            self.list_widget.addItem(item)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.button_box)
        
    def selected_index(self):
        return self.list_widget.currentRow()

class PlanDisplayDialog(QDialog):
    """Dialog for displaying and managing saved plans"""
    
    @staticmethod
    def ensure_app():
        """Ensure QApplication exists"""
        if not QApplication.instance():
            app = QApplication(sys.argv)
            return app
        return None
    
    def __init__(self, wm=None, parent=None):
        self.app = self.ensure_app()
        super().__init__(parent)
        self.wm = wm
        self.setWindowTitle("Saved Plans")
        self.setModal(True)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create list widget
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.display_plan)
        layout.addWidget(self.list_widget)
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        view_btn = QPushButton("View")
        view_btn.clicked.connect(lambda: self.display_plan(self.list_widget.currentItem()))
        button_layout.addWidget(view_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self.delete_plan)
        button_layout.addWidget(delete_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Load plans
        self.load_plans()
        
    def load_plans(self):
        """Load plans from working memory into list"""
        self.list_widget.clear()
        items=[f"{self.wm.get(item)['name']}: {str(self.wm.get(item)['item'])[:48]}" 
                for item in self.wm.keys() if self.wm.get(item)['name'].startswith('Plan')]
        for item in items:
            plan_name = item.split(':')[0]
            plan_xml = self.wm.get(plan_name)['item']
            item = QListWidgetItem(plan_name)
            item.setData(Qt.UserRole, plan_xml)
            self.list_widget.addItem(item)
            
    def display_plan(self, item):
        """Display selected plan"""
        if not item:
            return
        plan_xml = item.data(Qt.UserRole)
        if not plan_xml:
            return
            
        # Create display dialog
        text = 'Invalid XML!'
        plan_name = 'tbd'
        try:
            text=xml.format_xml(plan_xml)
            plan_name = xml.find('<name>',plan_xml)
        except:
            print(f'Error formatting XML: {plan_xml}')
        dialog = TextEditDialog(f"Plan: {plan_name}", text)
        dialog.exec_()
        
    def delete_plan(self):
        """Delete selected plan"""
        item = self.list_widget.currentItem()
        if not item:
            return
            
        plan_name = item.text()
        if confirmation_popup("Confirm Delete", f"Delete plan '{plan_name}'?"):
            self.wm.delete(plan_name)
            self.wm.save()
            self.list_widget.takeItem(self.list_widget.row(item))

    def exec_(self):
        """Override exec_ to handle QApplication"""
        result = super().exec_()
        if self.app:
            self.app.quit()
        return result