import random
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec
import concurrent.futures
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton, QDialog, QListWidget, QDialogButtonBox
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal

def generate_faiss_id(allocated_p):
    faiss_id = random.randint(1, 333333333)
    while allocated_p(faiss_id):
        faiss_id = random.randint(1, 333333333)
    return faiss_id

def confirmation_popup(action, argument, modal=True):
    dialog = TextEditDialog(action, argument, modal=modal)
    result = dialog.exec_()
    if result == QDialog.Accepted:
        return dialog.text_edit.toPlainText()
    else:
        return False

class TextEditDialog(QDialog):
    def __init__(self, static_text, editable_text, parent=None, modal=True):
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
    def __init__(self, items, parent=None):
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
