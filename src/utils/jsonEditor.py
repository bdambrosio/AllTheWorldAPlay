import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QMessageBox, QTextEdit, QDialog
from PyQt5.QtCore import pyqtSignal

class JsonEditor(QWidget):
    closed = pyqtSignal(object)
    def __init__(self, json_dict, parent=None):
        super().__init__(parent)
        if not isinstance(json_dict, dict):
            raise ValueError(f"Expected a dict, got {type(json_dict)}")
        self.json_dict = json_dict
        self.setWindowTitle("JSON Editor")
        self.setGeometry(300, 300, 800, 600)
        self.user_closed = True # true means user closed window rather than click ok or cancel
        
        # Text Edit for JSON
        self.textEdit = QTextEdit()
        self.textEdit.setText(json.dumps(self.json_dict, indent=2))

        # Ok button
        self.okButton = QPushButton('Ok')
        self.okButton.clicked.connect(self.okExit)

        # Cancel button
        self.cancelButton = QPushButton('Cancel')
        self.cancelButton.clicked.connect(self.cancelExit)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.textEdit)
        layout.addWidget(self.okButton)
        layout.addWidget(self.cancelButton)
        self.setLayout(layout)


    def okExit(self):
        try:
            updated_dict = json.loads(self.textEdit.toPlainText())
            self.closed.emit(updated_dict)
            self.user_closed = False
            self.close()
        except json.JSONDecodeError as e:
            QMessageBox.warning(self, "Invalid JSON", f"Error parsing JSON: {e}")

    def cancelExit(self):
        self.close()

    def closeEvent(self, event):
        if self.user_closed:
            self.closed.emit(None)
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = JsonEditor({"hello": "world"})
    editor.show()
    sys.exit(app.exec_())