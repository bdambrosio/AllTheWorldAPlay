import os, sys
import numpy as np
import pandas as pd
import faiss
from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QPushButton, QWidget, QLabel, QLineEdit
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex
from PyQt5.QtGui import QColor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library.semanticScholar3 import papers_library_filepath, paper_index_filepath, section_library_filepath, section_index_filepath


class PandasModel(QAbstractTableModel):
    def __init__(self, data, table_view):
        QAbstractTableModel.__init__(self)
        self._data = data
        self.table_view = table_view
        self.hidden_columns = []

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]-len(self.hidden_columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        # Adjust column index to account for hidden columns
        column = self.visibleColumn(index.column())
        
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), column])
        
        # Add condition for background color
        if role == Qt.BackgroundRole:
            if not self.test_for_sections(index.row()):
                return QColor(Qt.red)
        
        return None

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            # Adjust section index to account for hidden columns
            section = self.visibleColumn(section)
            return self._data.columns[section]
        return None

    def visibleColumn(self, section):
        # Adjust the column index to skip hidden columns
        for hidden_col in self.hidden_columns:
            hidden_index = self._data.columns.get_loc(hidden_col)
            if section >= hidden_index:
                section += 1
        return section

    def test_for_sections(self, row_num):
        sections = section_library_df[section_library_df['paper_id'].astype(str) == str(paper_library_df.iloc[row_num]['faiss_id'])]
        if sections is not None and len(sections) > 0:
            return True
        else:
            return False
        
    def removeRow(self, row, parent=None):
        global paper_indexIDMap, section_library_df, section_indexIDMap
        self.beginRemoveRows(QModelIndex(), row, row)

        df_row = self._data.loc[self._data.index[row]]
        paper_faiss_id = df_row['faiss_id']
        print(f'Removing paper {paper_faiss_id}')

        #first find and remove all sections from section_faiss and section_library_df:
        section_library_rows = section_library_df[section_library_df['paper_id'] == id]
        if section_library_rows is not None:
            section_rows_to_drop = []
            for row in section_library_rows.itertuples():
                section_id = row.faiss_id
                section_rows_to_drop.append(section_id)
            section_ids_to_mask = np.array(section_rows_to_drop, dtype=np.int64)
            section_indexIDMap.remove_ids(section_ids_to_mask)
            section_library_df.drop(section_library_rows.index, inplace=True)

        # now remove paper from paper_library_df and paper_faiss
        print(f'paper row index {df_row.index}')
        self._data.drop(self._data.index[row], inplace=True)
        paper_ids_to_mask = np.array([paper_faiss_id], dtype=np.int64)
        paper_indexIDMap.remove_ids(paper_ids_to_mask)
        
        self.endRemoveRows()
        self.table_view.viewport().update()  # Update viewport
        self.table_view.repaint()  # Force a repaint
        return True

# load library
directory = '../arxiv/'

paper_library_df = pd.read_parquet(papers_library_filepath)
print('loaded paper_library_df')
paper_indexIDMap = faiss.read_index(paper_index_filepath)
print(f"loaded '{paper_index_filepath}'")
section_indexIDMap = faiss.read_index(section_index_filepath)
print(f"loaded '{section_index_filepath}'")
section_library_df = pd.read_parquet(section_library_filepath)
print(f"loaded '{section_library_filepath}'\n  keys: {section_library_df.keys()}")

def save_library():
    faiss.write_index(section_indexIDMap, section_index_filepath)
    section_library_df.to_parquet(section_library_filepath)
    faiss.write_index(paper_indexIDMap, paper_index_filepath)
    paper_library_df.to_parquet(papers_library_filepath)


# Create the application
app = QApplication(sys.argv)
main_widget = QWidget()
main_layout = QVBoxLayout(main_widget)

table_view = QTableView()
model = PandasModel(paper_library_df, table_view)
table_view.setModel(model)
main_layout.addWidget(table_view)
# Set the selection behavior to select entire rows
table_view.setSelectionBehavior(QTableView.SelectRows)

remove_row_button = QPushButton("Remove Selected Row")
def remove_selected_row():
    model.layoutAboutToBeChanged.emit()  # Notify the view to prepare for layout changes
    index_list = table_view.selectionModel().selectedRows()
    rows = sorted(set(index.row() for index in index_list), reverse=True)
    for row in rows:
        model.removeRow(row)
    model.layoutChanged.emit()  # Notify the view that layout changes are done
remove_row_button.clicked.connect(remove_selected_row)
main_layout.addWidget(remove_row_button)

def find_by_id(faiss_id):
    try:
        faiss_id = int(faiss_id)
        matching_rows = paper_library_df[paper_library_df['faiss_id'].astype(str) == str(faiss_id)]
        if not matching_rows.empty:
            row_index = matching_rows.index[0]
            model.layoutAboutToBeChanged.emit()
            table_view.selectRow(row_index)
            table_view.scrollTo(model.index(row_index, 0), QTableView.PositionAtCenter)
            model.layoutChanged.emit()
        else:
            print(f"No matching row found for faiss_id: {faiss_id}")
    except ValueError:
        print("Invalid faiss_id. Please enter a valid integer.")


find_label = QLabel("Find by faiss_id:")
main_layout.addWidget(find_label)

find_input = QLineEdit()
main_layout.addWidget(find_input)

find_button = QPushButton("Find")
find_button.clicked.connect(lambda: find_by_id(find_input.text()))
main_layout.addWidget(find_button)

save_button = QPushButton("Save DataFrame")
save_button.clicked.connect(save_library)
main_layout.addWidget(save_button)

main_widget.setLayout(main_layout)
main_widget.show()
sys.exit(app.exec_())
