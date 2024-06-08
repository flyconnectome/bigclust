import pyperclip

import numpy as np
import pandas as pd

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list

# See here for a good tutorial on tables:
# https://www.pythonguis.com/tutorials/pyside6-qtableview-modelviews-numpy-pandas/


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
        self._data = data
        self._view = self._data.iloc[0:0, 0:0]  # start with an empty view
        self.update_indices()  # pre-compute the indices and columns
        self._hide_zeros = True
        self._synapse_threshold = 1
        self._col_sort = None
        self._col_filt = None
        self._upstream = True
        self._downstream = True

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list)
            value = self._view.values[index.row(), index.column()]
            if self._hide_zeros and value == 0:
                return ""
            else:
                return str(value)
        elif role == Qt.TextAlignmentRole:
            # Center the text in both vertical and horizontal directions
            return Qt.AlignCenter + Qt.AlignVCenter

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._view)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return self._view.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._columns[section])

            if orientation == Qt.Vertical:
                return str(self._indices[section])

    def select_rows(self, indices, drop_empty_cols=True, use_index=True):
        """Select rows by indices."""
        if use_index:
            self._view = self._data.loc[indices]
        else:
            self._view = self._data.iloc[indices]

        if isinstance(self._view.columns, pd.MultiIndex):
            if not self._upstream:
                self._view = self._view.drop(columns=["upstream"])
            if not self._downstream:
                self._view = self._view.drop(columns=["downstream"])

        # Apply synapse threshold
        if self._synapse_threshold:
            self._view = self._view.iloc[
                :, self._view.max(axis=0).values >= self._synapse_threshold
            ]

        # Apply column filter
        if self._col_filt:
            self._view = self._view.filter(regex=self._col_filt, axis=1)

        # Apply column sort
        if self._col_sort not in (None, "No sort") and self._view.shape[1] > 1:
            if self._col_sort == "By synapse count":
                srt = np.argsort(self._view.sum(axis=0).values)[::-1]
            elif self._col_sort == "By label":
                if isinstance(self._view.columns, pd.MultiIndex):
                    # Ignore upstream/downstream and sort by label
                    _, srt = self._view.columns.sortlevel(1)
                else:
                    srt = np.argsort(self._view.columns)
            elif self._col_sort == "By distance":
                # Sort by Euclidean distance
                d = pdist(self._view.T)
                srt = leaves_list(linkage(d, method="ward"))
            else:
                raise ValueError(f"Unknown sort order: {self._col_sort}")

            self._view = self._view.iloc[:, srt]

        self.update_indices()

        # Emit signal to trigger update
        self.layoutChanged.emit()

    def set_synapse_threshold(self, threshold):
        """Set the synapse threshold."""
        self._synapse_threshold = threshold
        self.select_rows(self._view.index)  # reselect rows

    def set_hide_zeros(self, hide_zeros):
        """Set whether to hide zeros."""
        self._hide_zeros = hide_zeros

        # Emit signal to trigger update
        self.layoutChanged.emit()

    def set_col_sort(self, sort):
        """Set the column sort."""
        self._col_sort = sort
        self.select_rows(self._view.index)  # reselect rows

    def set_filter_columns(self, filt):
        """Set the column filter."""
        self._col_filt = filt
        self.select_rows(self._view.index)

    def set_direction(self, upstream=True, downstream=True):
        """Set the direction of the table."""
        self._upstream = upstream
        self._downstream = downstream
        self.select_rows(self._view.index)

    def update_indices(self):
        """Update the indices and columns according to the current view."""
        SHORT = {"upstream": "up", "downstream": "ds"}

        def fmt(i):
            return f"{SHORT[i[0]]}:{i[1]}"

        if isinstance(self._view.index, pd.MultiIndex):
            self._indices = [fmt(i) for i in self._view.index.to_flat_index()]
        else:
            self._indices = self._view.index.astype(str)

        if isinstance(self._view.columns, pd.MultiIndex):
            self._columns = [fmt(i) for i in self._view.columns.to_flat_index()]
        else:
            self._columns = self._view.columns.astype(str)


class ConnectivityTable(QtWidgets.QWidget):
    """A widget to display a table of connectivity data.

    Parameters
    ----------
    data : pd.DataFrame
        The connectivity data to display.
    figure : Dendrogram, optional
        The dendrogram figure to connect to.
    width : int, optional
        The width of the widget.
    height : int, optional
        The height of the widget.

    """

    def __init__(self, data, figure=None, width=600, height=400):
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"

        super().__init__()
        # Use this to keep the window on top
        # (should make this toggleable)
        #self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        self._data = data
        self._figure = figure
        self.setWindowTitle("Connectivity")
        self.resize(width, height)

        # Set up layout
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # Build gui

        # First up: the table
        self._table = QtWidgets.QTableView()
        self._model = TableModel(self._data)
        self._table.setModel(self._model)
        self._layout.addWidget(self._table)

        # Add a double click event for header and rows
        self._table.horizontalHeader().sectionDoubleClicked.connect(self.find_header)
        self._table.verticalHeader().sectionDoubleClicked.connect(self.find_index)

        # For the control panel we want another layout
        self._control_layout = QtWidgets.QHBoxLayout()
        self._layout.addLayout(self._control_layout)

        # Add checkboxes for up- and downstream
        self._upstream = QtWidgets.QCheckBox("Upstream")
        self._upstream.setChecked(True)
        self._upstream.setToolTip("Show upstream connections")
        self._upstream.stateChanged.connect(self.update_direction)
        self._control_layout.addWidget(self._upstream)

        self._downstream = QtWidgets.QCheckBox("Downstream")
        self._downstream.setChecked(True)
        self._downstream.setToolTip("Show downstream connections")
        self._downstream.stateChanged.connect(self.update_direction)
        self._control_layout.addWidget(self._downstream)

        self._hide_zeros = QtWidgets.QCheckBox("Hide zeros")
        self._hide_zeros.setChecked(True)
        self._hide_zeros.setToolTip("Hide zero values")
        self._hide_zeros.stateChanged.connect(self.update_hide_zeros)
        self._control_layout.addWidget(self._hide_zeros)

        # Add a QSpinBox for the synapse threshold
        self._synapse_threshold = QtWidgets.QSpinBox()
        self._synapse_threshold.setToolTip("Set the synapse threshold")
        self._synapse_threshold.setRange(0, 1000)
        self._synapse_threshold.setValue(1)
        self._synapse_threshold.setSingleStep(1)
        self._synapse_threshold.valueChanged.connect(self.update_synapse_threshold)
        self._control_layout.addWidget(self._synapse_threshold)

        # Add a button to copy to clipboard
        self._copy_button = QtWidgets.QPushButton("Copy to clipboard")
        self._copy_button.setToolTip("Copy the current view to the clipboard")
        self._copy_button.clicked.connect(self.copy_to_clipboard)
        self._control_layout.addWidget(self._copy_button)

        # Start a new row in the layout
        self._control_layout2 = QtWidgets.QHBoxLayout()
        self._layout.addLayout(self._control_layout2)
        self._control_layout2.addWidget(QtWidgets.QLabel("Columns:"))

        # Add dropdown selection for column sorting
        self._sort_dropdown = QtWidgets.QComboBox()
        self._sort_dropdown.addItem("No sort")
        self._sort_dropdown.addItem("By synapse count")
        self._sort_dropdown.addItem("By label")
        self._sort_dropdown.addItem("By distance")
        self._sort_dropdown.currentIndexChanged.connect(self.update_sort)
        self._control_layout2.addWidget(self._sort_dropdown)

        # Add a field for filtering the columns
        self._search = QtWidgets.QLineEdit()
        self._search.setToolTip("Filter columns by name")
        self._search.setPlaceholderText("Filter")
        self._search.textChanged.connect(self.filter_columns)
        self._control_layout2.addWidget(self._search)

        # TODOs:
        # add toggles for:
        # - setting colors (perhaps based on dendrogram)
        # - toggle for normalized weight

    def update_synapse_threshold(self):
        """Update the synapse threshold."""
        self._model.set_synapse_threshold(self._synapse_threshold.value())

    def update_hide_zeros(self):
        """Update the hide zeros setting."""
        self._model.set_hide_zeros(self._hide_zeros.isChecked())

    def update_sort(self):
        """Update the sorting of the table."""
        self._model.set_col_sort(self._sort_dropdown.currentText())

    def update_direction(self):
        """Update the direction of the table."""
        self._model.set_direction(
            upstream=self._upstream.isChecked(), downstream=self._downstream.isChecked()
        )

    def filter_columns(self):
        """Filter the columns based on the search field."""
        search = self._search.text()
        self._model.set_filter_columns(search)

    def select(self, ids):
        """Select rows by IDs."""
        self._model.select_rows(ids, use_index=False)

    def find_header(self):
        """Find the currently selected header."""
        curr_col = self._table.currentIndex().column()
        label = self._model._view.columns[curr_col]

        # Drop the "upstream" or "downstream" prefix
        # if this is a multi-index
        if isinstance(label, tuple):
            label = label[1]

        if self._figure:
            self._figure.find_label(label, regex=True)

    def find_index(self):
        """Find the currently selected index."""
        curr_row = self._table.currentIndex().row()
        label = self._model._view.index[curr_row]

        if self._figure:
            self._figure.find_label(label, regex=True)

    def copy_to_clipboard(self):
        """Copy the table to the clipboard."""
        # Let's enforce some sensible limits to how many rows we can copy
        if self._model._view.shape[0] > 200:
            raise ValueError("Too many rows to copy to clipboard.")

        self._model._view.to_clipboard(index=True)
