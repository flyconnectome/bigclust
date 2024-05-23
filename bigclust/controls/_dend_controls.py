import pyperclip

import numpy as np
import pygfx as gfx

from functools import partial
from PySide6 import QtWidgets, QtCore

# TODOs:
# - add custom legend formatting (e.g. "{object.name}")
# - show type of object in legend
# - add dropdown to manipulate all selected objects
# - add filter to legend (use item.setHidden(True/False) to show/hide items)
# - highlight object in legend when hovered over in scene
# - make legend tabbed (QTabWidget)

CLIO_CLIENT = None
CLIO_ANN = None
FLYWIRE_ANN = None


class DendrogramControls(QtWidgets.QWidget):
    def __init__(self, figure, labels=[], datasets=[], width=200, height=400):
        super().__init__()
        self.figure = figure
        self.labels = labels
        self.datasets = datasets
        self.setWindowTitle("Controls")
        self.resize(width, height)

        self.tab_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.tab_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabs.setMovable(True)

        self.tab_layout.addWidget(self.tabs)

        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab1_layout = QtWidgets.QVBoxLayout()
        self.tab2_layout = QtWidgets.QVBoxLayout()
        self.tab1.setLayout(self.tab1_layout)
        self.tab2.setLayout(self.tab2_layout)

        self.tabs.addTab(self.tab1, "General")
        self.tabs.addTab(self.tab2, "Annotation")

        # Build gui
        self.build_control_gui()
        self.build_annotation_gui()

    def build_control_gui(self):
        """Build the GUI."""
        # Search bar
        self.search_text = QtWidgets.QLabel("Search")
        self.tab1_layout.addWidget(self.search_text)
        self.searchbar = QtWidgets.QLineEdit()
        self.searchbar.returnPressed.connect(self.find_next)
        # self.searchbar.textChanged.connect(self.figure.highlight_cluster)
        self.searchbar_completer = QtWidgets.QCompleter(self.labels)
        self.searchbar_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.searchbar.setCompleter(self.searchbar_completer)
        self.tab1_layout.addWidget(self.searchbar)

        # Add buttons for previous/next
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.clicked.connect(self.find_next)
        self.tab1_layout.addWidget(self.next_button)
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.clicked.connect(self.find_previous)
        self.tab1_layout.addWidget(self.prev_button)

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add the dropdown to action all selected objects
        self.sel_text = QtWidgets.QLabel("Selected:")
        self.tab1_layout.addWidget(self.sel_text)
        self.sel_action = QtWidgets.QPushButton(text="Pick action")
        self.tab1_layout.addWidget(self.sel_action)
        self.sel_action_menu = QtWidgets.QMenu(self)
        self.sel_action.setMenu(self.sel_action_menu)

        # Set actions for the dropdown
        self.sel_action_menu.addAction("New Random")
        # self.sel_action_menu.actions()[-1].triggered.connect(self.hide_selected)
        self.sel_action_menu.addAction("No cluster")
        # self.sel_action_menu.actions()[-1].triggered.connect(self.show_selected)
        self.sel_action_menu.addAction("Merge clusters")
        # self.sel_action_menu.actions()[-1].triggered.connect(self.show_selected)
        self.sel_action_menu.addAction("Open URL")
        # self.sel_action_menu.actions()[-1].triggered.connect(self.show_selected)

        # Add the dropdown to action copy to clipboard
        self.sel_clipboard_action = QtWidgets.QPushButton(text="To clipboard")
        self.tab1_layout.addWidget(self.sel_clipboard_action)
        self.sel_clipboard_action_menu = QtWidgets.QMenu(self)
        self.sel_clipboard_action.setMenu(self.sel_clipboard_action_menu)

        # Set actions for the clipboard dropdown
        self.sel_clipboard_action_menu.addAction("All")
        self.sel_clipboard_action_menu.actions()[-1].triggered.connect(
            self.selected_to_clipboard
        )

        # A dictionary mapping the button to the corresponding dataset(s)
        datasets = {ds: ds for ds in self.datasets}
        datasets.update(
            {ds[:-1]: (ds[:-1] + "R", ds[:-1] + "L") for ds in self.datasets if ds[-1] in "LR"}
        )
        for ds in sorted(list(datasets)):
            self.sel_clipboard_action_menu.addAction(f"{ds} only")
            self.sel_clipboard_action_menu.actions()[-1].triggered.connect(
                partial(self.selected_to_clipboard, dataset=datasets[ds])
            )

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add dropdown to choose color mode
        self.color_text = QtWidgets.QLabel("Color neurons by:")
        self.tab1_layout.addWidget(self.color_text)
        self.color_combo_box = QtWidgets.QComboBox()
        self.color_combo_box.addItem("Default")
        self.color_combo_box.addItem("Dataset")
        self.color_combo_box.addItem("Cluster")
        self.color_combo_box.addItem("Label")
        self.color_combo_box.setItemData(
            0, "Color neurons by dataset", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            1, "Color neurons by cluster", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            2, "Color neurons by label", QtCore.Qt.ToolTipRole
        )
        self.tab1_layout.addWidget(self.color_combo_box)

        # Set the action for the color combo box
        self.color_combo_box.currentIndexChanged.connect(self.set_color_mode)

        self.add_group_check = QtWidgets.QCheckBox("Add as group")
        self.add_group_check.setToolTip("Whether to add neurons as group when selected")
        self.add_group_check.setChecked(False)
        self.add_group_check.stateChanged.connect(self.set_add_group)
        self.tab1_layout.addWidget(self.add_group_check)

        # This would make it so the legend does not stretch when
        # we resize the window vertically
        self.tab1_layout.addStretch(1)

        return

    def build_annotation_gui(self):
        # Add buttons to push annotations
        self.push_ann_button = QtWidgets.QPushButton("Push to Clio")
        self.push_ann_button.setToolTip("Push the current annotation to Clio")
        self.push_ann_button.clicked.connect(self.push_annotation)
        self.tab2_layout.addWidget(self.push_ann_button)

        # Add checkboxes
        self.set_flywire_check = QtWidgets.QCheckBox("Set flywire type")
        self.set_flywire_check.setToolTip("Set the `flywire_type` field in the annotation")
        self.set_flywire_check.setChecked(True)
        self.tab2_layout.addWidget(self.set_flywire_check)
        self.set_type_check = QtWidgets.QCheckBox("Set type")
        self.set_type_check.setToolTip("Set the `type` field in the annotation")
        self.tab2_layout.addWidget(self.set_type_check)
        self.set_sanity_check = QtWidgets.QCheckBox("Sanity checks")
        self.set_sanity_check.setToolTip("Whether to perform sanity checks")
        self.set_sanity_check.setChecked(True)
        self.tab2_layout.addWidget(self.set_sanity_check)

        # This makes it so the legend does not stretch
        self.tab2_layout.addStretch(1)

    def add_split(self, layout):
        """Add horizontal divider."""
        # layout.addSpacing(5)
        layout.addWidget(QHLine())
        # layout.addSpacing(5)

    def push_annotation(self):
        """Push the current annotation to Clio."""
        selected_labels = self.figure.selected_labels
        selected_ids = self.figure.selected_ids
        if selected_ids is None:
            self.figure.show_message("No selection", color="red", duration=2)
            return

        # See if it's obvious which labels to push
        selected_labels = set(selected_labels)
        selected_labels -= {"untyped", "untyped*"}

        if len(selected_labels) == 0:
            self.figure.show_message("No labels to push", color="red", duration=2)
            return

        if len(selected_labels) > 1:
            self.figure.show_message("Multiple labels selected", color="red", duration=2)
            print("Multiple labels selected", selected_labels)
            return

        # Get the label
        label = selected_labels.pop()

        # Get the male CNS IDs
        from fafbseg import flywire
        bodyids = selected_ids[~flywire.is_valid_root(selected_ids)]

        if len(bodyids) == 0:
            self.figure.show_message("No male CNS neurons selected", color="red", duration=2)
            return

        # Get the annotation
        import clio
        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset='CNS')

        if self.set_sanity_check.isChecked():
            import cocoa as cc
            global CLIO_ANN
            if CLIO_ANN is None:
                print("Fetching Clio annotations...")
                CLIO_ANN = cc.MaleCNS().get_annotations()

            global FLYWIRE_ANN
            if FLYWIRE_ANN is None:
                FLYWIRE_ANN = cc.FlyWire(live_annot=True).get_annotations()

            for t in label.split(','):
                if t not in FLYWIRE_ANN.cell_type.unique() and t not in FLYWIRE_ANN.hemibrain_type.unique():
                    self.figure.show_message(f'Label {t} not found in FlyWire annotations', color="red", duration=2)

            if label in CLIO_ANN.flywire_type.values:
                self.figure.show_message(f'Label {label} already in Clio', color="red", duration=2)
                return

            # Update the annotations
            if self.set_flywire_check.isChecked():
                CLIO_ANN.loc[CLIO_ANN.bodyId.isin(bodyids), 'flywire_type'] = label
            if self.set_type_check.isChecked():
                CLIO_ANN.loc[CLIO_ANN.bodyId.isin(bodyids), 'type'] = label

        if self.set_flywire_check.isChecked():
            clio.set_fields(bodyids, flywire_type=label)
        if self.set_type_check.isChecked():
             clio.set_fields(bodyids, type=label)

        self.figure.show_message(f"Set {label} for {len(bodyids)} neurons", color="lightgreen", duration=2)
        print(f"Set {label} for {len(bodyids)} neurons:", bodyids)

    def set_add_group(self):
        """Set whether to add neurons as group when selected."""
        self.figure._add_as_group = self.add_group_check.isChecked()

    def find_next(self):
        """Find next occurrence."""
        text = self.searchbar.text()
        if text:
            if not hasattr(self, "_label_search") or self._label_search.label != text:
                self._label_search = self.figure.find_label(text)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                self._label_search.next()

    def find_previous(self):
        """Find previous occurrence."""
        text = self.searchbar.text()
        if text:
            if not hasattr(self, "_label_search") or self._label_search.label != text:
                self._label_search = self.figure.find_label(text)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                self._label_search.prev()

    def selected_to_clipboard(self, dataset=None):
        """Copy selected items to clipboard."""
        if self.figure.selected is not None:
            indices = self.figure._leafs_order[self.figure._selected]

            if isinstance(dataset, str):
                indices = [i for i in indices if self.figure._leaf_types[i] == dataset]
            elif isinstance(dataset, (list, set, tuple)):
                indices = [i for i in indices if self.figure._leaf_types[i] in dataset]

            ids = self.figure._ids[indices]
            pyperclip.copy(",".join(np.array(ids).astype(str)))

    def set_color_mode(self):
        """Set the color mode."""
        mode = self.color_combo_box.currentText()
        self.figure.set_viewer_color_mode(mode.lower())

    def close(self):
        """Close the controls."""
        super().close()


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
