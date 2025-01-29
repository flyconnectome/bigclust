import re
import os
import uuid
import pyperclip
import traceback

import pandas as pd
import numpy as np

from functools import partial
from PySide6 import QtWidgets, QtCore
from concurrent.futures import ThreadPoolExecutor

# TODOs:
# - add custom legend formatting (e.g. "{object.name}")
# - show type of object in legend
# - add dropdown to manipulate all selected objects
# - add filter to legend (use item.setHidden(True/False) to show/hide items)
# - highlight object in legend when hovered over in scene
# - make legend tabbed (QTabWidget)

CLIO_CLIENT = None
CLIO_ANN = None
NEUPRINT_CLIENT = None
FLYWIRE_ANN = None
HB_ANN = None


def requires_selection(func):
    """Decorator to check if a selection is required."""

    def wrapper(self, *args, **kwargs):
        if self.figure.selected_ids is None or len(self.figure.selected_ids) == 0:
            self.figure.show_message("No neurons selected", color="red", duration=2)
            return
        return func(self, *args, **kwargs)

    return wrapper


class DendrogramControls(QtWidgets.QWidget):
    def __init__(self, figure, labels=[], datasets=[], width=200, height=400):
        super().__init__()
        self.figure = figure
        self.labels = labels
        self.datasets = datasets
        self.setWindowTitle("Controls")
        self.resize(width, height)
        self.label_overrides = {}

        self.tab_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.tab_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabs.setMovable(True)

        self.tab_layout.addWidget(self.tabs)

        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()
        self.tab1_layout = QtWidgets.QVBoxLayout()
        self.tab2_layout = QtWidgets.QVBoxLayout()
        self.tab3_layout = QtWidgets.QVBoxLayout()
        self.tab1.setLayout(self.tab1_layout)
        self.tab2.setLayout(self.tab2_layout)
        self.tab3.setLayout(self.tab3_layout)

        self.tabs.addTab(self.tab1, "General")
        self.tabs.addTab(self.tab2, "Annotation")
        self.tabs.addTab(self.tab3, "Neuroglancer")

        # Deactivate tabs
        if not os.environ.get("BC_ANNOTATION", "0") == "1":
            self.tabs.setTabEnabled(1, False)
        if not hasattr(self.figure, "_ngl_viewer"):
            self.tabs.setTabEnabled(2, False)

        # Build gui
        self.build_control_gui()
        self.build_annotation_gui()
        self.build_neuroglancer_gui()

        # Holds the futures for requested data
        self.futures = {}
        self.pool = ThreadPoolExecutor(4)

    def build_control_gui(self):
        """Build the GUI."""
        # Search bar
        self.search_text = QtWidgets.QLabel("Search")
        self.tab1_layout.addWidget(self.search_text)
        self.searchbar = QtWidgets.QLineEdit()
        self.searchbar.setToolTip(
            "Search for a label in the scene. Use a leading '/' to search for a regex."
        )
        self.searchbar.returnPressed.connect(self.find_next)
        # self.searchbar.textChanged.connect(self.figure.highlight_cluster)
        self.searchbar_completer = QtWidgets.QCompleter(self.labels)
        self.searchbar_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.searchbar.setCompleter(self.searchbar_completer)
        self.tab1_layout.addWidget(self.searchbar)

        # Add buttons for previous/next
        self.button_layout = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.clicked.connect(self.find_previous)
        self.button_layout.addWidget(self.prev_button)
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.clicked.connect(self.find_next)
        self.button_layout.addWidget(self.next_button)
        self.tab1_layout.addLayout(self.button_layout)

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add the dropdown to action all selected objects
        self.selection_layout = QtWidgets.QHBoxLayout()
        self.tab1_layout.addLayout(self.selection_layout)
        self.sel_text = QtWidgets.QLabel("Selected:")
        self.selection_layout.addWidget(self.sel_text)
        self.sel_action = QtWidgets.QPushButton(text="Pick action")
        self.sel_action_menu = QtWidgets.QMenu(self)
        self.sel_action.setMenu(self.sel_action_menu)

        # Set actions for the dropdown
        self.sel_action_menu.addAction("(New Random)")
        # self.sel_action_menu.actions()[-1].triggered.connect(self.hide_selected)
        self.sel_action_menu.addAction("(No cluster)")
        # self.sel_action_menu.actions()[-1].triggered.connect(self.show_selected)
        self.sel_action_menu.addAction("(Merge clusters)")
        # self.sel_action_menu.actions()[-1].triggered.connect(self.show_selected)
        self.sel_action_menu.addAction("(Open URL)")
        # self.sel_action_menu.actions()[-1].triggered.connect(self.show_selected)

        # Add the dropdown to action copy to clipboard
        self.sel_clipboard_action = QtWidgets.QPushButton(text="To clipboard")
        self.selection_layout.addWidget(self.sel_clipboard_action)
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
            {
                ds[:-1]: (ds[:-1] + "R", ds[:-1] + "L")
                for ds in self.datasets
                if ds[-1] in "LR"
            }
        )
        for ds in sorted(list(datasets)):
            self.sel_clipboard_action_menu.addAction(f"{ds} only")
            self.sel_clipboard_action_menu.actions()[-1].triggered.connect(
                partial(self.selected_to_clipboard, dataset=datasets[ds])
            )

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add dropdown to choose leaf labels
        self.label_layout = QtWidgets.QHBoxLayout()
        self.tab1_layout.addLayout(self.label_layout)
        self.dend_labels = QtWidgets.QLabel("Labels:")
        self.label_layout.addWidget(self.dend_labels)
        self.label_combo_box = QtWidgets.QComboBox()
        self.label_combo_box.addItem("Default")
        for col in self.figure._table.columns:
            self.label_combo_box.addItem(col)
        self.label_layout.addWidget(self.label_combo_box)
        self.label_combo_box.currentIndexChanged.connect(self.set_leaf_labels)
        self._current_leaf_labels = self.label_combo_box.currentText()

        self.label_count_check = QtWidgets.QCheckBox("Add label counts")
        self.label_count_check.setToolTip("Whether to add counts to the labels")
        self.label_count_check.setChecked(False)
        self.label_count_check.stateChanged.connect(self.set_label_counts)
        self.tab1_layout.addWidget(self.label_count_check)

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
        self.color_combo_box.addItem("Random")
        self.color_combo_box.setItemData(
            0, "Color neurons by viewer default", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            1, "Color neurons by dataset", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            2, "Color neurons by cluster", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            3, "Color neurons by label", QtCore.Qt.ToolTipRole
        )
        self.color_combo_box.setItemData(
            4, "Randomly color neurons", QtCore.Qt.ToolTipRole
        )
        self.tab1_layout.addWidget(self.color_combo_box)

        # Set the action for the color combo box
        self.color_combo_box.currentIndexChanged.connect(self.set_color_mode)

        self.add_group_check = QtWidgets.QCheckBox("Add as group")
        self.add_group_check.setToolTip("Whether to add neurons as group when selected")
        self.add_group_check.setChecked(False)
        self.add_group_check.stateChanged.connect(self.set_add_group)
        self.tab1_layout.addWidget(self.add_group_check)

        self.dclick_deselect = QtWidgets.QCheckBox("Deselect on double-click")
        self.dclick_deselect.setToolTip("You can always deselect using ESC")
        self.dclick_deselect.setChecked(self.figure.deselect_on_dclick)
        self.dclick_deselect.stateChanged.connect(self.set_dclick_deselect)
        self.tab1_layout.addWidget(self.dclick_deselect)

        self.empty_deselect = QtWidgets.QCheckBox("Deselect on empty selection")
        self.empty_deselect.setToolTip("You can always deselect using ESC")
        self.empty_deselect.setChecked(self.figure.deselect_on_empty)
        self.empty_deselect.stateChanged.connect(self.set_empty_deselect)
        self.tab1_layout.addWidget(self.empty_deselect)

        # This would make it so the legend does not stretch when
        # we resize the window vertically
        self.tab1_layout.addStretch(1)

        return

    def build_neuroglancer_gui(self):
        # Add buttons to generate neuroglancer scene
        self.ngl_open_button = QtWidgets.QPushButton("Open in browser")
        self.ngl_open_button.setToolTip(
            "Open the current scene in a new browser window"
        )
        self.ngl_open_button.clicked.connect(self.ngl_open)
        self.tab3_layout.addWidget(self.ngl_open_button)

        self.ngl_copy_button = QtWidgets.QPushButton("Copy to clipboard")
        self.ngl_copy_button.setToolTip("Copy the current scene to the clipboard")
        self.ngl_copy_button.clicked.connect(self.ngl_copy)
        self.tab3_layout.addWidget(self.ngl_copy_button)

        # This makes it so the legend does not stretch
        self.tab3_layout.addStretch(1)

    def build_annotation_gui(self):
        # Add buttons to push annotations
        self.push_ann_button = QtWidgets.QPushButton("Push annotations")
        self.push_ann_button.setToolTip(
            "Push the current annotation to selected fields"
        )
        self.push_ann_button.clicked.connect(self.push_annotation)
        self.tab2_layout.addWidget(self.push_ann_button)

        self.ann_combo_box = QtWidgets.QComboBox()
        self.ann_combo_box.setEditable(True)
        self.tab2_layout.addWidget(self.ann_combo_box)

        self.clear_ann_button = QtWidgets.QPushButton("Clear annotations")
        self.clear_ann_button.setToolTip("Clear the current annotations")
        self.clear_ann_button.clicked.connect(self.clear_annotation)
        self.tab2_layout.addWidget(self.clear_ann_button)

        # Add checkboxes
        self.set_label = QtWidgets.QLabel("Which fields to set:")
        self.tab2_layout.addWidget(self.set_label)

        self.set_type_check = QtWidgets.QCheckBox("Clio: type")
        self.set_type_check.setToolTip("Set the `type` field in Clio")
        self.tab2_layout.addWidget(self.set_type_check)

        self.set_flywire_check = QtWidgets.QCheckBox("Clio: flywire_type")
        self.set_flywire_check.setToolTip("Set the `flywire_type` field in Clio")
        self.set_flywire_check.setChecked(False)
        self.tab2_layout.addWidget(self.set_flywire_check)

        self.set_manc_check = QtWidgets.QCheckBox("Clio: manc_type")
        self.set_manc_check.setToolTip("Set the `manc_type` field in Clio")
        self.set_manc_check.setChecked(False)
        self.tab2_layout.addWidget(self.set_manc_check)

        self.set_mcns_type_check = QtWidgets.QCheckBox("FlyTable: malecns_type")
        self.set_mcns_type_check.setToolTip("Set the `malecns_type` field in FlyTable")
        self.tab2_layout.addWidget(self.set_mcns_type_check)

        self.set_label2 = QtWidgets.QLabel("Settings:")
        self.tab2_layout.addWidget(self.set_label2)

        self.set_sanity_check = QtWidgets.QCheckBox("Sanity checks")
        self.set_sanity_check.setToolTip("Whether to perform sanity checks")
        self.set_sanity_check.setChecked(True)
        self.tab2_layout.addWidget(self.set_sanity_check)

        # Add dropdown to set dimorphism status
        self.sel_dimorphism_action = QtWidgets.QPushButton(text="Set dimorphism")
        self.tab2_layout.addWidget(self.sel_dimorphism_action)
        self.sel_dimorphism_action_menu = QtWidgets.QMenu(self)
        self.sel_dimorphism_action.setMenu(self.sel_dimorphism_action_menu)

        # Set actions for the clipboard dropdown
        self.sel_dimorphism_action_menu.addAction("Sex-specific")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism("sex-specific")
        )
        self.sel_dimorphism_action_menu.addAction("Sexually dimorphic")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism("sexually dimorphic")
        )
        self.sel_dimorphism_action_menu.addAction("Pot. sex-specific")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism("potentially sex-specific")
        )
        self.sel_dimorphism_action_menu.addAction("Pot. sexually dimorphic")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism("potentially sexually dimorphic")
        )
        self.sel_dimorphism_action_menu.addAction("Not dimorphic")
        self.sel_dimorphism_action_menu.actions()[-1].triggered.connect(
            lambda x: self.selected_set_dimorphism(None)
        )

        # Make a separate layout with tighter margins for the buttons
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setSpacing(0)  # No space between buttons
        button_layout.setContentsMargins(0, 0, 0, 0)  # No margins around the buttons
        self.tab2_layout.addLayout(button_layout)

        # Add button to set new Clio group
        self.clio_group_button = QtWidgets.QPushButton("Set new Clio group")
        self.clio_group_button.setToolTip(
            "Assign new Clio group. This will use the lowest body ID as group ID."
        )
        self.clio_group_button.clicked.connect(self.new_clio_group)
        button_layout.addWidget(self.clio_group_button)

        # Add button to suggest new MCNS type
        self.suggest_type_button = QtWidgets.QPushButton("Suggest male-only type")
        self.suggest_type_button.setToolTip(
            "Suggest new male-only type based on main input neuropil(s). See console for output."
        )
        self.suggest_type_button.clicked.connect(self.suggest_type)
        button_layout.addWidget(self.suggest_type_button)

        # Add button to suggest new CB type
        self.suggest_cb_type_button = QtWidgets.QPushButton("Suggest new CB-type")
        self.suggest_cb_type_button.setToolTip("Suggest new CBXXXX type.")
        self.suggest_cb_type_button.clicked.connect(self.suggest_cb_type)
        button_layout.addWidget(self.suggest_cb_type_button)

        # Add button to set new super type
        self.set_supertype_button = QtWidgets.QPushButton("Set new SuperType")
        self.set_supertype_button.setToolTip(
            "Assign selected neurons to a supertype. This will use the lowest ID as supertype ID."
        )
        self.set_supertype_button.clicked.connect(self.new_super_type)
        button_layout.addWidget(self.set_supertype_button)

        # This makes it so the legend does not stretch
        self.tab2_layout.addStretch(1)

    def add_split(self, layout):
        """Add horizontal divider."""
        # layout.addSpacing(5)
        layout.addWidget(QHLine())
        # layout.addSpacing(5)

    def update_ann_combo_box(self):
        """Update the items in the annotation combo box."""
        # First clear all existing items
        self.ann_combo_box.clear()

        if self.figure.selected_labels is None:
            return

        # Now add the new items currently selected
        for label in sorted(list(set(self.figure.selected_labels))):
            if re.match(".*?\([0-9]+\)", label):
                label = label.split("(")[0]

            # Replace the "*"
            label = label.replace("*", "")

            if label in ("untyped",):
                continue
            self.ann_combo_box.addItem(label)

    @requires_selection
    def selected_set_dimorphism(self, dimorphism):
        """Push dimorphism to Clio/FlyTable."""
        assert dimorphism in (
            "sex-specific",
            "sexually dimorphic",
            "potentially sex-specific",
            "potentially sexually dimorphic",
            None,
        )
        selected_ids = self.figure.selected_ids

        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # Get the annotation
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[(dimorphism, uuid.uuid4())] = self.pool.submit(
            _push_dimorphism,
            dimorphism=dimorphism,
            bodyids=bodyids,
            rootids=rootids,
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
        )

    @requires_selection
    def push_annotation(self):
        """Push the current annotation to Clio/FlyTable."""
        if not any(
            (
                self.set_flywire_check.isChecked(),
                self.set_type_check.isChecked(),
                self.set_mcns_type_check.isChecked(),
                self.set_manc_check.isChecked(),
            )
        ):
            self.figure.show_message("No fields to push", color="red", duration=2)
            return

        label = self.ann_combo_box.currentText()
        if not label:
            self.figure.show_message("No label to push", color="red", duration=2)
            return

        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # Get the annotation
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[(label, uuid.uuid4())] = self.pool.submit(
            _push_annotations,
            label=label,
            bodyids=bodyids
            if self.set_flywire_check.isChecked()
            or self.set_type_check.isChecked()
            or self.set_manc_check.isChecked()
            else None,
            rootids=rootids if self.set_mcns_type_check.isChecked() else None,
            set_flywire=self.set_flywire_check.isChecked(),
            set_type=self.set_type_check.isChecked(),
            set_mcns_type=self.set_mcns_type_check.isChecked(),
            set_manc_type=self.set_manc_check.isChecked(),
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
            controls=self,
        )

        if self.set_type_check.isChecked() and len(bodyids) and CLIO_ANN is not None:
            # Update the CLIO annotations
            CLIO_ANN.loc[
                CLIO_ANN.get("bodyId", CLIO_ANN.get("bodyid")).isin(bodyids), "type"
            ] = label

    @requires_selection
    def clear_annotation(self):
        """Clear the currently selected fields."""
        if not any(
            (
                self.set_flywire_check.isChecked(),
                self.set_type_check.isChecked(),
                self.set_mcns_type_check.isChecked(),
                self.set_manc_check.isChecked(),
            )
        ):
            self.figure.show_message("No fields to clear", color="red", duration=2)
            return

        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # Get the annotation
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[uuid.uuid4()] = self.pool.submit(
            _clear_annotations,
            bodyids=bodyids
            if self.set_flywire_check.isChecked()
            or self.set_type_check.isChecked()
            or self.set_manc_check.isChecked()
            else None,
            rootids=rootids if self.set_mcns_type_check.isChecked() else None,
            clear_flywire=self.set_flywire_check.isChecked(),
            clear_type=self.set_type_check.isChecked(),
            clear_mcns_type=self.set_mcns_type_check.isChecked(),
            clear_manc_type=self.set_manc_check.isChecked(),
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
            controls=self,
        )

    @requires_selection
    def new_super_type(self):
        """Set a new super type for given IDs."""
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # New type name
        new_type = min(selected_ids)

        # Get the clio module
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[(new_type, uuid.uuid4())] = self.pool.submit(
            _push_super_type,
            super_type=new_type,
            bodyids=bodyids,
            rootids=rootids,
            clio=clio,  #  pass the module
            sanity_checks=self.set_sanity_check.isChecked(),
            ftu=ftu,
            figure=self.figure,
        )

    @requires_selection
    def new_clio_group(self):
        """Set a new Clio group for given IDs."""
        # MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        _, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        if not len(bodyids):
            self.figure.show_message(
                "No MCNS neurons selected", color="red", duration=2
            )
            return

        group = min(bodyids)

        # Get the clio module
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        # Submit the annotations
        self.futures[(group, uuid.uuid4())] = self.pool.submit(
            _push_group,
            group=group,
            bodyids=bodyids,
            clio=clio,  #  pass the module
            figure=self.figure,
        )

    @requires_selection
    def suggest_type(self):
        """Suggest a new male-only type for given IDs."""
        selected_ids = self.figure.selected_ids
        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        _, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        if not len(bodyids):
            self.figure.show_message(
                "No MCNS neurons selected", color="red", duration=2
            )
            return

        # Threading this doesn't make much sense
        suggest_new_label(bodyids=bodyids)

    def suggest_cb_type(self):
        """Suggest a new CB type."""
        # Threading this doesn't make much sense
        import ftu

        print("Next free CB tyoe:", ftu.info.get_next_cb_id())

    def set_add_group(self):
        """Set whether to add neurons as group when selected."""
        self.figure._add_as_group = self.add_group_check.isChecked()

    def set_dclick_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_dclick = self.dclick_deselect.isChecked()

    def set_empty_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_empty = self.empty_deselect.isChecked()

    def set_label_counts(self):
        """Set whether to add counts to the labels."""
        self.set_leaf_labels()  # Update the labels

    def find_next(self):
        """Find next occurrence."""
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if (
                not hasattr(self, "_label_search")
                or self._label_search.label != text
                or self._label_search.regex != regex
            ):
                self._label_search = self.figure.find_label(text, regex=regex)

            # LabelSearch can be `None` if no match found
            if self._label_search:
                self._label_search.next()

    def find_previous(self):
        """Find previous occurrence."""
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if (
                not hasattr(self, "_label_search")
                or self._label_search.label != text
                or self._label_search.regex != regex
            ):
                self._label_search = self.figure.find_label(text, regex=regex)

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

    def set_leaf_labels(self):
        """Set the leaf labels."""
        label = self.label_combo_box.currentText()

        if label == "Default":
            label = self.figure._default_label_col

        # Nothing to do here
        if self._current_leaf_labels != label:
            self._last_leaf_labels, self._current_leaf_labels = (
                self._current_leaf_labels,
                label,
            )

        labels = self.figure._table[label].astype(str).fillna("").values

        # For labels that were set manually by the user (via pushing annotations)
        for i, label in self.label_overrides.items():
            # Label overrides {dend index: label}
            # We need to translate into original indices
            labels[self.figure._leafs_order[i]] = label

        # Add counts - e.g. "CB12345(10)"
        if self.label_count_check.isChecked():
            counts = pd.Series(labels).value_counts().to_dict()  # dict is much faster
            labels = [
                f"{label}({counts[label]})" if counts[label] > 1 else label
                for label in labels
            ]
        self.figure.labels = labels

        # Update searchbar completer
        if not hasattr(self, "_label_models"):
            self._label_models = {}
        if (label, self.label_count_check.isChecked()) not in self._label_models:
            self._label_models[(label, self.label_count_check.isChecked())] = (
                QtCore.QStringListModel(np.unique(labels).tolist())
            )

        self.searchbar_completer.setModel(
            self._label_models[(label, self.label_count_check.isChecked())]
        )

    def switch_labels(self):
        """Switch between current and last labels."""
        if hasattr(self, "_last_leaf_labels"):
            self.label_combo_box.setCurrentText(self._last_leaf_labels)
            self.set_leaf_labels()
            self.figure.show_message(
                f"Labels: {self._current_leaf_labels}", color="lightgreen", duration=2
            )

    def close(self):
        """Close the controls."""
        super().close()

    def ngl_open(self):
        if not hasattr(self.figure, "_ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure._ngl_viewer.neuroglancer_scene()
        scene.open()

    def ngl_copy(self):
        if not hasattr(self.figure, "_ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure._ngl_viewer.neuroglancer_scene()
        scene.to_clipboard()
        self.figure.show_message(
            "Link copied to clipboard", color="lightgreen", duration=2
        )


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


def _push_annotations(
    label,
    bodyids,
    rootids,
    clio,
    ftu,
    set_flywire=True,
    set_type=True,
    set_mcns_type=True,
    set_manc_type=True,
    figure=None,
    controls=None,
):
    """Push the current annotation to Clio/FlyTable."""
    try:
        if bodyids is not None and len(bodyids):
            kwargs = {}
            if set_flywire:
                kwargs["flywire_type"] = label
            if set_type:
                kwargs["type"] = label
            if set_manc_type:
                kwargs["manc_type"] = label

            clio.set_fields(bodyids, **kwargs)

        if set_mcns_type and rootids is not None and len(rootids):
            ftu.info.update_fields(
                rootids,
                malecns_type=label,
                malecns_type_source=os.environ.get("BC_ANNOTATION_USER", "bigclust"),
                id_col="root_783",
                dry_run=False,
            )

        set_any_malecns = set_flywire or set_manc_type or set_type
        if set_any_malecns and set_mcns_type:
            msg = f"Set {label} for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif set_flywire or set_type or set_manc_type:
            msg = f"Set {label} for {len(bodyids)} male CNS neurons"
        elif set_mcns_type:
            msg = f"Set {label} for {len(rootids)} FlyWire neurons"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids) and set_any_malecns:
            print("  ", bodyids)
        if rootids is not None and len(rootids) and set_mcns_type:
            print("  ", rootids)

        if figure:
            # Update the labels in the dendrogram
            if set_any_malecns and bodyids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, bodyids)]
                figure.set_leaf_label(ind, f"{label}(!)")
                controls.label_overrides.update({i: f"{label}(!)" for i in ind})
            if set_mcns_type and rootids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, rootids)]
                figure.set_leaf_label(ind, f"{label}(!)")
                controls.label_overrides.update({i: f"{label}(!)" for i in ind})

            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except BaseException as e:
        if figure:
            figure.show_message(
                "Error pushing annotations (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def _push_dimorphism(
    dimorphism,
    bodyids,
    rootids,
    clio,
    ftu,
    figure=None,
):
    """Push dimorphism status to Clio/FlyTable."""
    try:
        if bodyids is not None and len(bodyids):
            label = (
                dimorphism.replace("sex-specific", "male-specific")
                if dimorphism
                else None
            )

            clio.set_fields(bodyids, dimorphism=label)

        if rootids is not None and len(rootids):
            label = (
                dimorphism.replace("sex-specific", "female-specific")
                if dimorphism
                else None
            )

            ftu.info.update_fields(
                rootids, dimorphism=label, id_col="root_783", dry_run=False
            )

        if bodyids is not None and rootids is not None:
            msg = f"Set dimorphism to '{dimorphism}' for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif bodyids is not None:
            msg = (
                f"Set dimorphism to '{dimorphism}' for {len(bodyids)} male CNS neurons"
            )
        elif rootids is not None:
            msg = f"Set dimorphism to '{dimorphism}' for {len(rootids)} FlyWire neurons"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids):
            print("  ", bodyids)
        if rootids is not None and len(rootids):
            print("  ", rootids)

        if figure:
            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except BaseException as e:
        if figure:
            figure.show_message(
                "Error pushing dimorphism status (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def _push_super_type(
    super_type,
    bodyids,
    rootids,
    clio,
    ftu,
    sanity_checks=True,
    figure=None,
):
    """Push supertype to Clio/FlyTable."""
    try:
        # # Sanity checks.
        # if sanity_checks:
        #     # First get the required data
        #     mcns_data = None
        #     if bodyids is not None and len(bodyids):
        #         mcns_data = clio.fetch_annotations(bodyids)
        #     fw_data = None
        #     if rootids is not None and len(rootids):
        #         table = ftu.info.get_table()
        #         fw_data = table[table.root_783.isin(np.array(rootids).astype(str).tolist())]

        #     # 1. Do all neurons have the same hemilineage?
        #     hl = []
        #     if mcns_data is not None:
        #         hl += mcns_data.get("itolee_hl").values.tolist()
        #     if fw_data is not None:
        #         hl += fw_data.get("ito_lee_hemilineage").values.tolist()

        #     if len(set(hl)) > 1:
        #         raise ValueError("Not all neurons have the same hemilineage:", set(hl))

        #     # 2. Are all the types in

        # Make sure supertype is a string
        super_type = str(super_type)

        if bodyids is not None and len(bodyids):
            clio.set_fields(bodyids, supertype=super_type)

        if rootids is not None and len(rootids):
            ftu.info.update_fields(
                rootids, supertype=super_type, id_col="root_783", dry_run=False
            )

        if bodyids is not None and rootids is not None:
            msg = f"Set super type to '{super_type}' for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif bodyids is not None:
            msg = (
                f"Set super type to '{super_type}' for {len(bodyids)} male CNS neurons"
            )
        elif rootids is not None:
            msg = f"Set super type to '{super_type}' for {len(rootids)} FlyWire neurons"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids):
            print("  ", bodyids)
        if rootids is not None and len(rootids):
            print("  ", rootids)

        if figure:
            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except BaseException as e:
        if figure:
            figure.show_message(
                "Error pushing super type (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def _clear_annotations(
    bodyids,
    rootids,
    clio,
    ftu,
    clear_flywire=True,
    clear_type=True,
    clear_mcns_type=True,
    clear_manc_type=True,
    figure=None,
    controls=None,
):
    """Push the current annotation to Clio."""
    cleared_fields = []
    cleared_ids = []
    try:
        if bodyids is not None and len(bodyids):
            kwargs = {}
            if clear_type:
                kwargs["type"] = None
                cleared_fields += ["`type`"]
            if clear_flywire:
                kwargs["flywire_type"] = None
                cleared_fields.append("`flywire_type`")
            if clear_manc_type:
                kwargs["manc_type"] = None
                cleared_fields.append("`manc_type`")

            clio.set_fields(bodyids, **kwargs)
            cleared_ids.append(f"{len(bodyids)} maleCNS")

        if clear_mcns_type and rootids is not None and len(rootids):
            ftu.info.update_fields(
                rootids,
                malecns_type=None,
                malecns_type_source=None,
                id_col="root_783",
                dry_run=False,
            )
            cleared_fields.append("`malecns_type`")
            cleared_ids.append(f"{len(rootids)} FlyWire")

        msg = f"Cleared {', '.join(cleared_fields)} for {' and '.join(cleared_ids)} neuron(s)"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids) and (clear_flywire or clear_type):
            print("  ", bodyids)
        if rootids is not None and len(rootids) and clear_mcns_type:
            print("  ", rootids)

        if figure:
            # Update the labels in the dendrogram
            if (clear_flywire or clear_type or clear_manc_type) and bodyids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, bodyids)]
                figure.set_leaf_label(ind, "(cleared)(!)")
                controls.label_overrides.update({i: "(cleared)(!)" for i in ind})
            if clear_mcns_type and rootids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, rootids)]
                figure.set_leaf_label(ind, "(cleared)(!)")
                controls.label_overrides.update({i: "(cleared)(!)" for i in ind})

            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except:
        if figure:
            figure.show_message(
                "Error pushing annotations (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def _push_group(
    group,
    bodyids,
    clio,
    figure=None,
):
    """Push group to Clio."""
    try:
        if bodyids is not None:
            clio.set_fields(bodyids, group=group)

        msg = f"Set group {group} for {len(bodyids)} maleCNS neurons"

        print(f"{msg}:")
        print("  ", bodyids)

        if figure:
            # Update the labels in the dendrogram
            figure.set_leaf_label(
                figure.selected[np.isin(figure.selected_ids, bodyids)],
                f"group_{group}(!)",
            )
            # Show the message
            figure.show_message(msg, color="lightgreen", duration=2)
    except:
        if figure:
            figure.show_message(
                "Error pushing annotations (see console)", color="red", duration=2
            )
        traceback.print_exc()
        raise


def suggest_new_label(bodyids):
    """Suggest a new male-only label."""

    # First we need to find the main input neuropil for these neurons
    import neuprint as neu

    global NEUPRINT_CLIENT
    if NEUPRINT_CLIENT is None:
        NEUPRINT_CLIENT = neu.Client("https://neuprint-cns.janelia.org", dataset="cns")

    meta, roi = neu.fetch_neurons(
        neu.NeuronCriteria(bodyId=bodyids), client=NEUPRINT_CLIENT
    )

    # Drop non-primary ROIs
    roi = roi[roi.roi.isin(NEUPRINT_CLIENT.primary_rois)]

    # Remove the hemisphere information
    roi["roi"] = roi.roi.str.replace("(R)", "").str.replace("(L)", "")

    # Find the ROIs that collectively hold > 50% of the neurons input
    roi_in = roi.groupby("roi").post.sum().sort_values(ascending=False)
    roi_in = roi_in / roi_in.sum()

    global HB_ANN
    if HB_ANN is None:
        HB_ANN = pd.read_csv(
            "https://github.com/flyconnectome/flywire_annotations/raw/refs/heads/main/supplemental_files/Supplemental_file5_hemibrain_meta.csv"
        )

    import cocoa as cc

    global CLIO_ANN
    if CLIO_ANN is None:
        print("Fetching Clio annotations...")
        CLIO_ANN = cc.MaleCNS().get_annotations()

    print("Suggested cell type for IDs:", bodyids)
    for roi in roi_in.index.values[:4]:
        # Check if we already have male-specific types for this ROI
        this_mcns = CLIO_ANN[
            CLIO_ANN.type.str.match(f"{roi}[0-9]+m", na=False)
        ].type.unique()

        if len(this_mcns):
            new_id = max([int(t[len(roi) : len(roi) + 3]) for t in this_mcns]) + 1
        else:
            # Check if we already have hemibrain types for this ROI
            this_hb = HB_ANN[
                HB_ANN.type.str.match(f"{roi}[0-9]+", na=False)
            ].morphology_type.unique()

            if len(this_hb):
                highest_hb = max([int(t[len(roi) :]) for t in this_hb])

                # Start with the next hundred after the highest hemibrain type
                new_id = (highest_hb // 100 + 1) * 100
                if (new_id - highest_hb) < 10:
                    new_id += 100
            else:
                new_id = 1

        print(f"{roi}{new_id:03}m ({roi_in[roi]:.2%})")


def is_root_id(x):
    """Check if the ID is a root ID (as opposed to a body ID) based on its length."""
    if not isinstance(x, (np.ndarray, tuple, list)):
        x = [x]
    return np.array([len(str(i)) > 15 for i in x])


def sort_ids(ids, meta):
    """Sort given IDs into FlyWire root IDs and male CNS body IDs.

    Parameters
    ----------
    ids :       array-like
                IDs to sort.
    meta :      DataFrame
                Meta data for the neurons. Order should match the IDs.
                This is used to determine whether the IDs are FlyWire root IDs
                or Male CNS body IDs. This requires are `dataset` column
                which, by convention, uses e.g. `Fw` or `FlyWire` + a side suffix
                for FlyWire and `Mcns` or `MaleCNS` + a side suffix for the
                Male CNS.

    Returns
    -------
    rootids :   array-like
                FlyWire root IDs.
    bodyids :   array-like
                Male CNS body IDs.

    """
    ids = np.asarray(ids)

    assert "dataset" in meta.columns, "Meta data must have a 'dataset' column"

    # Process dataset column
    dataset_lower = meta.dataset.fillna("").str.lower()

    # Get FlyWire root IDs
    is_fw_root = dataset_lower.str.startswith("fw") | dataset_lower.str.startswith(
        "flywire"
    )
    rootids = ids[is_fw_root]

    # Get MaleCNS body IDs
    is_mcns = dataset_lower.str.startswith("mcns") | dataset_lower.str.startswith(
        "malecns"
    )
    bodyids = ids[is_mcns]

    return rootids, bodyids
