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
        self.searchbar.setToolTip("Search for a label in the scene. Use a leading '/' to search for a regex.")
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

        self.set_flywire_check = QtWidgets.QCheckBox("Clio: flywire_type")
        self.set_flywire_check.setToolTip("Set the `flywire_type` field in Clio")
        self.set_flywire_check.setChecked(True)
        self.tab2_layout.addWidget(self.set_flywire_check)

        self.set_type_check = QtWidgets.QCheckBox("Clio: type")
        self.set_type_check.setToolTip("Set the `type` field in Clio")
        self.tab2_layout.addWidget(self.set_type_check)

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

        # Add button to set new Clio group
        self.clio_group_button = QtWidgets.QPushButton("New Clio group")
        self.clio_group_button.setToolTip(
            "Assign new Clio group. This will use the lowest body ID as group ID."
        )
        self.clio_group_button.clicked.connect(self.new_clio_group)
        self.tab2_layout.addWidget(self.clio_group_button)

        # Add button to suggest new MCNS type
        self.suggest_type_button = QtWidgets.QPushButton("Suggest male-only type")
        self.suggest_type_button.setToolTip(
            "Suggest new male-only type based on main input neuropil(s). See console for output."
        )
        self.suggest_type_button.clicked.connect(self.suggest_type)
        self.tab2_layout.addWidget(self.suggest_type_button)

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

    def selected_set_dimorphism(self, dimorphism):
        """Push dimorphism to Clio/FlyTable."""
        assert dimorphism in (
            "sex-specific",
            "sexually dimorphic",
            "potentially sex-specific",
            "potentially sexually-dimorphic",
            None,
        )
        selected_ids = self.figure.selected_ids
        if selected_ids is None:
            self.figure.show_message("No selection", color="red", duration=2)
            return

        # Get the male CNS IDs
        from fafbseg import flywire

        is_root = flywire.is_valid_root(selected_ids)
        bodyids = selected_ids[~is_root]
        rootids = selected_ids[is_root]

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

    def push_annotation(self):
        """Push the current annotation to Clio/FlyTable."""
        if not any(
            (
                self.set_flywire_check.isChecked(),
                self.set_type_check.isChecked(),
                self.set_mcns_type_check.isChecked(),
            )
        ):
            self.figure.show_message("No fields to push", color="red", duration=2)
            return

        label = self.ann_combo_box.currentText()
        selected_ids = self.figure.selected_ids
        if selected_ids is None:
            self.figure.show_message("No selection", color="red", duration=2)
            return

        if not label:
            self.figure.show_message("No label to push", color="red", duration=2)
            return

        # Get the male CNS IDs
        from fafbseg import flywire

        is_root = flywire.is_valid_root(selected_ids)
        bodyids = selected_ids[~is_root]
        rootids = selected_ids[is_root]

        # Get the annotation
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # if self.set_sanity_check.isChecked():
        #     import cocoa as cc
        #     global CLIO_ANN
        #     if CLIO_ANN is None:
        #         print("Fetching Clio annotations...")
        #         CLIO_ANN = cc.MaleCNS().get_annotations()

        #     global FLYWIRE_ANN
        #     if FLYWIRE_ANN is None:
        #         FLYWIRE_ANN = cc.FlyWire(live_annot=True).get_annotations()

        #     for t in label.split(','):
        #         if t not in FLYWIRE_ANN.cell_type.unique() and t not in FLYWIRE_ANN.hemibrain_type.unique():
        #             self.figure.show_message(f'Label {t} not found in FlyWire annotations', color="red", duration=2)

        #     if label in CLIO_ANN.flywire_type.values:
        #         self.figure.show_message(f'Label {label} already in Clio', color="red", duration=2)
        #         return

        #     # Update the annotations
        #     if self.set_flywire_check.isChecked():
        #         CLIO_ANN.loc[CLIO_ANN.bodyId.isin(bodyids), 'flywire_type'] = label
        #     if self.set_type_check.isChecked():
        #         CLIO_ANN.loc[CLIO_ANN.bodyId.isin(bodyids), 'type'] = label

        # Submit the annotations
        self.futures[(label, uuid.uuid4())] = self.pool.submit(
            _push_annotations,
            label=label,
            bodyids=bodyids
            if self.set_flywire_check.isChecked() or self.set_type_check.isChecked()
            else None,
            rootids=rootids if self.set_mcns_type_check.isChecked() else None,
            set_flywire=self.set_flywire_check.isChecked(),
            set_type=self.set_type_check.isChecked(),
            set_mcns_type=self.set_mcns_type_check.isChecked(),
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
        )

        if self.set_type_check.isChecked() and len(bodyids) and CLIO_ANN is not None:
            # Update the CLIO annotations
            CLIO_ANN.loc[
                CLIO_ANN.get("bodyId", CLIO_ANN.get("bodyid")).isin(bodyids), "type"
            ] = label

        # if (
        #     self.set_flywire_check.isChecked() or self.set_type_check.isChecked()
        # ) and self.set_mcns_type_check.isChecked():
        #     msg = f"Set {label} for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        # elif self.set_flywire_check.isChecked() or self.set_type_check.isChecked():
        #     msg = f"Set {label} for {len(bodyids)} male CNS neurons"
        # elif self.set_mcns_type_check.isChecked():
        #     msg = f"Set {label} for {len(rootids)} FlyWire neurons"

        # self.figure.show_message(msg, color="lightgreen", duration=2)
        # print(f"{msg}:")
        # if len(bodyids) and (
        #     self.set_flywire_check.isChecked() or self.set_type_check.isChecked()
        # ):
        #     print("  ", bodyids)
        # if len(rootids) and self.set_mcns_type_check.isChecked():
        #     print("  ", rootids)

    def clear_annotation(self):
        """Clear the currently selected fields."""
        if not any(
            (
                self.set_flywire_check.isChecked(),
                self.set_type_check.isChecked(),
                self.set_mcns_type_check.isChecked(),
            )
        ):
            self.figure.show_message("No fields to clear", color="red", duration=2)
            return

        selected_ids = self.figure.selected_ids
        if selected_ids is None:
            self.figure.show_message("No neurons selected", color="red", duration=2)
            return

        # Get the male CNS IDs
        from fafbseg import flywire

        is_root = flywire.is_valid_root(selected_ids)
        bodyids = selected_ids[~is_root]
        rootids = selected_ids[is_root]

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
            if self.set_flywire_check.isChecked() or self.set_type_check.isChecked()
            else None,
            rootids=rootids if self.set_mcns_type_check.isChecked() else None,
            clear_flywire=self.set_flywire_check.isChecked(),
            clear_type=self.set_type_check.isChecked(),
            clear_mcns_type=self.set_mcns_type_check.isChecked(),
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
        )

    def new_clio_group(self):
        """Set a new Clio group for given IDs."""
        selected_ids = self.figure.selected_ids
        if selected_ids is None:
            self.figure.show_message("No selection", color="red", duration=2)
            return

        # Get the male CNS IDs
        from fafbseg import flywire

        is_root = flywire.is_valid_root(selected_ids)
        bodyids = selected_ids[~is_root]

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

    def suggest_type(self):
        """Suggest a new male-only type for given IDs."""
        selected_ids = self.figure.selected_ids
        if selected_ids is None:
            self.figure.show_message("No selection", color="red", duration=2)
            return

        # Get the male CNS IDs
        from fafbseg import flywire

        is_root = flywire.is_valid_root(selected_ids)
        bodyids = selected_ids[~is_root]

        if not len(bodyids):
            self.figure.show_message(
                "No MCNS neurons selected", color="red", duration=2
            )
            return

        # Threading this doesn't make much sense
        suggest_new_label(bodyids=bodyids)

    def set_add_group(self):
        """Set whether to add neurons as group when selected."""
        self.figure._add_as_group = self.add_group_check.isChecked()

    def set_dclick_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_dclick = self.dclick_deselect.isChecked()

    def find_next(self):
        """Find next occurrence."""
        text = self.searchbar.text()
        if text:
            regex = False
            if text.startswith("/"):
                regex = True
                text = text[1:]

            if not hasattr(self, "_label_search") or self._label_search.label != text or self._label_search.regex != regex:
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

            if not hasattr(self, "_label_search") or self._label_search.label != text or self._label_search.regex != regex:
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
    figure=None,
):
    """Push the current annotation to Clio/FlyTable."""
    try:
        if bodyids is not None and len(bodyids):
            if set_flywire and set_type:
                clio.set_fields(bodyids, flywire_type=label, type=label)
            elif set_flywire:
                clio.set_fields(bodyids, flywire_type=label)
            elif set_type:
                clio.set_fields(bodyids, type=label)

        if set_mcns_type and rootids is not None and len(rootids):
            ftu.info.update_fields(
                rootids,
                malecns_type=label,
                malecns_type_source=os.environ.get("BC_ANNOTATION_USER", "bigclust"),
                id_col="root_783",
                dry_run=False,
            )

        if (set_flywire or set_type) and set_mcns_type:
            msg = f"Set {label} for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif set_flywire or set_type:
            msg = f"Set {label} for {len(bodyids)} male CNS neurons"
        elif set_mcns_type:
            msg = f"Set {label} for {len(rootids)} FlyWire neurons"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids) and (set_flywire or set_type):
            print("  ", bodyids)
        if rootids is not None and len(rootids) and set_mcns_type:
            print("  ", rootids)

        if figure:
            # Update the labels in the dendrogram
            if (set_flywire or set_type) and bodyids is not None:
                figure.set_leaf_label(
                    figure.selected[np.isin(figure.selected_ids, bodyids)],
                    f"{label}(!)",
                )
            if set_mcns_type and rootids is not None:
                figure.set_leaf_label(
                    figure.selected[np.isin(figure.selected_ids, rootids)],
                    f"{label}(!)",
                )

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
            if dimorphism == "sex-specific":
                label = "male-specific"
            else:
                label = dimorphism

            clio.set_fields(bodyids, dimorphism=label)

        if rootids is not None and len(rootids):
            if dimorphism == "sex-specific":
                label = "female-specific"
            else:
                label = dimorphism

            ftu.info.update_fields(
                rootids,
                dimorphism=label,
                id_col="root_783",
                dry_run=False
            )

        if bodyids is not None and rootids is not None:
            msg = f"Set dimorphism to '{dimorphism}' for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif bodyids is not None:
            msg = f"Set dimorphism to '{dimorphism}' for {len(bodyids)} male CNS neurons"
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


def _clear_annotations(
    bodyids,
    rootids,
    clio,
    ftu,
    clear_flywire=True,
    clear_type=True,
    clear_mcns_type=True,
    figure=None,
):
    """Push the current annotation to Clio."""
    cleared_fields = []
    cleared_ids = []
    try:
        if bodyids is not None and len(bodyids):
            if clear_flywire and clear_type:
                clio.set_fields(bodyids, flywire_type=None, type=None)
                cleared_fields += ["`type`", "`flywire_type`"]
            elif clear_flywire:
                clio.set_fields(bodyids, flywire_type=None)
                cleared_fields.append("`flywire_type`")
            elif clear_type:
                clio.set_fields(bodyids, type=None)
                cleared_fields.append("`type`")
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
            if (clear_flywire or clear_type) and bodyids is not None:
                figure.set_leaf_label(
                    figure.selected[np.isin(figure.selected_ids, bodyids)],
                    "(cleared)(!)",
                )
            if clear_mcns_type and rootids is not None:
                figure.set_leaf_label(
                    figure.selected[np.isin(figure.selected_ids, rootids)],
                    "(cleared)(!)",
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
