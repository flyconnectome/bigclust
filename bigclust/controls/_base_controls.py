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


class BaseControls(QtWidgets.QWidget):
    """Base class for controls."""

    def __init__(
        self, figure, labels=[], datasets=[], width=200, height=400, build_gui=True
    ):
        super().__init__()
        self.figure = figure

        if hasattr(self.figure, "table"):
            self.meta_data = self.figure.table
        if hasattr(self.figure, "_table"):
            self.meta_data = self.figure._table
        elif hasattr(self.figure, "data"):
            self.meta_data = self.figure.data
        elif hasattr(self.figure, "_data"):
            self.meta_data = self.figure._data
        else:
            raise ValueError(
                "Figure does not have a table or data attribute. Cannot access meta data."
            )

        self.labels = labels
        self.datasets = datasets
        self.setWindowTitle("Controls")
        self.resize(width, height)
        self.label_overrides = {}

        # Build gui
        if build_gui:
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
            self.tab4 = QtWidgets.QWidget()
            self.tab1_layout = QtWidgets.QVBoxLayout()
            self.tab2_layout = QtWidgets.QVBoxLayout()
            self.tab3_layout = QtWidgets.QVBoxLayout()
            self.tab4_layout = QtWidgets.QVBoxLayout()
            self.tab1.setLayout(self.tab1_layout)
            self.tab2.setLayout(self.tab2_layout)
            self.tab3.setLayout(self.tab3_layout)
            self.tab4.setLayout(self.tab4_layout)

            self.tabs.addTab(self.tab1, "General")
            self.tabs.addTab(self.tab2, "Annotation")
            self.tabs.addTab(self.tab3, "Neuroglancer")
            self.tabs.addTab(self.tab4, "Settings")

            # Deactivate tabs
            if not os.environ.get("BC_ANNOTATION", "0") == "1":
                self.tabs.setTabEnabled(1, False)
            if not hasattr(self.figure, "_ngl_viewer"):
                self.tabs.setTabEnabled(2, False)

            self.build_control_gui()
            self.build_annotation_gui()
            self.build_neuroglancer_gui()
            self.build_settings_gui()

        # Holds the futures for requested data
        self.futures = {}
        self.pool = ThreadPoolExecutor(4)

    @property
    def selected_indices(self):
        """Get the selected IDs."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    def build_control_gui(self):
        """Build the GUI."""
        # Search bar
        self.tab1_layout.addWidget(QtWidgets.QLabel("Search"))
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
        self.selection_layout.addWidget(QtWidgets.QLabel("Selected:"))
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
        self.sel_clipboard_action = QtWidgets.QPushButton(text="IDs to Clipboard")
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
            self.sel_clipboard_action_menu.addAction(f"{ds}-only")
            self.sel_clipboard_action_menu.actions()[-1].triggered.connect(
                partial(self.selected_to_clipboard, dataset=datasets[ds])
            )

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add dropdown to choose leaf labels
        self.label_layout = QtWidgets.QHBoxLayout()
        self.tab1_layout.addLayout(self.label_layout)
        self.label_layout.addWidget(QtWidgets.QLabel("Labels:"))
        self.label_combo_box = QtWidgets.QComboBox()
        self.label_combo_box.addItem("Default")
        for col in sorted(self.meta_data.columns):
            self.label_combo_box.addItem(col)
        self.label_layout.addWidget(self.label_combo_box)
        self.label_combo_box.currentIndexChanged.connect(self.set_labels)
        self._current_leaf_labels = self.label_combo_box.currentText()

        # Checkbox for whether to show label counts
        self.label_count_check = QtWidgets.QCheckBox("Show label counts")
        self.label_count_check.setToolTip("Whether to add counts to the labels.")
        self.label_count_check.setChecked(False)
        self.label_count_check.stateChanged.connect(self.set_label_counts)
        self.tab1_layout.addWidget(self.label_count_check)

        # Add horizontal divider
        self.add_split(self.tab1_layout)

        # Add dropdown to choose color mode
        self.tab1_layout.addWidget(QtWidgets.QLabel("Color neurons by:"))
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

        # Add checkbox to determine whether to use colours
        self.ngl_use_colors = QtWidgets.QCheckBox("Use colors")
        self.ngl_use_colors.setToolTip(
            "Whether to use re-use colors from bigclust for the neuroglancer scene. If False, colours will be determined by neuroglancer."
        )
        self.ngl_use_colors.setChecked(True)
        self.tab3_layout.addWidget(self.ngl_use_colors)

        # Dropdown to choose whether to split neurons into layers other than source
        self.tab3_layout.addWidget(QtWidgets.QLabel("Group neurons into layers by:"))
        self.ngl_split_combo_box = QtWidgets.QComboBox()
        self.ngl_split_combo_box.addItem("Source")
        self.ngl_split_combo_box.addItem("Color")
        self.ngl_split_combo_box.addItem("Label")
        self.ngl_split_combo_box.setToolTip(
            "Determine how neurons are grouped into layers."
        )
        self.tab3_layout.addWidget(self.ngl_split_combo_box)

        # Checkbox for whether to cache neurons
        self.ngl_cache_neurons = QtWidgets.QCheckBox("Cache neurons")
        self.ngl_cache_neurons.setToolTip("Whether cache neuron meshes.")
        if hasattr(self.figure, "_ngl_viewer"):
            self.ngl_cache_neurons.setChecked(self.figure._ngl_viewer.use_cache)
        else:
            self.ngl_cache_neurons.setChecked(False)
        self.ngl_cache_neurons.stateChanged.connect(self.set_ngl_cache)
        self.tab3_layout.addWidget(self.ngl_cache_neurons)

        # Checkbox for debug mode
        self.ngl_debug_mode = QtWidgets.QCheckBox("Debug mode")
        self.ngl_debug_mode.setToolTip(
            "Whether to show debug information for the neuroglancer view"
        )
        self.ngl_debug_mode.setChecked(False)
        self.ngl_debug_mode.stateChanged.connect(self.set_ngl_debug)
        self.tab3_layout.addWidget(self.ngl_debug_mode)

        # This makes it so the legend does not stretch
        self.tab3_layout.addStretch(1)

    def build_settings_gui(self):
        # Add dropdown to determine render mode
        self.tab4_layout.addWidget(QtWidgets.QLabel("Render trigger:"))

        self.render_mode_dropdown = QtWidgets.QComboBox()
        self.render_mode_dropdown.setToolTip(
            "Set trigger for re-rendering the scene. See documentation for details."
        )
        self.render_mode_dropdown.addItems(["Continuous", "Reactive", "Active Window"])
        self.render_mode_dropdown.setItemData(
            0, "Continuously render the scene.", QtCore.Qt.ToolTipRole
        )
        self.render_mode_dropdown.setItemData(
            1,
            "Render only when the scene changes.",
            QtCore.Qt.ToolTipRole,
        )
        self.render_mode_dropdown.setItemData(
            2, "Render only when the window is active.", QtCore.Qt.ToolTipRole
        )
        render_trigger_vals = ["continuous", "reactive", "active_window"]
        self.render_mode_dropdown.currentIndexChanged.connect(
            lambda x: setattr(
                self.figure,
                "render_trigger",
                render_trigger_vals[self.render_mode_dropdown.currentIndex()],
            )
        )
        # Set default item to whatever the currently set render trigger is
        self.render_mode_dropdown.setCurrentIndex(
            render_trigger_vals.index(self.figure.render_trigger)
        )
        self.tab4_layout.addWidget(self.render_mode_dropdown)

        # Add slide for max frame rate
        label = QtWidgets.QLabel("Max frame rate:")
        label.setToolTip(
            "Set the maximum frame rate for the figure. Press F while the figure window is active to show current frame rate."
        )
        self.tab4_layout.addWidget(label)

        self.max_frame_rate_layout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(self.max_frame_rate_layout)

        self.max_frame_rate_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_frame_rate_slider.setRange(5, 100)
        self.max_frame_rate_slider.setValue(self.figure.max_fps)
        self.max_frame_rate_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "max_fps", int(x))
        )
        self.max_frame_rate_slider.valueChanged.connect(
            lambda x: self.max_frame_rate_value_label.setText(f"{x} FPS")
        )
        self.max_frame_rate_layout.addWidget(self.max_frame_rate_slider)

        self.max_frame_rate_value_label = QtWidgets.QLabel(
            f"{int(self.figure.max_fps)} FPS"
        )
        self.max_frame_rate_layout.addWidget(self.max_frame_rate_value_label)

        # Add SpinBox for font size
        hlayout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(hlayout)
        label = QtWidgets.QLabel("Font size:")
        label.setToolTip("Set the font size for the labels in the figure.")
        hlayout.addWidget(label)
        self.font_size_slider = QtWidgets.QSpinBox()
        self.font_size_slider.setRange(1, 200)
        self.font_size_slider.setValue(self.figure.font_size)
        self.font_size_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "font_size", x)
        )
        hlayout.addWidget(self.font_size_slider)

        # Add slider for number of labels visible at once
        label = QtWidgets.QLabel("Max visible labels:")
        label.setToolTip(
            "Set the maximum number of labels visible at once. This is useful for large datasets. May negatively impact performance."
        )
        self.tab4_layout.addWidget(label)

        self.max_label_vis_layout = QtWidgets.QHBoxLayout()
        self.tab4_layout.addLayout(self.max_label_vis_layout)

        self.max_label_vis_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_label_vis_slider.setRange(1, 5_000)
        self.max_label_vis_slider.setValue(self.figure.label_vis_limit)
        self.max_label_vis_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "label_vis_limit", int(x))
        )
        self.max_label_vis_slider.valueChanged.connect(
            lambda x: self.max_label_vis_value_label.setText(f"{x} labels")
        )
        self.max_label_vis_layout.addWidget(self.max_label_vis_slider)

        self.max_label_vis_value_label = QtWidgets.QLabel(
            f"{int(self.figure.label_vis_limit)} labels"
        )
        self.max_label_vis_layout.addWidget(self.max_label_vis_value_label)

        # This makes it so the legend does not stretch
        self.tab4_layout.addStretch(1)

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

        self.tab2_layout.addWidget(QtWidgets.QLabel("Which fields to set/clear:"))

        # Create a horizontal layout to hold the two vertical layouts:
        # one for Clio and one for FlyTable
        horizontal_layout = QtWidgets.QHBoxLayout()
        self.tab2_layout.addLayout(horizontal_layout)

        # Create the first vertical layout
        left_vertical_layout = QtWidgets.QVBoxLayout()
        horizontal_layout.addLayout(left_vertical_layout)

        left_vertical_layout.addWidget(QtWidgets.QLabel("Clio:"))
        self.set_clio_type = QtWidgets.QCheckBox("type")
        left_vertical_layout.addWidget(self.set_clio_type)
        self.set_clio_flywire_type = QtWidgets.QCheckBox("flywire_type")
        left_vertical_layout.addWidget(self.set_clio_flywire_type)
        self.set_clio_hemibrain_type = QtWidgets.QCheckBox("hemibrain_type")
        left_vertical_layout.addWidget(self.set_clio_hemibrain_type)
        self.set_clio_manc_type = QtWidgets.QCheckBox("manc_type")
        left_vertical_layout.addWidget(self.set_clio_manc_type)

        # Add a vertical line to separate the layouts
        vertical_line = QVLine()
        horizontal_layout.addWidget(vertical_line)

        # Create the second vertical layout
        right_vertical_layout = QtWidgets.QVBoxLayout()
        horizontal_layout.addLayout(right_vertical_layout)

        right_vertical_layout.addWidget(QtWidgets.QLabel("FlyTable:"))
        self.set_flytable_type = QtWidgets.QCheckBox("cell_type")
        right_vertical_layout.addWidget(self.set_flytable_type)
        self.set_flytable_mcns_type = QtWidgets.QCheckBox("malecns_type")
        right_vertical_layout.addWidget(self.set_flytable_mcns_type)
        self.set_flytable_hemibrain_type = QtWidgets.QCheckBox("hemibrain_type")
        right_vertical_layout.addWidget(self.set_flytable_hemibrain_type)

        self.tab2_layout.addWidget(QtWidgets.QLabel("Settings:"))

        self.set_sanity_check = QtWidgets.QCheckBox("Sanity checks")
        self.set_sanity_check.setToolTip("Whether to perform sanity checks")
        self.set_sanity_check.setChecked(True)
        self.tab2_layout.addWidget(self.set_sanity_check)

        # Add dropdown to set dimorphism status
        self.sel_dimorphism_action = QtWidgets.QPushButton(text="Set dimorphism")
        self.tab2_layout.addWidget(self.sel_dimorphism_action)
        self.sel_dimorphism_action_menu = QtWidgets.QMenu(self)
        self.sel_dimorphism_action.setMenu(self.sel_dimorphism_action_menu)

        # Set actions for the dropdown
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
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)  # No margins around the grid
        grid_layout.setSpacing(0)  # No space between buttons
        grid_layout.setColumnStretch(1, 1)  # Make the button column stretchable
        self.tab2_layout.addLayout(grid_layout)

        # Add checkbox + button to set new Clio group
        checkbox = QtWidgets.QCheckBox(" ")  # Do not remove the whitespace!
        checkbox.setToolTip("Check to activate the button to set a new Clio group.")
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(
            lambda x: self.clio_group_button.setEnabled(x == 2)
        )
        grid_layout.addWidget(checkbox, 0, 0)
        self.clio_group_button = QtWidgets.QPushButton("Set new Clio group")
        self.clio_group_button.setToolTip(
            "Assign new Clio group. This will use the lowest body ID as group ID."
        )
        self.clio_group_button.clicked.connect(self.new_clio_group)
        self.clio_group_button.setEnabled(False)
        grid_layout.addWidget(self.clio_group_button, 0, 1)

        # Add button to suggest new MCNS type
        self.suggest_type_button = QtWidgets.QPushButton("Suggest male-only type")
        self.suggest_type_button.setToolTip(
            "Suggest new male-only type based on main input neuropil(s). See console for output."
        )
        self.suggest_type_button.clicked.connect(self.suggest_type)
        grid_layout.addWidget(self.suggest_type_button, 1, 0, 1, -1)

        # Add button to suggest new CB type
        self.suggest_cb_type_button = QtWidgets.QPushButton("Suggest new CB-type")
        self.suggest_cb_type_button.setToolTip("Suggest new CBXXXX type.")
        self.suggest_cb_type_button.clicked.connect(self.suggest_cb_type)
        grid_layout.addWidget(self.suggest_cb_type_button, 2, 0, 1, -1)

        # Add checkbox + button to set new super type
        checkbox = QtWidgets.QCheckBox(" ")  # Do not remove the whitespace!
        checkbox.setToolTip("Check to activate the button to set a new supertype.")
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(
            lambda x: self.set_supertype_button.setEnabled(x == 2)
        )
        grid_layout.addWidget(checkbox, 3, 0)
        self.set_supertype_button = QtWidgets.QPushButton("Set new SuperType")
        self.set_supertype_button.setToolTip(
            "Assign selected neurons to a supertype. This will use the lowest ID as supertype ID."
        )
        self.set_supertype_button.clicked.connect(self.new_super_type)
        self.set_supertype_button.setEnabled(False)
        grid_layout.addWidget(self.set_supertype_button, 3, 1)

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
            # Skip if this label is NaN or None (i.e. not a string)
            if not isinstance(label, str):
                continue

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
                self.set_clio_type.isChecked(),
                self.set_clio_flywire_type.isChecked(),
                self.set_clio_hemibrain_type.isChecked(),
                self.set_clio_manc_type.isChecked(),
                self.set_flytable_type.isChecked(),
                self.set_flytable_mcns_type.isChecked(),
                self.set_flytable_hemibrain_type.isChecked(),
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

        # Which fields to set
        clio_to_set = []
        if self.set_clio_type.isChecked():
            clio_to_set.append("type")
        if self.set_clio_flywire_type.isChecked():
            clio_to_set.append("flywire_type")
        if self.set_clio_hemibrain_type.isChecked():
            clio_to_set.append("hemibrain_type")
        if self.set_clio_manc_type.isChecked():
            clio_to_set.append("manc_type")
        clio_to_set = tuple(clio_to_set)

        flytable_to_set = []
        if self.set_flytable_type.isChecked():
            flytable_to_set.append("cell_type")
        if self.set_flytable_mcns_type.isChecked():
            flytable_to_set.append("malecns_type")
        if self.set_flytable_hemibrain_type.isChecked():
            flytable_to_set.append("hemibrain_type")
        flytable_to_set = tuple(flytable_to_set)

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
            clio_to_set=clio_to_set,
            flytable_to_set=flytable_to_set,
            bodyids=bodyids if clio_to_set else None,
            rootids=rootids if flytable_to_set else None,
            clio=clio,  #  pass the module
            ftu=ftu,  #  pass the module
            figure=self.figure,
            controls=self,
        )

        if clio_to_set and len(bodyids) and CLIO_ANN is not None:
            # Update the CLIO annotations
            for col in clio_to_set:
                CLIO_ANN.loc[
                    CLIO_ANN.get("bodyId", CLIO_ANN.get("bodyid")).isin(bodyids), col
                ] = label

    @requires_selection
    def clear_annotation(self):
        """Clear the currently selected fields."""
        if not any(
            (
                self.set_clio_type.isChecked(),
                self.set_clio_flywire_type.isChecked(),
                self.set_clio_manc_type.isChecked(),
                self.set_clio_hemibrain_type.isChecked(),
                self.set_flytable_type.isChecked(),
                self.set_flytable_mcns_type.isChecked(),
                self.set_flytable_hemibrain_type.isChecked(),
            )
        ):
            self.figure.show_message("No fields to clear", color="red", duration=2)
            return

        # Extract FlyWire root and MaleCNS body IDs from the selected IDs
        # N.B. This requires meta data to be present.
        selected_ids = self.figure.selected_ids
        rootids, bodyids = sort_ids(selected_ids, self.figure.selected_meta)

        # Which fields to clear
        clio_to_clear = []
        if self.set_clio_type.isChecked():
            clio_to_clear.append("type")
        if self.set_clio_flywire_type.isChecked():
            clio_to_clear.append("flywire_type")
        if self.set_clio_hemibrain_type.isChecked():
            clio_to_clear.append("hemibrain_type")
        if self.set_clio_manc_type.isChecked():
            clio_to_clear.append("manc_type")
        clio_to_clear = tuple(clio_to_clear)

        flytable_to_clear = []
        if self.set_flytable_type.isChecked():
            flytable_to_clear.append("cell_type")
        if self.set_flytable_mcns_type.isChecked():
            flytable_to_clear.append("malecns_type")
        if self.set_flytable_hemibrain_type.isChecked():
            flytable_to_clear.append("hemibrain_type")
        flytable_to_clear = tuple(flytable_to_clear)

        # Get the annotation
        import clio

        global CLIO_CLIENT
        if CLIO_CLIENT is None:
            CLIO_CLIENT = clio.Client(dataset="CNS")

        import ftu

        # Submit the annotations
        self.futures[uuid.uuid4()] = self.pool.submit(
            _clear_annotations,
            bodyids=bodyids if clio_to_clear else None,
            rootids=rootids if flytable_to_clear else None,
            clio_to_clear=clio_to_clear,
            flytable_to_clear=flytable_to_clear,
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

    def set_ngl_cache(self):
        """Set whether the ngl viewer should cache neurons."""
        if hasattr(self.figure, "_ngl_viewer"):
            self.figure._ngl_viewer.use_cache = self.ngl_cache_neurons.isChecked()

            if not self.ngl_cache_neurons.isChecked():
                self.figure._ngl_viewer.clear_cache()

    def set_ngl_debug(self):
        """Set debug mode for ngl viewer."""
        if hasattr(self.figure, "_ngl_viewer"):
            self.figure._ngl_viewer.debug = self.ngl_debug_mode.isChecked()

    def set_dclick_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_dclick = self.dclick_deselect.isChecked()

    def set_empty_deselect(self):
        """Set whether to deselect on double-click."""
        self.figure.deselect_on_empty = self.empty_deselect.isChecked()

    def set_label_counts(self):
        """Set whether to add counts to the labels."""
        self.set_labels()  # Update the labels

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
            indices = self.selected_indices

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

    def set_labels(self):
        """Set the leaf labels."""
        raise NotImplementedError(
            "This method should be implemented in a subclass."
        )

    def switch_labels(self):
        """Switch between current and last labels."""
        if hasattr(self, "_last_leaf_labels"):
            self.label_combo_box.setCurrentText(self._last_leaf_labels)
            self.set_labels()
            self.figure.show_message(
                f"Labels: {self._current_leaf_labels}", color="lightgreen", duration=2
            )

    def close(self):
        """Close the controls."""
        super().close()

    def ngl_open(self):
        if not hasattr(self.figure, "_ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure._ngl_viewer.neuroglancer_scene(
            group_by=self.ngl_split_combo_box.currentText().lower(),
            use_colors=self.ngl_use_colors.isChecked(),
        )
        scene.open()

    def ngl_copy(self):
        if not hasattr(self.figure, "_ngl_viewer"):
            raise ValueError("Figure has no neuroglancer viewer")
        scene = self.figure._ngl_viewer.neuroglancer_scene(
            group_by=self.ngl_split_combo_box.currentText().lower(),
            use_colors=self.ngl_use_colors.isChecked(),
        )
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
    clio_to_set,
    flytable_to_set,
    figure=None,
    controls=None,
):
    """Push the current annotation to Clio/FlyTable."""
    try:
        if bodyids is not None and len(bodyids) and clio_to_set:
            kwargs = {}

            for field in clio_to_set:
                kwargs[field] = label

            clio.set_fields(bodyids, **kwargs)

        if rootids is not None and len(rootids) and flytable_to_set:
            kwargs = {}

            for field in flytable_to_set:
                kwargs[field] = label

                if field in ("cell_type", "hemibrain_type", "malecns_type"):
                    kwargs[f"{field}_source"] = os.environ.get(
                        "BC_ANNOTATION_USER", "bigclust"
                    )

            ftu.info.update_fields(
                rootids,
                **kwargs,
                id_col="root_783",
                dry_run=False,
            )

        if clio_to_set and flytable_to_set:
            msg = f"Set {label} for {len(bodyids)} maleCNS and {len(rootids)} FlyWire neurons"
        elif clio_to_set:
            msg = f"Set {label} for {len(bodyids)} male CNS neurons"
        elif flytable_to_set:
            msg = f"Set {label} for {len(rootids)} FlyWire neurons"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids) and clio_to_set:
            print("  ", bodyids)
        if rootids is not None and len(rootids) and flytable_to_set:
            print("  ", rootids)

        if figure:
            # Update the labels in the figure
            if clio_to_set and bodyids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, bodyids)]
                figure.set_labels(ind, f"{label}(!)")
                controls.label_overrides.update({i: f"{label}(!)" for i in ind})
            if flytable_to_set and rootids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, rootids)]
                figure.set_labels(ind, f"{label}(!)")
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
    clio_to_clear,
    flytable_to_clear,
    figure=None,
    controls=None,
):
    """Clear the given fields from to Clio/FlyTable."""
    cleared_fields = []
    cleared_ids = []
    try:
        if bodyids is not None and len(bodyids) and clio_to_clear:
            kwargs = {}

            for field in clio_to_clear:
                kwargs[field] = None
                cleared_fields.append(f"`{field}`")

            clio.set_fields(bodyids, **kwargs)
            cleared_ids.append(f"{len(bodyids)} maleCNS")

        if rootids is not None and len(rootids) and flytable_to_clear:
            kwargs = {}

            for field in flytable_to_clear:
                kwargs[field] = None
                cleared_fields.append(f"`{field}`")

                if field in ("cell_type", "hemibrain_type", "malecns_type"):
                    kwargs[f"{field}_source"] = None

            ftu.info.update_fields(
                rootids,
                **kwargs,
                id_col="root_783",
                dry_run=False,
            )
            cleared_fields.append("`malecns_type`")
            cleared_ids.append(f"{len(rootids)} FlyWire")

        msg = f"Cleared {', '.join(cleared_fields)} for {' and '.join(cleared_ids)} neuron(s)"

        print(f"{msg}:")
        if bodyids is not None and len(bodyids) and clio_to_clear:
            print("  ", bodyids)
        if rootids is not None and len(rootids) and flytable_to_clear:
            print("  ", rootids)

        if figure:
            # Update the labels in the figure
            if clio_to_clear and bodyids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, bodyids)]
                figure.set_labels(ind, "(cleared)(!)")
                controls.label_overrides.update({i: "(cleared)(!)" for i in ind})
            if flytable_to_clear and rootids is not None:
                ind = figure.selected[np.isin(figure.selected_ids, rootids)]
                figure.set_labels(ind, "(cleared)(!)")
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
            # Update the labels in the figure
            figure.set_labels(
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
