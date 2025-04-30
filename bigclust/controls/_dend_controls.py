import os

import numpy as np
import pandas as pd

from PySide6 import QtWidgets, QtCore

from ._base_controls import BaseControls

# TODOs:
# - add custom legend formatting (e.g. "{object.name}")
# - show type of object in legend
# - add dropdown to manipulate all selected objects
# - add filter to legend (use item.setHidden(True/False) to show/hide items)
# - highlight object in legend when hovered over in scene
# - make legend tabbed (QTabWidget)


class DendrogramControls(BaseControls):
    def __init__(self, figure, labels=[], datasets=[], width=200, height=400):
        super().__init__(
            figure, labels=labels, datasets=datasets, width=width, height=height
        )

        # Run some dendrogram-specific UI elements
        self.customize_gui()

    @property
    def selected_indices(self):
        """Get the selected IDs."""
        return self.figure._leafs_order[self.figure._selected]

    def customize_gui(self):
        """Customize GUI for the dendrogram."""
        # Add a SpinBox for label rotation
        hlayout = QtWidgets.QHBoxLayout()
        # N.B. we need to insert the layout before the last one, which is a spacer
        self.tab4_layout.insertLayout(self.tab4_layout.count() - 1, hlayout)
        label = QtWidgets.QLabel("Label rotation:")
        label.setToolTip("Set the rotation for the labels in the figure.")
        hlayout.addWidget(label)
        self.label_rotation_slider = QtWidgets.QSpinBox()
        self.label_rotation_slider.setRange(0, 360)
        self.label_rotation_slider.setValue(self.figure.label_rotation)
        self.label_rotation_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "label_rotation", int(x))
        )
        hlayout.addWidget(self.label_rotation_slider)

        # Checkbox for whether to show label lines
        self.label_lines_check = QtWidgets.QCheckBox("Show label lines")
        self.label_lines_check.setToolTip("Whether to plot lines grouping labels.")
        self.label_lines_check.setChecked(False)
        self.label_lines_check.stateChanged.connect(self.set_label_lines)

        # Get the position of the "Show label counts" checkbox and insert new checkbox below
        index = self.tab1_layout.indexOf(self.label_count_check)
        self.tab1_layout.insertWidget(
            index + 1, self.label_lines_check
        )

    def set_label_lines(self):
        """Set whether to show label lines."""
        self.figure.show_label_lines = self.label_lines_check.isChecked()

    def set_labels(self):
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

        labels = self.meta_data[label].astype(str).fillna("").values

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

        # Update label lines
        if hasattr(self.figure, "_label_line_group"):
            # Re-trigger making label lines
            self.figure.make_label_lines()
