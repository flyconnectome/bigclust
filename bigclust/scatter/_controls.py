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

from ..controls._base_controls import BaseControls, requires_selection

# TODOs:
# - add custom legend formatting (e.g. "{object.name}")
# - show type of object in legend
# - add dropdown to manipulate all selected objects
# - add filter to legend (use item.setHidden(True/False) to show/hide items)
# - highlight object in legend when hovered over in scene
# - make legend tabbed (QTabWidget)


# - self.font_size_slider: different settings [x]
# - self.label_rotation_label: remove [x]
# - self.point_size_label: add [x]
# - new methods: update_umap_setting, run_umap_maybe [x]
# - rename self.set_point_labels -> self.set_leaf_labels [x]
# - replace self.set_label_lines with self.set_label_outlines [x]
# - selected IDs:
#       - in dendrogam: self.figure._leafs_order[self.figure._selected]
#       - in scatter: self.figure._selected
# - labels:
#     - in dendrogram: self.figure._table[label].astype(str).fillna("").values
#     - in scatter: self.figure._data[label].astype(str).fillna("").values
#  -


class ScatterControls(BaseControls):
    def __init__(self, figure, labels=[], datasets=[], width=200, height=400):
        super().__init__(figure, labels=labels, datasets=datasets, width=width, height=height)

        # Some custom scatter-specific UI elements
        self.customize_gui()

        # Build UMAP tab
        self.tab5 = QtWidgets.QWidget()
        self.tab5_layout = QtWidgets.QVBoxLayout()
        self.tab5.setLayout(self.tab5_layout)
        self.tabs.addTab(self.tab5, "UMAP")
        self.build_umap_gui()
        if not hasattr(self.figure, "_dists") or self.figure._dists is None:
            self.tabs.setTabEnabled(4, False)

    @property
    def selected_indices(self):
        """Get the selected IDs."""
        return self.figure._selected

    def customize_gui(self):
        """Customize GUI for the dendrogram."""
        # Add SpinBox for point size
        hlayout = QtWidgets.QHBoxLayout()
        # N.B. we need to insert the layout before the last one, which is a spacer
        self.tab4_layout.insertLayout(self.tab4_layout.count() - 1, hlayout)
        self.point_size_label = QtWidgets.QLabel("Point size:")
        self.point_size_label.setToolTip(
            "Set the size for the scatter points in the plot."
        )
        hlayout.addWidget(self.point_size_label)
        self.point_size_slider = QtWidgets.QSpinBox()
        self.point_size_slider.setRange(1, 200)
        self.point_size_slider.setValue(self.figure.point_size)
        self.point_size_slider.valueChanged.connect(
            lambda x: setattr(self.figure, "point_size", int(x))
        )
        hlayout.addWidget(self.point_size_slider)

        # Adjust the font size slider
        self.font_size_slider.setRange(0.0001, 200)
        self.font_size_slider.setSingleStep(0.2)
        self.font_size_slider.setValue(self.figure.font_size)

        # Checkbox for whether to show label outlines
        self.label_outlines_check = QtWidgets.QCheckBox("Show label outlines")
        self.label_outlines_check.setToolTip(
            "Whether to add draw polygons around neurons with the same label."
        )
        self.label_outlines_check.setChecked(False)
        self.label_outlines_check.stateChanged.connect(self.set_label_outlines)

        # Get the position of the "Show label counts" checkbox and insert new checkbox below
        index = self.tab1_layout.indexOf(self.label_count_check)
        self.tab1_layout.insertWidget(
            index + 1, self.label_outlines_check
        )

    def build_umap_gui(self):
        """Build the GUI for the UMAP tab."""
        # Add a button to run the umap
        self.umap_button = QtWidgets.QPushButton("Run UMAP")
        self.umap_button.setToolTip(
            "Run dimensionality reduction on the current dataset. This will overwrite the current positions."
        )
        self.umap_button.clicked.connect(self.run_umap)
        self.tab5_layout.addWidget(self.umap_button)

        # Add a dropdown to choose the method
        self.umap_method_label = QtWidgets.QLabel("Method:")
        self.tab5_layout.addWidget(self.umap_method_label)
        self.umap_method_combo_box = QtWidgets.QComboBox()
        self.umap_method_combo_box.setToolTip(
            "Select the method to use for dimensionality reduction."
        )
        for item in ("UMAP", "MDS"):
            self.umap_method_combo_box.addItem(item)
        if hasattr(self.figure, "_vects") and self.figure._vects is not None:
            self.umap_method_combo_box.addItem("PaCMAP")
        self.umap_method_combo_box.currentIndexChanged.connect(
            self.update_umap_settings
        )
        self.tab5_layout.addWidget(self.umap_method_combo_box)

        if isinstance(self.figure._dists, dict):
            # Add a dropdown to choose the distances for UMAP clustering
            self.umap_dist_label = QtWidgets.QLabel("Distance:")
            self.tab5_layout.addWidget(self.umap_dist_label)
            self.umap_dist_combo_box = QtWidgets.QComboBox()
            self.umap_dist_combo_box.setToolTip(
                "Select the distance to use for UMAP clustering."
            )
            for key in sorted(self.figure._dists.keys()):
                self.umap_dist_combo_box.addItem(key)
            self.umap_dist_combo_box.currentIndexChanged.connect(self.run_umap_maybe)
            self.tab5_layout.addWidget(self.umap_dist_combo_box)

        # Add a checkbox to automatically run UMAP
        self.umap_auto_run = QtWidgets.QCheckBox("Auto run")
        self.umap_auto_run.setToolTip(
            "Whether to automatically run dimensionality reduction when changing settings."
        )
        self.umap_auto_run.setChecked(False)
        self.umap_auto_run.stateChanged.connect(
            lambda: setattr(self.figure, "_auto_umap", self.umap_auto_run.isChecked())
        )
        self.tab5_layout.addWidget(self.umap_auto_run)

        ## Settings for UMAP:

        # Create a wrapper layout and widget for UMAP settings
        self.umap_settings_widget = QtWidgets.QWidget()
        self.umap_settings_layout = QtWidgets.QVBoxLayout()
        self.umap_settings_widget.setLayout(self.umap_settings_layout)
        self.tab5_layout.addWidget(self.umap_settings_widget)

        # Spinbox for number of neighbors
        hlayout = QtWidgets.QHBoxLayout()
        self.umap_settings_layout.addLayout(hlayout)
        n_neighbors_label = QtWidgets.QLabel("Number of neighbors:")
        n_neighbors_label.setToolTip(
            "Set the number of neighbors for the UMAP. This is useful for large datasets."
        )
        hlayout.addWidget(n_neighbors_label)
        self.umap_n_neighbors_slider = QtWidgets.QSpinBox()
        self.umap_n_neighbors_slider.setRange(1, 200)
        self.umap_n_neighbors_slider.setSingleStep(1)
        self.umap_n_neighbors_slider.setValue(15)
        self.umap_n_neighbors_slider.valueChanged.connect(self.run_umap_maybe)
        hlayout.addWidget(self.umap_n_neighbors_slider)

        # Spinbox for minimum distance
        hlayout = QtWidgets.QHBoxLayout()
        self.umap_settings_layout.addLayout(hlayout)
        umap_min_dist_label = QtWidgets.QLabel("Minimum distance:")
        umap_min_dist_label.setToolTip(
            "Smaller values will result in a more clustered/clumped embedding where nearby points "
            "on the manifold are drawn closer together, while larger values will "
            "result on a more even dispersal of points. The value should be set "
            "relative to the ``spread`` value, which determines the scale at which "
            "embedded points will be spread out."
        )
        hlayout.addWidget(umap_min_dist_label)
        self.umap_min_dist_slider = QtWidgets.QDoubleSpinBox()
        self.umap_min_dist_slider.setRange(0.0, 10.0)
        self.umap_min_dist_slider.setSingleStep(0.05)
        self.umap_min_dist_slider.setValue(0.1)
        self.umap_min_dist_slider.valueChanged.connect(self.run_umap_maybe)
        hlayout.addWidget(self.umap_min_dist_slider)

        # Spinbox for spread
        hlayout = QtWidgets.QHBoxLayout()
        self.umap_settings_layout.addLayout(hlayout)
        spread_label = QtWidgets.QLabel("Spread:")
        spread_label.setToolTip(
            "The effective scale of embedded points. In combination with ``min_dist`` "
            "this determines how clustered/clumped the embedded points are."
        )
        hlayout.addWidget(spread_label)
        self.umap_spread_slider = QtWidgets.QDoubleSpinBox()
        self.umap_spread_slider.setRange(0.0, 10.0)
        self.umap_spread_slider.setSingleStep(0.05)
        self.umap_spread_slider.setValue(1)
        self.umap_spread_slider.valueChanged.connect(self.run_umap_maybe)
        hlayout.addWidget(self.umap_spread_slider)

        ## Settings for MDS

        # Create a wrapper layout and widget for MDS settings
        self.mds_settings_widget = QtWidgets.QWidget()
        self.mds_settings_layout = QtWidgets.QVBoxLayout()
        self.mds_settings_widget.setLayout(self.mds_settings_layout)
        self.tab5_layout.addWidget(self.mds_settings_widget)

        # Spinbox for number of initialisations
        hlayout = QtWidgets.QHBoxLayout()
        self.mds_settings_layout.addLayout(hlayout)
        n_init_label = QtWidgets.QLabel("Number of initializations:")
        n_init_label.setToolTip("Set the number of initializations for the MDS.")
        hlayout.addWidget(n_init_label)
        self.mds_n_init_slider = QtWidgets.QSpinBox()
        self.mds_n_init_slider.setRange(1, 200)
        self.mds_n_init_slider.setSingleStep(1)
        self.mds_n_init_slider.setValue(4)
        self.mds_n_init_slider.valueChanged.connect(self.run_umap_maybe)
        hlayout.addWidget(self.mds_n_init_slider)

        # Spinbox for max number of iterations
        hlayout = QtWidgets.QHBoxLayout()
        self.mds_settings_layout.addLayout(hlayout)
        max_iter_label = QtWidgets.QLabel("Max iterations:")
        max_iter_label.setToolTip(
            "Set the maximum number of iterations for the MDS. This is useful for large datasets."
        )
        hlayout.addWidget(max_iter_label)
        self.mds_max_iter_slider = QtWidgets.QSpinBox()
        self.mds_max_iter_slider.setRange(1, 10000)
        self.mds_max_iter_slider.setSingleStep(1)
        self.mds_max_iter_slider.setValue(300)
        self.mds_max_iter_slider.valueChanged.connect(self.run_umap_maybe)
        hlayout.addWidget(self.mds_max_iter_slider)

        # Spinbox for relative tolerance
        hlayout = QtWidgets.QHBoxLayout()
        self.mds_settings_layout.addLayout(hlayout)
        rel_tol_label = QtWidgets.QLabel("Relative tolerance:")
        rel_tol_label.setToolTip(
            "Relative tolerance with respect to stress at which to declare convergence."
        )
        hlayout.addWidget(rel_tol_label)
        self.mds_eps_slider = QtWidgets.QDoubleSpinBox()
        self.mds_eps_slider.setRange(0.0000, 1.0000)
        self.mds_eps_slider.setSingleStep(0.001)
        self.mds_eps_slider.setDecimals(4)
        self.mds_eps_slider.setValue(0.001)
        self.mds_eps_slider.valueChanged.connect(self.run_umap_maybe)
        hlayout.addWidget(self.mds_eps_slider)

        ## General settings

        # Random seed
        hlayout = QtWidgets.QHBoxLayout()
        self.tab5_layout.addLayout(hlayout)
        random_seed_label = QtWidgets.QLabel("Random seed:")
        hlayout.addWidget(random_seed_label)
        self.umap_random_seed = QtWidgets.QLineEdit()
        self.umap_random_seed.setToolTip(
            "Set the random seed. Leave empty for random initialization."
        )
        self.umap_random_seed.setPlaceholderText("random initialization")
        self.umap_random_seed.setText(str(42))
        self.umap_random_seed.textChanged.connect(lambda x: self.run_umap_maybe())
        hlayout.addWidget(self.umap_random_seed)

        # Stretch
        self.tab5_layout.addStretch(1)

        # Make sure the UMAP settings are hidden by default
        self.update_umap_settings()

    def set_label_outlines(self):
        """Draw polygons around neurons with the same label."""
        self.figure.show_label_lines = self.label_outlines_check.isChecked()

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

        labels = self.figure._data[label].astype(str).fillna("").values

        # For labels that were set manually by the user (via pushing annotations)
        for i, label in self.label_overrides.items():
            # Label overrides {dend index: label}
            # We need to translate into original indices
            labels[i] = label

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

    def update_umap_settings(self):
        """Update the UMAP settings based on the selected method."""
        if self.umap_method_combo_box.currentText() == "UMAP":
            self.umap_settings_widget.show()
            self.mds_settings_widget.hide()
            self.umap_button.setText("Run UMAP")
        elif self.umap_method_combo_box.currentText() == "MDS":
            self.umap_settings_widget.hide()
            self.mds_settings_widget.show()
            self.umap_button.setText("Run MDS")
        else:
            self.umap_settings_widget.hide()
            self.mds_settings_widget.hide()
            self.umap_button.setText("Run PCA")

        self.run_umap_maybe()

    def run_umap(self):
        """Run umap and move points to their new positions."""
        if self.umap_method_combo_box.currentText() == "UMAP":
            import umap

            fit = umap.UMAP(
                metric="precomputed",
                n_components=2,
                n_neighbors=self.umap_n_neighbors_slider.value(),
                min_dist=self.umap_min_dist_slider.value(),
                spread=self.umap_spread_slider.value(),
                random_state=int(self.umap_random_seed.text())
                if self.umap_random_seed.text()
                else None,
            )
        elif self.umap_method_combo_box.currentText() == "MDS":
            from sklearn.manifold import MDS

            fit = MDS(
                n_components=2,
                n_init=self.mds_n_init_slider.value(),
                max_iter=self.mds_max_iter_slider.value(),
                eps=self.mds_eps_slider.value(),
                dissimilarity="precomputed",
                random_state=int(self.umap_random_seed.text())
                if self.umap_random_seed.text()
                else None,
            )
        elif self.umap_method_combo_box.currentText() == "PCA":
            # We need KernelPCA because we are using a precomputed distance matrix
            from sklearn.decomposition import KernelPCA

            fit = KernelPCA(
                n_components=2,
                kernel="precomputed",
            )
        elif self.umap_method_combo_box.currentText() == "PaCMAP":
            import pacmap

            fit = pacmap.PaCMAP(
                n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0
            )

        if isinstance(self.figure._dists, dict):
            dists = self.figure._dists[self.umap_dist_combo_box.currentText()]
        else:
            dists = self.figure._dists

        if isinstance(dists, pd.DataFrame):
            dists = dists.values

        xy = fit.fit_transform(dists.astype(np.float64))

        # This moves points to their new positions
        self.figure.move_points(xy)

    def run_umap_maybe(self):
        """Run UMAP if the auto-run checkbox is checked."""
        if self.umap_auto_run.isChecked():
            self.run_umap()
