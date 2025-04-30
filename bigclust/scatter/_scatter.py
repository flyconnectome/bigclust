import re

import numpy as np
import pygfx as gfx
import pandas as pd
import pylinalg as la
import matplotlib.colors as mcl

from numbers import Number
from functools import partial

from ._controls import ScatterControls

from .._figure import Figure, update_figure
from .._selection import SelectionGizmo
from .._visuals import points2gfx, text2gfx


AVAILABLE_MARKERS = list(gfx.MarkerShape)
# Drop markers which look too similar to others
AVAILABLE_MARKERS.remove("ring")


# TODOs:
# - add PCA step to reduce the number of dimensions before UMAP
# - add additional coloring options for points
# - pop-up a new UMAP figure with just the selected points
# - cycle through UMAP components (1v2, 1v3, 2v3, etc.)
# - show third component as edges between points (thicker edges = closer points)
# - show outlines for different labels in different colors
# - make labels dropdown a multi-select


class ScatterPlot(Figure):
    """A Scatterplot for UMAP embeddings.

    Parameters
    ----------
    data :  DataFrame
            Data to be plotted. Must contain an `x` and a `y` column
            with coordinates for each point. Otherwise need to provide
            a distance matrix in `dists`.
    dists : DataFrame or dict, optional
            Data to be used for the UMAP embedding. If a DataFrame,
            it must contain a distance matrix. If a dict, must be
            `{dataset: distance matrix}`.
    labels : str, optional
            Column name in `data` containing labels for each point.
            Default is 'label'.
    ids :   str, optional
            Column name in `data` containing IDs for each point.
            Default is 'id'.
    colors : str, optional
            Column name in `data` used to generate colors for each point.
    hover_info : str, optional
            Column name in `data` used to generate hover information for each point.

    """

    _selection_color = "y"

    def __init__(
        self,
        data,
        dists=None,
        labels="label",
        ids="id",
        colors="color",
        markers="dataset",
        hover_info="hover_info",
        point_size=10,
        **kwargs,
    ):
        super().__init__(size=(1000, 400), **kwargs)

        if "x" not in data.columns:
            if dists is None:
                raise ValueError(
                    "No coordinates provided and no distance matrix given."
                )

            print("Generating UMAP coordinates...", flush=True, end="")
            import umap

            fit = umap.UMAP(
                n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, n_jobs=1
            )

            if isinstance(dists, dict):
                coords = fit.fit_transform(list(dists.values())[0])
            else:
                coords = fit.fit_transform(dists)
            data["x"] = coords[:, 0]
            data["y"] = coords[:, 1]

        self._data = data
        self._positions = np.array(data[["x", "y"]].values).astype(np.float32)
        self._default_label_col = labels
        self._labels = (
            np.array(data[labels].values) if labels is not None else None
        )  # make sure to use  a copy
        if self._labels is not None:
            # `_label_visuals` is in the same order as `_labels`
            self._label_visuals = [None] * len(self._labels)
        self._ids = np.asarray(data[ids]) if ids is not None else None
        self._colors = np.asarray(data[colors]) if colors is not None else None
        self._markers = np.asarray(data[markers]) if markers is not None else None
        self._dists = dists

        # Datasets are used to avoid collisions when the same ID is used in different datasets
        self._datasets = (
            self._data["dataset"].values if "dataset" in self._data.columns else None
        )

        self._selected = None
        self.deselect_on_empty = False

        if hover_info is not None:
            if "{" in hover_info:
                hover_info = data.apply(hover_info.format_map, axis=1)
            else:
                hover_info = data[hover_info]
        self._hover_info = np.asarray(hover_info) if hover_info is not None else None

        self._point_size = point_size
        self._font_size = 0.01
        self.label_vis_limit = 200  # number of labels shown at once before hiding all
        self.label_refresh_rate = 30  # update labels every n frames

        # Add the selection gizmo
        self.selection_gizmo = SelectionGizmo(
            self.renderer,
            self.camera,
            self.scene,
            callback_after=lambda x: self.select_points(
                x.bounds, additive="Control" in x._event_modifiers
            ),
        )

        # self.renderer.add_event_handler(self._mouse_press, "pointer_down")
        self.deselect_on_dclick = False

        # This group will hold text labels that need to move but not scale with the dendrogram
        self._text_group = gfx.Group()
        self.scene.add(self._text_group)

        # Generate the visuals
        self.make_visuals()

        # Generate the labels
        self._label_group = gfx.Group()
        self._label_group.visible = True
        self._text_group.add(self._label_group)

        # Setup hover info
        if hover_info is not None:

            def hover(event):
                # Note: we could use e.g. shift-hover to show
                # more/different info?
                if event.type == "pointer_enter":
                    # Translate position to world coordinates
                    pos = self._screen_to_world((event.x, event.y))

                    # Find the closest leaf
                    vis = event.current_target
                    coords = vis.geometry.positions.data
                    dist = np.linalg.norm(coords[:, :2] - pos[:2], axis=1)
                    closest = np.argmin(dist)
                    point_ix = vis._point_ix[closest]

                    # Position and show the hover widget
                    # self._hover_widget.local.position = coords[closest]
                    self._hover_widget.visible = True

                    # Set the text
                    # N.B. there is some funny behaviour where repeatedly setting the same
                    # text will cause the bounding box to increase every time. To avoid this
                    # we have to reset the text to anything but an empty string.
                    self._hover_widget.children[1].set_text(
                        "asdfgasdfasdfsdafsfasdfasg"
                    )
                    self._hover_widget.children[1].set_text(str(hover_info[point_ix]))

                    # Scale the background to fit the text
                    # bb = self._hover_widget.children[1].get_world_bounding_box()
                    # extent = bb[1] - bb[0]

                    # The text bounding box is currently not very accurate. For example,
                    # a single-line text has no height. Hence, we need to add some padding:
                    # extent = (extent + [0, 1.2, 0]) * 1.2
                    # self._hover_widget.children[0].local.scale_x = extent[0]
                    # self._hover_widget.children[0].local.scale_y = extent[1]

                elif self._hover_widget.visible:
                    self._hover_widget.visible = False

            for vis in self._point_visuals:
                vis.add_event_handler(hover, "pointer_enter", "pointer_leave")

            self._hover_widget = self.make_hover_widget()
            self.overlay_scene.add(self._hover_widget)

        # Show the points
        self.camera.show_object(self._scatter_group)

        # Add some keyboard shortcuts for moving and scaling the dendrogam
        def move_camera(x, y):
            self.camera.world.x += x
            self.camera.world.y += y
            self._render_stale = True
            self.canvas.request_draw()

        self.key_events["ArrowLeft"] = lambda: setattr(
            self, "font_size", max(self.font_size - 1, 1)
        )
        self.key_events["ArrowRight"] = lambda: setattr(
            self, "font_size", self.font_size + 1
        )
        self.key_events["ArrowUp"] = lambda: setattr(
            self, "point_size", max(self.point_size - 1, 1)
        )
        self.key_events["ArrowDown"] = lambda: setattr(
            self, "point_size", self.point_size + 1
        )
        self.key_events["Escape"] = lambda: self.deselect_all()
        self.key_events["l"] = lambda: self.toggle_labels()

        def _toggle_last_label():
            """Toggle between the last label and the original labels."""
            # If no controls, there is nothing to toggle
            if not hasattr(self, "_controls"):
                return
            self._controls.switch_labels()

        self.key_events["m"] = _toggle_last_label

        def _control_label_vis():
            """Show only labels currently visible."""
            if self._control_label_vis_every_n % self.label_refresh_rate:
                self._control_label_vis_every_n += 1
                return

            self._control_label_vis_every_n = 1

            if self._labels is not None and self._label_group.visible:
                # Check which leafs are currently visible
                iv = self.is_visible_pos(self._positions)

                # If more than the limit, don't show any labels
                if iv.sum() > self.label_vis_limit:
                    for i, t in enumerate(self._label_group.children):
                        t.visible = False
                else:
                    self.show_labels(np.where(iv)[0])
                    self.hide_labels(np.where(~iv)[0])

        # Turns out this is too slow to be run every frame - we're throttling it to every 30 frames
        if self._labels is not None:
            self._control_label_vis_every_n = 1
            self.add_animation(_control_label_vis)

        self.add_animation(self._process_moves)

    def __len__(self):
        return len(self._data)

    @property
    def labels(self):
        """Return the labels of leafs in the dendrogram."""
        return self._labels

    @labels.setter
    @update_figure
    def labels(self, x):
        """Set the labels of leafs in the dendrogram."""
        if x is None:
            self._labels = None
            self._label_visuals = None
            return
        assert len(x) == len(self), "Number of labels must match number of leafs."
        self._labels = np.asarray(x)
        self.update_point_labels()  # updates the visuals

    @property
    def font_size(self):
        return int(self._font_size * 100)

    @font_size.setter
    @update_figure
    def font_size(self, size):
        size = size / 100
        self._font_size = size
        for t in self._label_visuals:
            if isinstance(t, gfx.Text):
                t.font_size = size

    @property
    def point_size(self):
        return self._point_size

    @point_size.setter
    @update_figure
    def point_size(self, size):
        self._point_size = size
        for vis in self._point_visuals:
            if isinstance(vis, gfx.Points):
                vis.material.size = size

    @property
    def selected(self):
        """Return the indices of selected points in the dendrogram."""
        return self._selected

    @selected.setter
    @update_figure
    def selected(self, x):
        """Select given points in the dendrogram."""
        if isinstance(x, type(None)):
            x = []
        elif isinstance(x, int):
            x = [x]

        if isinstance(x, np.ndarray) and x.dtype == bool:
            assert len(x) == len(self), (
                "Selection mask must be the same length as the dendrogram."
            )
            x = np.where(x)[0]

        # Set the selected leafs (make sure to sort them)
        self._selected = np.asarray(sorted(x), dtype=int)

        # Create the new selection visuals
        self.highlight_points(self._selected, color=self._selection_color)

        # Update the controls
        # if hasattr(self, "_controls"):
        #     self._controls.update_ann_combo_box()

        if hasattr(self, "_ngl_viewer"):
            # `self._selected` is in order of the dendrogram, we need to translate it to the original order
            if len(self._selected) > 0:
                self._ngl_viewer.show(
                    self._ids[self.selected],
                    datasets=self._datasets[self.selected],
                    add_as_group=getattr(self, "_add_as_group", False),
                )
            else:
                self._ngl_viewer.clear()

    @property
    def selected_ids(self):
        """Return the IDs of selected leafs in the dendrogram."""
        if self.selected is None or not len(self.selected):
            return None
        if self._ids is None:
            raise ValueError("No IDs were provided.")
        return self._ids[self.selected]

    @property
    def selected_labels(self):
        """Return the labels of selected leafs in the dendrogram."""
        if self.selected is None or not len(self.selected):
            return None
        if self._labels is None:
            raise ValueError("No labels were provided.")
        return self._labels[self.selected]

    @property
    def selected_meta(self):
        """Return the metadata of selected leafs in the dendrogram."""
        if self.selected is None or not len(self.selected):
            return None
        if self._data is None:
            raise ValueError("No metadata was provided.")
        return self._data.iloc[self.selected]

    @property
    def show_label_lines(self):
        """Show or hide the label outlines."""
        if not hasattr(self, "_show_label_lines"):
            return False
        return self._show_label_lines

    @show_label_lines.setter
    @update_figure
    def show_label_lines(self, x):
        assert isinstance(x, bool), "`show_label_lines` must be a boolean."

        if x == self.show_label_lines:
            return

        if x:
            if not getattr(self, "_label_line_group", None):
                self.make_label_lines()
            self._label_line_group.visible = True
        elif not x and getattr(self, "_label_line_group", None):
            self._label_line_group.visible = False

        self._show_label_lines = x

    @classmethod
    def from_observation_vector(
        cls,
        vect,
        data,
        reducer="umap",
        metric="cosine",
        fit_kwargs=None,
    ):
        """Create a Scatterplot from an observation vector.

        Parameters
        ----------
        vect :  DataFrame
                The observation vector to use for the scatterplot.
        data :  DataFrame
                The data to use for the scatterplot. Must be in the same
                order as the observation vector.
        reducer : str, optional
                The dimensionality reduction method to use. Can be either
                'umap' or 'pca'. Default is 'umap'.
        metric : str, optional
                The metric to use for the dimensionality reduction. Default is 'cosine'.
        fit_kwargs : dict, optional
                Additional keyword arguments to pass to the dimensionality
                reduction method.

        """
        import umap

        fit = umap.UMAP(metric=metric, local_connectivity=4, n_neighbors=4)
        xy = fit.fit_transform(vect)

        data["x"] = xy[:, 0]
        data["y"] = xy[:, 1]

        # Create the scatterplot
        scatter = cls(
            data,
            labels="label",
            ids="id",
            colors="color",
            markers="dataset",
            hover_info="hover_info",
            point_size=10,
        )
        scatter._positions = xy
        scatter._default_label_col = "label"

    def deselect_all(self):
        self.selected = None

    def highlight_points(self, points, color="y"):
        """Highlight given points in the plot.

        Parameters
        ----------
        points :    array of int or bool | None
                    Either indices of points to highlight or a boolean mask.
                    Use `None` to clear all highlights.
        color :     str
                    Color to use for highlighting.

        """
        # Clear existing selection
        if hasattr(self, "_highlight_visuals"):
            for vis in self._highlight_visuals:
                if vis.parent:
                    vis.parent.remove(vis)
            del self._highlight_visuals

        # If no points are given, return
        if points is None:
            return

        # If a boolean mask is given, convert it to indices
        if isinstance(points, np.ndarray) and points.dtype == bool:
            assert len(points) == len(self), (
                "Selection mask must be the same length as the dendrogram."
            )
            points = np.where(points)[0]
        elif isinstance(points, int):
            points = [points]
        elif isinstance(points, list):
            points = np.array(points)
        elif not isinstance(points, np.ndarray):
            raise ValueError(f"Expected array or list, got {type(points)}.")

        if len(points) == 0:
            return

        # Create the new selection visuals
        if len(self._selected) > 0:
            self._highlight_visuals = self.make_points(
                mask=np.isin(np.arange(len(self)), points)
            )
            for vis in self._highlight_visuals:
                vis.material.edge_color = "yellow"
                vis.material.edge_width = 2
                vis.material.color = (1, 1, 1, 0)
                self._scatter_group.add(vis)

    @update_figure
    def toggle_labels(self):
        """Toggle the visibility of labels."""
        self._label_group.visible = not self._label_group.visible

    def make_visuals(self, labels=True, clear=False):
        """Generate the pygfx visuals for the scatterplot."""
        if clear:
            self.clear()

        # Create the group for the points
        self._scatter_group = gfx.Group()
        self._scatter_group._object_id = "scatter"
        self.scene.add(self._scatter_group)

        self._point_visuals = self.make_points()
        for p in self._point_visuals:
            self._scatter_group.add(p)

    def make_points(self, mask=None):
        """Create the visuals for the points."""
        visuals = []

        if self._markers is None:
            markers = np.full(len(self), "circle")
        else:
            assert len(self._markers) == len(self), (
                "Length of leaf_types must match length of dendrogram."
            )
            unique_types = np.unique(self._markers)

            assert len(unique_types) <= len(AVAILABLE_MARKERS), (
                "Only 10 unique types are supported."
            )
            marker_map = dict(zip(unique_types, AVAILABLE_MARKERS))
            markers = np.array([marker_map[t] for t in self._markers])

        # Create the visuals
        for m in np.unique(markers):
            color = "w"
            if mask is None:
                this = self._data.iloc[markers == m]
                ix = np.where(markers == m)[0]
                if self._colors is not None:
                    color = np.array(
                        [mcl.to_rgba(c) for c in self._colors[markers == m]]
                    )
            else:
                this = self._data.iloc[mask & (markers == m)]
                ix = np.where(mask & (markers == m))[0]
                if self._colors is not None:
                    color = np.array(
                        [mcl.to_rgba(c) for c in self._colors[mask & (markers == m)]]
                    )
            if len(this) == 0:
                continue

            vis = points2gfx(
                np.append(
                    this[["x", "y"]].values,
                    np.zeros(len(this)).reshape(-1, 1),
                    axis=1,
                ),
                color=color,
                size=self.point_size,
                marker=m,
                pick_write=self._hover_info is not None,
            )
            vis._point_ix = ix
            visuals.append(vis)

        return visuals

    def make_hover_widget(self, color=(1, 1, 1, 0.5), font_color=(0, 0, 0, 1)):
        """Generate a widget for hover info."""
        # The widget will be added to the overlay scene which uses a NDC camera
        # which means the coordinates will be in the range [-1, 1] regardless of
        # the actual size of the scene/window.

        widget = gfx.Group()
        widget.visible = False
        widget.local.position = (-0.75, 0, 0)

        widget.add(
            gfx.Mesh(
                gfx.plane_geometry(2 / 4, 2),  # full screen height, 1/4 width
                gfx.MeshBasicMaterial(color=color),
            )
        )
        widget.add(
            text2gfx(
                "Hover info",
                color=font_color,
                font_size=12,  # fix font size
                anchor="middle-center",
                screen_space=True,  # without this the text would be scewed
            )
        )

        return widget

    def make_label_lines(self):
        """Generate the polygones around each unique label."""
        from scipy.spatial import ConvexHull, Delaunay

        # Create a group and add to scene
        if not getattr(self, "_label_line_group", None):
            self._label_line_group = gfx.Group()
            self._label_line_group.visible = self.show_label_lines
            self.scene.add(self._label_line_group)

        # Clear the group (we might call this function to update the lines)
        self._label_line_group.clear()

        # Generate a dictionary mapping a unique label to the indices
        labels = {
            l: np.where(self._labels == l)[0]
            for l in np.unique(self._labels[~pd.isnull(self._labels)])
        }

        # Generate a line for each label
        vertices = []
        faces = []
        n_vertices = 0
        for l, indices in labels.items():
            # Get the points for this label
            points = self._positions[indices]

            # Generate a convex hull around the points
            if len(points) < 3:
                continue
            hull = ConvexHull(points)

            tri = Delaunay(points[hull.vertices])

            # Add vertices and faces to the list
            vertices.append(points[hull.vertices])
            faces.append(tri.simplices + n_vertices)
            n_vertices += len(hull.vertices)

        # Concatenate all vertices and faces
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)

        # Add a third coordinate to the vertices
        vertices = np.append(
            vertices, np.zeros((vertices.shape[0], 1), dtype=np.float32), axis=1
        )
        # Make sure vertices are in the back
        vertices[:, 2] = -1

        vis = gfx.Mesh(
            gfx.Geometry(indices=faces.astype(np.int32), positions=vertices.astype(np.float32)),
            gfx.MeshBasicMaterial(color=(1, 1, 1, 0.1)),
        )

        # Create a mesh for the label lines and add to the group
        self._label_line_group.add(vis)

    def show_controls(self):
        """Show controls."""
        if not hasattr(self, "_controls"):
            self._controls = ScatterControls(
                self,
                labels=list(set(self._labels)),
                datasets=list(set(self._datasets)),
            )
        self._controls.show()

    def _toggle_controls(self):
        """Switch controls on and off."""
        if not hasattr(self, "_controls"):
            self.show_controls()
        elif self._controls.isVisible():
            self.hide_controls()
        else:
            self.show_controls()

    @update_figure
    def show_labels(self, which=None):
        """Show labels for the leafs.

        Parameters
        ----------
        which : list, optional
                List of indices  (left to right) for which to show
                the labels. If None, all labels are shown.

        """
        if self._labels is None:
            return

        if which is None:
            which = np.arange(len(self))
        elif isinstance(which, Number):
            which = np.array([which])
        elif isinstance(which, list):
            which = np.array(which)

        if not isinstance(which, (list, np.ndarray)):
            raise ValueError(f"Expected list or array, got {type(which)}.")

        for ix in which:
            if ix < 0:
                ix = len(self) + ix

            if self._label_visuals[ix] is None:
                t = text2gfx(
                    str(self._labels[ix]),
                    position=(
                        self._positions[ix, 0] + 0.005,
                        self._positions[ix, 1],
                        0,
                    ),
                    font_size=self._font_size,
                    anchor="middle-left",
                    pickable=True,
                )

                def _highlight(event, text):
                    self.find_label(text._text, go_to_first=False)

                t.add_event_handler(partial(_highlight, text=t), "double_click")

                # `_label_visuals` is in the same order as `_labels`
                self._label_visuals[ix] = t
                self._label_group.add(t)

                # Track where this label is supposed to show up (for scaling)
                t._absolute_position = self._positions[ix]

                # Center the text
                t.text_align = "center"

            self._label_visuals[ix].visible = True

    @update_figure
    def hide_labels(self, which=None):
        """Hide labels for the leafs.

        Parameters
        ----------
        which : list, optional
                List of indices into the dendrogram (left to right) for which to hide
                the labels. If None, all labels are hidden.

        """
        if self._labels is None:
            return

        if which is None:
            which = np.arange(len(self))
        elif isinstance(which, int):
            which = np.array([which])
        elif isinstance(which, list):
            which = np.array(which)

        if not isinstance(which, (list, np.ndarray)):
            raise ValueError(f"Expected list or array, got {type(which)}.")

        for ix in which:
            if self._label_visuals[ix] is None:
                continue

            self._label_visuals[ix].visible = False

    @update_figure
    def find_label(
        self, label, regex=False, highlight=True, go_to_first=True, verbose=True
    ):
        """Find and center the plot on a given label.

        Parameters
        ----------
        label : str
                The label to search for.
        highlight : bool, optional
                Whether to highlight the found label.
        go_to_first : bool, optional
                Whether to go to the first occurrence of the label.
        verbose : bool, optional
                Whether to show a message when the label is found.

        Returns
        -------
        LabelSearch
            An object that can be used to iterate over the found labels.

        """
        ls = LabelSearch(self, label, go_to_first=go_to_first, regex=regex)

        if highlight:
            self.highlight_labels(ls.indices)

        if verbose:
            self.show_message(f"Found {len(ls)} occurrences of '{label}'", duration=3)

        return ls

    @update_figure
    def highlight_labels(self, x, color="y"):
        """Highlight labels in the plot.

        Parameters
        ----------
        x :      str | iterable | None
                 Can be either:
                  - a string with a label to highlight.
                  - an iterable with indices  of points to highlight
                  - `None` to clear the highlights.
        color : str, optional
                The color to use for highlighting.

        """
        if self._labels is None:
            return

        # Reset existing highlights
        for vis in self._label_visuals:
            if vis is None:
                continue
            if hasattr(vis.material, "_original_color"):
                vis.material.color = vis.material._original_color

        # Return here if we're only clearing the highlights
        if x is None:
            return

        if isinstance(x, str):
            for i, label in enumerate(self._labels):
                if label != x:
                    continue

                if self._label_visuals[i] is None:
                    self.show_labels(i)
                visual = self._label_visuals[i]

                visual.material._original_color = visual.material.color
                visual.material.color = color
        elif isinstance(x, (list, np.ndarray)):
            for ix in x:
                # Index in the original order
                if self._label_visuals[ix] is None:
                    self.show_labels(ix)
                visual = self._label_visuals[ix]

                visual.material._original_color = visual.material.color
                visual.material.color = color

    def close(self):
        """Close the figure."""
        if hasattr(self, "_controls"):
            self._controls.close()
        super().close()

    @update_figure
    def select_points(self, bounds, additive=False):
        """Select all selectable objects in the region."""
        # Get the positions and original indices of the leaf nodes
        positions_abs = []
        indices = []
        for l in self._point_visuals:
            positions_abs.append(
                la.vec_transform(l.geometry.positions.data, l.world.matrix)
            )
            indices.append(l._point_ix)
        positions_abs = np.vstack(positions_abs)
        indices = np.concatenate(indices)

        # Check if any of the points are within the selection region
        selected = (
            (positions_abs[:, 0] >= bounds[0, 0])
            & (positions_abs[:, 0] <= bounds[1, 0])
            & (positions_abs[:, 1] >= bounds[0, 1])
            & (positions_abs[:, 1] <= bounds[1, 1])
        )
        selected = indices[selected]

        if not len(selected) and not self.deselect_on_empty:
            return

        if additive and self.selected is not None:
            selected = np.unique(np.concatenate((self.selected, selected)))

        self.selected = selected

    def sync_viewer(self, viewer):
        """Sync the dendrogram with a neuroglancer viewer."""
        self._ngl_viewer = viewer

        # Activate the neuroglancer controls tab
        if hasattr(self, "_controls"):
            self._controls.tabs.setTabEnabled(2, True)

    @update_figure
    def update_point_labels(self):
        """Update the point labels."""
        if self._labels is None:
            return

        for i, l in enumerate(self._label_visuals):
            if l is None:
                continue
            l.set_text(self._labels[i])
            l._text = self._labels[i]

    @update_figure
    def update_point_position(self):
        """Update the point positions from the data file."""
        # Add a z coordinate of 1
        xyz = np.append(
            self._positions, np.ones((len(self._positions), 1)), axis=1
        ).astype(np.float32)

        # Update the positions of the points
        for vis in self._point_visuals:
            vis.geometry.positions.set_data(xyz[vis._point_ix])
            vis.geometry.positions.update_full()

        # Update the positions of the labels
        for i, l in enumerate(self._label_visuals):
            if l is None:
                continue
            l.local.position = (
                self._positions[i, 0] + 0.005,
                self._positions[i, 1],
                0,
            )

        # Update the positions of selected points
        if hasattr(self, "_highlight_visuals"):
            for vis in self._highlight_visuals:
                vis.geometry.positions.set_data(xyz[vis._point_ix])
                vis.geometry.positions.update_full()

    def move_points(self, new_positions, n_frames=20):
        """Move the points to new positions.

        Parameters
        ----------
        new_positions : array-like
                        The new positions to move to.
        n_frames :      int, optional
                        The number of frames to move over.
                        Default is 100.

        """
        # Check if the new positions are the same length as the old ones
        if len(new_positions) != len(self._positions):
            raise ValueError(
                f"New positions must be the same length as the old ones. "
                f"Got {len(new_positions)} and {len(self._positions)}."
            )

        # Calculate the vector from new to old positions
        vec = new_positions - self._positions

        # Slice the vector into n_frames segments
        # (we may want to do non-linear interpolation later)
        steps = vec / n_frames

        # Clear the label outlines if they exist
        if hasattr(self, "_label_line_group"):
            self._label_line_group.clear()

        # Stack n_frame times in a new dimension
        self.to_move = np.repeat(steps.reshape(-1, 2, 1), n_frames, axis=2)

    def _process_moves(self):
        """Move the points to the new positions."""
        # Check if we have any points to move
        if (
            not hasattr(self, "to_move")
            or self.to_move is None
            or self.to_move.size == 0
        ):
            return

        # Pop the first step from the to_move array
        new_positions = self._positions + self.to_move[:, :, 0]
        self._positions = new_positions
        self._data["x"], self._data["y"] = new_positions[:, 0], new_positions[:, 1]
        self.to_move = self.to_move[:, :, 1:]

        # Update the positions of the points
        self.update_point_position()

        # If there are no more steps, remove the to_move attribute
        if self.to_move.size == 0:
            del self.to_move
            if self.show_label_lines:
                self.make_label_lines()
            self._render_stale = True

    @update_figure
    def set_labels(self, indices, label):
        """Change the label of given point(s) in the figure."""
        if self._labels is None:
            raise ValueError("No labels were provided.")

        if not isinstance(indices, (list, np.ndarray, tuple, set)):
            indices = [indices]

        if isinstance(label, str):
            label = [label] * len(indices)

        self._labels[indices] = label
        for ix, lab in zip(indices, label):
            if self._label_visuals[ix] is not None:
                self._label_visuals[ix].set_text(lab)
                self._label_visuals[ix]._text = lab


class LabelSearch:
    """Class to search for and iterate over dendrogram labels.

    Parameters
    ----------
    scatter :       Scatterplot
                    The plot to search in.
    label :         str
                    The label to search for.
    rotate :        bool, optional
                    Whether to rotate through all occurrences of the label.
    go_to_first :   bool, optional
                    Whether to go to the first occurrence of the label at
                    initialization.
    regex :         bool, optional
                    Whether to interpret the label as a regular expression.

    """

    def __init__(self, scatter, label, rotate=True, go_to_first=True, regex=False):
        self.scatter = scatter
        self.label = label
        self._ix = None  # start with no index (will be initialized in `next` or `prev`)
        self.regex = regex
        self._rotate = rotate

        if scatter._labels is None:
            print("No labels available.")
            return

        # Search labels
        self.indices = self.search_labels(label)

        # Search IDs if no labels were found
        if len(self.indices) == 0:
            if isinstance(label, Number) or label.isdigit():
                self.indices = self.search_ids(int(label))

        # If still no labels found, return
        if len(self.indices) == 0:
            print(f"Label '{label}' not found.")
            return

        # Start at the first label
        if go_to_first:
            self.next()

    def search_labels(self, label):
        """Search for a label in the scatter."""
        if not self.regex:
            return np.where(self.scatter._labels == label)[0]
        else:
            return np.where(
                [
                    re.search(str(label), l) is not None
                    for l in self.scatter._labels_ordered
                ]
            )[0]

    def search_ids(self, id):
        """Search for an ID in the scatter."""
        return np.where(self.scatter._ids == id)[0]

    def __len__(self):
        return len(self.indices)

    def next(self):
        """Go to the next label."""
        if self._ix is None:
            self._ix = 0
        elif self._ix >= (len(self.indices) - 1):
            if not self._rotate:
                raise StopIteration
            else:
                self._ix = 0
        else:
            self._ix += 1

        self.scatter.camera.local.x = self.scatter._positions[self.indices[self._ix], 0]
        self.scatter.camera.local.y = self.scatter._positions[self.indices[self._ix], 1]

        self.scatter._render_stale = True

    def prev(self):
        """Go to the previous label."""
        if self._ix is None:
            self._ix = len(self.indices) - 1
        elif self._ix <= 0:
            if not self._rotate:
                raise StopIteration
            else:
                self._ix = len(self.indices) - 1
        else:
            self._ix -= 1

        self.scatter.camera.local.x = self.scatter._positions[self.indices[self._ix], 0]
        self.scatter.camera.local.y = self.scatter._positions[self.indices[self._ix], 1]

        self.scatter._render_stale = True
