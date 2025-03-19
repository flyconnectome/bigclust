import re
import math
import cmap

import pygfx as gfx
import numpy as np
import pylinalg as la

from functools import partial
from numbers import Number
from scipy.cluster.hierarchy import dendrogram as _dendrogram

from .utils import adjust_linkage_colors
from ._selection import SelectionGizmo
from ._figure import Figure, update_figure
from ._visuals import lines2gfx, points2gfx, text2gfx
from .controls import DendrogramControls

"""
TODOs:
- make y-axis labels move with the camera
- add a neuroglancer window (using octarine + plugin?)

"""


class Dendrogram(Figure):
    """A dendrogram plot.

    Notes:
    - the annotation tab in the control panel is by default disabled; set environment variable `BC_ANNOTATION=1` to enable

    Parameters
    ----------
    linkage :       numpy array
                    The linkage matrix from `scipy.cluster.hierarchy.linkage`.
    table :         A pandas Dataframe containing cluster metadata.
                    Must match the order of rows in the original distance matrix.
    labels :        str, optional
                    Name of column containing labels for the leaves in the dendrogram.
    clusters :      str, optional
                    Name of column containings labels for each leaf in the dendrogram.
                    Order of clusters must match the order of labels in the original distance matrix.
    cluster_colors : dict | str, optional
                    Dictionary of cluster colors. If None, a default palette is used. If string,
                    must be the name of a color palette from the `cmap` module.
    hover_info :    str, optional
                    Either the name of a column containing or hover information or a format string that
                    can reference multiple columns. Hover info must exist for each leaf in the dendrogram.
                    If given, hovering over a leaf will show the corresponding info.
    leaf_types :    str, optional
                    Column name specifying types for each leaf. Each unique type will be assigned a different marker.
    hinge_labels :  iterable, optional
                    List of labels to show at the hinge points of the dendrogram where clusters are merged.
                    Expected to match order of linkage.
    **kwargs :      dict, optional
                    Additional keyword arguments are passed to the `Figure` class.

    """

    x_spacing = 10
    axes_color = (1, 1, 1, 0.4)
    grid_color = (1, 1, 1, 0.1)

    def __init__(
        self,
        linkage,
        table,
        labels="label",
        ids="id",
        clusters="cluster",
        cluster_colors=None,
        hover_info="hover_info",
        leaf_types="dataset",
        hinge_labels=None,
        **kwargs,
    ):
        super().__init__(size=(1000, 400), **kwargs)

        self._linkage = np.asarray(linkage)
        self._table = table.reset_index(drop=True)
        self._default_label_col = labels
        self._labels = (
            np.array(table[labels].values) if labels is not None else None
        )  # make sure to use  a copy
        if self._labels is not None:
            # `_label_visuals` is in the same order as `_labels`
            self._label_visuals = [None] * len(self._labels)
        self._ids = np.asarray(table[ids]) if ids is not None else None
        self._clusters = np.asarray(table[clusters]) if clusters is not None else None
        self._leaf_types = (
            np.asarray(table[leaf_types]) if leaf_types is not None else None
        )
        self._rotate_labels = True
        self._selected = None

        if hover_info is not None:
            if "{" in hover_info:
                hover_info = self._table.apply(hover_info.format_map, axis=1).values
            else:
                hover_info = self._table[hover_info].values
        self._hover_info = np.asarray(hover_info) if hover_info is not None else None

        self._leaf_size = self.x_spacing / 10
        self._font_size = 2
        self.label_vis_limit = 200  # number of labels shown at once before hiding all
        self._label_refresh_rate = 30  # update labels every n frames

        # Let scipy do the heavy lifting
        self._dendrogram = _dendrogram(
            linkage,
            no_plot=True,
            labels=None,  # we don't actually need leafs in the dendrogram
            above_threshold_color="w",
            color_threshold=1.2,
        )

        # This tells us for each leaf in the original distance matrix where it is in the dendrogram
        self._leafs_order = np.array(self._dendrogram["leaves"])
        # This is the inverse of the above, i.e. for each leaf in the dendrogram (left to right),
        # what is its index in the original distance matrix
        self._leafs_order_inv = np.argsort(self._leafs_order)
        self._labels_ordered = (
            self._labels[self._leafs_order] if self._labels is not None else None
        )

        # This is the order of IDs the dendrogram left to right
        self._ids_ordered = (
            self._ids[self._leafs_order] if self._ids is not None else None
        )

        if self._clusters is not None:
            adjust_linkage_colors(self._dendrogram, self._clusters, cluster_colors)

        self._hinge_labels = (
            np.asarray(hinge_labels) if hinge_labels is not None else None
        )
        if self._hinge_labels is not None:
            assert len(self._hinge_labels) == len(self._linkage), (
                "Hinge labels must match the linkage matrix."
            )

            # Calculate the position of the hinge labels in the dendrogram
            import cocoa as cc

            # This is an (N, 2) array of x/y coordinates for our N hinge labels
            self._hinge_label_pos = cc.cluster_utils.map_linkage_to_dendrogram(
                self._dendrogram, self._linkage
            )
            # The coordinates match the coordinates on the matplotlib figure where the x coordinate is (index + 0.5) * 10
            # Because bigclust is using its own x-spacing, we need to adjust the x coordinates
            self._hinge_label_pos[:, 0] = (
                self._hinge_label_pos[:, 0] / 10 * self.x_spacing
            )

            # Prepare the visuals for the hinge labels
            self._hinge_label_visuals = [None] * len(self._hinge_labels)

        # Add the selection gizmo
        self.selection_gizmo = SelectionGizmo(
            self.renderer,
            self.camera,
            self.scene,
            callback_after=lambda x: self.select_leafs(
                x.bounds, additive="Control" in x._event_modifiers
            ),
        )

        # self.renderer.add_event_handler(self._mouse_press, "pointer_down")
        self.deselect_on_dclick = False

        # This group will hold text labels that need to move but not scale with the dendrogram
        self._text_group = gfx.Group()
        self.scene.add(self._text_group)

        # Generate the dendrogram visuals
        self.make_visuals()

        # Generate the labels
        self._label_group = gfx.Group()
        self._label_group.visible = True
        self._text_group.add(self._label_group)

        self._hinge_label_group = gfx.Group()
        self._hinge_label_group.visible = True
        self._text_group.add(self._hinge_label_group)

        # Setup hover info
        if self._hover_info is not None:

            def hover(event):
                # Note: we could use e.g. shift-hover to show
                # more/different info?
                if event.type == "pointer_enter":
                    # Translate position to world coordinates
                    pos = self._screen_to_world((event.x, event.y))

                    # Find the closest leaf
                    vis = event.current_target
                    leafs = vis.geometry.positions.data
                    dist = np.linalg.norm(leafs[:, :2] - pos[:2], axis=1)
                    closest = np.argmin(dist)

                    # Position and show the hover widget
                    self._hover_widget.local.position = leafs[closest]
                    self._hover_widget.visible = True

                    # Set the text
                    # N.B. there is some funny behaviour where repeatedly setting the same
                    # text will cause the bounding box to increase every time. To avoid this
                    # we have to reset the text to anything but an empty string.
                    self._hover_widget.children[1].set_text(
                        "asdfgasdfasdfsdafsfasdfasg"
                    )
                    self._hover_widget.children[1].set_text(
                        str(self._hover_info[self._leafs_order[vis._leaf_ix[closest]]])
                    )

                    # Scale the background to fit the text
                    bb = self._hover_widget.children[1].get_world_bounding_box()
                    extent = bb[1] - bb[0]

                    # The text bounding box is currently not very accurate. For example,
                    # a single-line text has no height. Hence, we need to add some padding:
                    extent = (extent + [0, 1.2, 0]) * 1.2
                    self._hover_widget.children[0].local.scale_x = extent[0]
                    self._hover_widget.children[0].local.scale_y = extent[1]

                elif self._hover_widget.visible:
                    self._hover_widget.visible = False

            for vis in self._leaf_visuals:
                vis.add_event_handler(hover, "pointer_enter", "pointer_leave")

            self._hover_widget = self.make_hover_widget()
            self.scene.add(self._hover_widget)

        # Show the dendrogram
        self.camera.show_object(self._dendrogram_group)
        # fig.camera.show_rect(dn.xlim[0], dn.xlim[1], dn.ylim[0], dn.ylim[1])

        # Add some keyboard shortcuts for moving and scaling the dendrogam
        def move_camera(x, y):
            self.camera.world.x += x
            self.camera.world.y += y
            self._render_stale = True
            self.canvas.request_draw()

        # self.key_events["ArrowLeft"] = lambda: self.set_xscale(
        #     self._dendrogram_group.local.matrix[0, 0] * 0.9
        # )
        # self.key_events["ArrowRight"] = lambda: self.set_xscale(
        #     self._dendrogram_group.local.matrix[0, 0] * 1.1
        # )
        self.key_events["ArrowLeft"] = lambda: move_camera(-10, 0)
        self.key_events[("ArrowLeft", ("Shift",))] = lambda: move_camera(-30, 0)
        self.key_events["ArrowRight"] = lambda: move_camera(10, 0)
        self.key_events[("ArrowRight", ("Shift",))] = lambda: move_camera(30, 0)

        self.key_events["ArrowUp"] = lambda: self.set_yscale(
            self._dendrogram_group.local.matrix[1, 1] * 1.1
        )
        self.key_events["ArrowDown"] = lambda: self.set_yscale(
            self._dendrogram_group.local.matrix[1, 1] * 0.9
        )
        self.key_events["Escape"] = lambda: self.deselect_all()
        self.key_events["l"] = lambda: self.toggle_labels()

        def _toggle_last_label():
            """Toggle between the last label and the original labels."""
            # If no controls, there is nothing to toggle
            if not hasattr(self, '_controls'):
                return
            self._controls.switch_labels()

        self.key_events['m'] = _toggle_last_label

        # def _deselect(event):
        #     print(event.type)

        # self.renderer.add_event_handler(_deselect, "double_click", "pointer_down")

        # Set the initial scale. As a rule of thumb, the dendrogram looks decent when it is
        # about 50 world units high
        self.set_yscale(50 / self.ylim[1])

        def _control_label_vis():
            """Show only labels currently visible."""
            if self._control_label_vis_every_n % self._label_refresh_rate:
                self._control_label_vis_every_n += 1
                return

            self._control_label_vis_every_n = 1

            if self._labels is not None and self._label_group.visible:
                # Check which leafs are currently visible
                pos = np.zeros((len(self), 2))
                pos[:, 0] = (np.arange(len(self)) + 0.5) * self.x_spacing
                iv = self.is_visible_pos(pos)

                # If more than the limit, don't show any labels
                if iv.sum() > self.label_vis_limit:
                    for i, t in enumerate(self._label_group.children):
                        t.visible = False
                else:
                    self.show_labels(np.where(iv)[0])
                    self.hide_labels(np.where(~iv)[0])

            if self._hinge_labels is not None and self._hinge_label_group.visible:
                # Check which hinge labels are currently visible
                iv = self.is_visible_pos(self._hinge_label_pos)

                # If more than the limit, don't show any labels
                if iv.sum() > self.label_vis_limit:
                    for i, t in enumerate(self._hinge_label_group.children):
                        t.visible = False
                else:
                    self.show_hinge_labels(np.where(iv)[0])
                    self.hide_hinge_labels(np.where(~iv)[0])

        # Turns out this is too slow to be run every frame - we're throttling it to every 30 frames
        if self._labels is not None or self._hinge_labels is not None:
            self._control_label_vis_every_n = 1
            self.add_animation(_control_label_vis)

    def _mouse_press(self, event):
        """For debugging: print event coordinates on Shift-click."""
        if "Shift" not in event.modifiers:
            return
        print(event.x, event.y)

    def __len__(self):
        """Number of original observations in the dendrogram."""
        return self._linkage.shape[0] + 1

    @update_figure
    def select_leafs(self, bounds, additive=False):
        """Select all selectable objects in the region."""
        # Get the positions and original indices of the leaf nodes
        positions_abs = []
        indices = []
        for l in self._leaf_visuals:
            positions_abs.append(
                la.vec_transform(l.geometry.positions.data, l.world.matrix)
            )
            indices.append(l._leaf_ix)
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

    @update_figure
    def deselect_all(self, *args):
        """Deselect all selected leafs."""
        self.selected = None

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
        self._labels_ordered = self._labels[self._leafs_order]
        self.update_leaf_labels()  # updates the visuals

    @property
    def show_label_lines(self):
        """Whether or to show label lines."""
        return getattr(self, "_show_label_lines", False)

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

            # Set offset to move labels below label lines
            pos = self._label_line_group.children[0].geometry.positions.data
            self._label_group.local.y = np.nanmin(pos[:, 1]) - 1

        elif not x and getattr(self, "_label_line_group", None):
            self._label_line_group.visible = False

            # Reset the offset for labels
            self._label_group.local.y = 0

        self._show_label_lines = x

    @property
    def labels_ordered(self):
        """Return the labels of leafs in the dendrogram in the order of the dendrogram."""
        return self._labels_ordered

    @property
    def selected_ids(self):
        """Return the IDs of selected leafs in the dendrogram."""
        if self.selected is None or not len(self.selected):
            return None
        if self._ids is None:
            raise ValueError("No IDs were provided.")
        return self._ids[self._leafs_order[self.selected]]

    @property
    def selected_labels(self):
        """Return the labels of selected leafs in the dendrogram."""
        if self.selected is None or not len(self.selected):
            return None
        if self._labels is None:
            raise ValueError("No labels were provided.")
        return self._labels[self._leafs_order[self.selected]]

    @property
    def selected_meta(self):
        """Return the metadata of selected leafs in the dendrogram."""
        if self.selected is None or not len(self.selected):
            return None
        if self._table is None:
            raise ValueError("No metadata was provided.")
        return self._table.iloc[self._leafs_order[self.selected]]

    @property
    def selected(self):
        """Return the selected leafs in the dendrogram."""
        return self._selected

    @selected.setter
    @update_figure
    def selected(self, x):
        """Select given leafs in the dendrogram."""
        if isinstance(x, type(None)):
            x = []
        elif isinstance(x, int):
            x = [x]

        if isinstance(x, np.ndarray) and x.dtype == bool:
            x = np.where(x)[0]

        # Set the selected leafs
        self._selected = np.asarray(sorted(x), dtype=int)

        # Clear existing selection
        if hasattr(self, "_selection_visuals"):
            for vis in self._selection_visuals:
                if vis.parent:
                    vis.parent.remove(vis)
            del self._selection_visuals

        # Create the new selection visuals
        if len(self._selected) > 0:
            self._selection_visuals = self.make_leafs(
                mask=np.isin(np.arange(len(self)), self._selected)
            )
            for vis in self._selection_visuals:
                vis.material.edge_color = "yellow"
                vis.material.edge_width = 0.2
                vis.material.color = (1, 1, 1, 0)
                self._dendrogram_group.add(vis)

        # Update the controls
        if hasattr(self, "_controls"):
            self._controls.update_ann_combo_box()

        if hasattr(self, "_ngl_viewer"):
            # `self._selected` is in order of the dendrogram, we need to translate it to the original order
            if len(self._selected) > 0:
                self._ngl_viewer.show(
                    # self._leafs_order[self._selected],
                    self._ids_ordered[self.selected],
                    add_as_group=getattr(self, "_add_as_group", False),
                )
            else:
                self._ngl_viewer.clear()

        if hasattr(self, "_synced_widgets"):
            for w, func in self._synced_widgets:
                try:
                    # func(self._leafs_order[self._selected])
                    func(self._ids_ordered[self.selected])
                except BaseException as e:
                    print(f"Failed to sync widget {w}:\n", e)

    @property
    def xlim(self):
        """X-axis limits of the dendrogram."""
        return (0, len(self) * self.x_spacing)

    @property
    def ylim(self):
        """Y-axis limits of the dendrogram."""
        return (0, self._linkage[:, 2].max())

    @property
    def dendrogram_group(self):
        return self._dendrogram_group

    @property
    def leafs(self):
        self._leaf_visuals

    @property
    def font_size(self):
        return self._font_size

    @font_size.setter
    @update_figure
    def font_size(self, size):
        self._font_size = size
        for t in self._label_visuals:
            if isinstance(t, gfx.Text):
                t.font_size = size
        for t in getattr(self, "_hinge_label_visuals", []):
            if isinstance(t, gfx.Text):
                t.font_size = size * 0.75

        # The hover widget is basically set up such that the text is size 1
        # So we just scale the whole thing accordingly when the font size changes
        if hasattr(self, "_hover_widget"):
            self._hover_widget.local.scale = [size, size, size]

    @property
    def leaf_size(self):
        return self._leaf_size

    @leaf_size.setter
    @update_figure
    def leaf_size(self, size):
        self._leaf_size = size
        for l in self._leaf_visuals:
            l.material.size = size

    @property
    def deselect_on_dclick(self):
        """Whether to deselect all leafs on double-click."""
        return getattr(self, "_deselect_on_dclick", False)

    @deselect_on_dclick.setter
    def deselect_on_dclick(self, x):
        assert isinstance(x, bool), "deselect_on_dclick must be a boolean."

        if x and not self.deselect_on_dclick:
            self.renderer.add_event_handler(self.deselect_all, "double_click")
        elif not x and self.deselect_on_dclick:
            self.renderer.remove_event_handler(self.deselect_all, "double_click")

        self._deselect_on_dclick = x

    @property
    def deselect_on_empty(self):
        """Whether to deselect all leafs when selection is empty."""
        return getattr(self, "_deselect_on_empty", False)

    @deselect_on_empty.setter
    def deselect_on_empty(self, x):
        assert isinstance(x, bool), "deselect_on_empty must be a boolean."

        self._deselect_on_empty = x

    @update_figure
    def set_xscale(self, x):
        self._dendrogram_group.local.scale_x = x

        def _set_xscale(t, x):
            if isinstance(t, gfx.Text):
                t.local.x = t._absolute_position_x * x

        self._text_group.traverse(lambda t: _set_xscale(t, x), skip_invisible=False)

    @update_figure
    def set_yscale(self, y):
        self._dendrogram_group.local.scale_y = y

        def _set_yscale(t, y):
            if isinstance(t, gfx.Text):
                # Text below the dendrogram does not have to be adjusted
                if t.local.y <= 0:
                    return
                t.local.y = t._absolute_position_y * y

        self._text_group.traverse(lambda t: _set_yscale(t, y), skip_invisible=False)

    def make_hover_widget(self, color=(1, 1, 1, 0.5), font_color=(0, 0, 0, 1)):
        """Generate a widget for hover info."""
        widget = gfx.Group()
        widget.visible = False
        widget.local.position = (
            0,
            0,
            2,
        )  # this means it's centered and slightly in front

        widget.add(
            gfx.Mesh(
                gfx.plane_geometry(1, 1),
                gfx.MeshBasicMaterial(color=color),
            )
        )
        widget.children[0].local.position = (
            0,
            0,
            1,
        )  # this means it's centered and slightly in front
        widget.add(
            text2gfx("Hover info", color=font_color, font_size=1, anchor="middle-center")
        )
        widget.children[1].local.position = (
            0,
            0,
            2,
        )  # this means it's centered and slightly in front

        return widget

    def make_axes(self, xticks=False, yticks=True, gridlines=True):
        """Generate the pygfx visuals for the axes."""
        # Make the axes lines
        # x_axis = np.array([[0, 0, 1], [self.xlim[1], 0, 1], [None, None, None]])
        y_axis = np.array([[0, 0, 1], [0, self.ylim[1], 1], [None, None, None]])

        group = gfx.Group()
        # group.add(lines2gfx(x_axis, color=self.axes_color))
        group.add(lines2gfx(y_axis, color=self.axes_color))

        if gridlines:
            gridlines = []
            div = 10 ** (round(math.log10(self.ylim[1])) - 1)
            for i in np.arange(0, self.ylim[1] + div, div):
                gridlines += [[0, i, -1], [self.xlim[1], i, -1], [None, None, None]]
            group.add(lines2gfx(np.array(gridlines), color=self.grid_color))

        # Make x-ticks and labels
        if xticks:
            xticks = []
            for i in range(1, len(self)):
                xticks += [
                    [i * self.x_spacing, 0, -1],
                    [i * self.x_spacing, -0.5, -1],
                    [None, None, None],
                ]

            group.add(lines2gfx(np.array(xticks), color=self.axes_color))

        # Make y-ticks and labels
        if yticks:
            yticks = []
            yticklabels = []
            div = 10 ** (round(math.log10(self.ylim[1])) - 1)
            for i in np.arange(0, self.ylim[1] + div, div):
                if div <= 1:
                    i = round(i, str(float(div)).split(".")[1].count("0") + 1)
                yticks += [[0, i, -1], [-1, i, -1], [None, None, None]]
                yticklabels.append(i)

            group.add(lines2gfx(np.array(yticks), color=self.axes_color))

            for i, t in enumerate(yticklabels):
                self._text_group.add(
                    text2gfx(
                        str(t),
                        position=(-1.5, t, 0),
                        font_size=1,
                        anchor="middle-right",
                    )
                )
                # Track where this label is supposed to show up (for scaling)
                self._text_group.children[-1]._absolute_position_x = -1.5
                self._text_group.children[-1]._absolute_position_y = t

        return group

    def make_dendrogram(self):
        """Generate the lines of the dendrogram."""
        lines = []

        # For colors: translate matplotlib C1-N into actual RGB colors
        cn_colors = [
            c for c in set(self._dendrogram["color_list"]) if str(c).startswith("C")
        ]
        palette = {
            f"C{i+1}": tuple(c)
            for i, c in enumerate(cmap.Colormap("tab10").iter_colors(len(cn_colors)))
        }

        # Now compile the lines
        colors = []
        for i in range(len(self._dendrogram["icoord"])):
            x = self._dendrogram["icoord"][i]
            y = self._dendrogram["dcoord"][i]
            for j in range(4):
                lines.append([x[j], y[j], 0])
                c = self._dendrogram["color_list"][i]
                if c in palette:
                    colors.append(palette[c])
                else:
                    colors.append(tuple(cmap.Color(c)))
            lines.append([None, None, None])

        return lines2gfx(np.array(lines), color=np.array(colors))

    def __get_lines_old(self):
        """Turn the linkage matrix into the lines of a dendrogram."""
        # Number of original observations
        N = self._linkage.shape[0] + 1

        pos = {i: (i * self.x_spacing, 0) for i in range(0, N)}

        # Each row of the linkage matrix represents a merge of two observations
        lines = []
        for i, row in enumerate(self._linkage):
            # The first two columns are the indices of the two observations
            # being merged. Note that they are 0-indexed.
            left, right = row[:2].astype(int)
            left_pos, right_pos = pos[left], pos[right]

            # The third column is the distance between the two observations
            distance = row[2]

            # The fourth column is the number of original observations in the
            # merged cluster (we don't currently need this)
            # count = row[3]

            # The new observation is at index N + i
            new = N + i

            # Record the position of the new observation
            pos[new] = ((left_pos[0] + right_pos[0]) / 2, distance)

            # Now we need to generate the two vertical and the new horizontal lines
            h1 = [(left_pos[0], left_pos[1]), (left_pos[0], distance), (None, None)]
            h2 = [(right_pos[0], right_pos[1]), (right_pos[0], distance), (None, None)]
            v = [(left_pos[0], distance), (right_pos[0], distance), (None, None)]

            lines += h1
            lines += h2
            lines += v

        return np.array(lines)

    def make_leafs(self, mask=None):
        """Generate points for the leafs of the dendrogram."""
        # For colors: translate matplotlib C1-N into actual RGB colors
        cn_colors = [
            c
            for c in set(self._dendrogram["leaves_color_list"])
            if str(c).startswith("C")
        ]
        palette = {
            f"C{i+1}": tuple(c)
            for i, c in enumerate(cmap.Colormap("tab10").iter_colors(len(cn_colors)))
        }

        if self._leaf_types is None:
            markers = np.full(len(self), "circle")
        else:
            assert len(self._leaf_types) == len(
                self
            ), "Length of leaf_types must match length of dendrogram."
            unique_types = np.unique(self._leaf_types)
            available_markers = list(gfx.MarkerShape)
            # Drop markers which look too similar to other
            available_markers.remove("ring")

            assert len(unique_types) <= len(
                available_markers
            ), "Only 10 unique types are supported."
            marker_map = dict(zip(unique_types, available_markers))
            markers = np.array(
                [marker_map[t] for t in self._leaf_types[self._dendrogram["leaves"]]]
            )

        leaf_pos = []
        colors = []
        for i in range(len(self)):
            if isinstance(mask, (list, np.ndarray)) and not mask[i]:
                continue
            leaf_pos.append([(i + 0.5) * self.x_spacing, 0, 1])

            c = self._dendrogram["leaves_color_list"][i]
            if c in palette:
                colors.append(palette[c])
            else:
                colors.append(tuple(cmap.Color(c)))
        colors = np.array(colors)
        leaf_pos = np.array(leaf_pos)

        if isinstance(mask, (list, np.ndarray)):
            markers = markers[mask]

        leafs = []
        for m in np.unique(markers):
            ix = markers == m
            # ix_dend = np.array(self._dendrogram["leaves"])[ix]
            leafs.append(
                points2gfx(
                    leaf_pos[ix],
                    color=colors[ix],
                    size_space="world",
                    marker=m,
                    pick_write=self._hover_info is not None,
                    size=self.leaf_size,
                )
            )
            # This keeps track of the original indices of the leafs
            leafs[-1]._leaf_ix = np.where(ix)[0]
        return leafs

    def make_visuals(self, labels=True, clear=False):
        """Generate the pygfx visuals for the dendrogram."""
        if clear:
            self.clear()

        self._axes_visuals = self.make_axes()
        self._dendrogram_visuals = self.make_dendrogram()
        self._leaf_visuals = self.make_leafs()

        # Create the group for the dendrogram
        self._dendrogram_group = gfx.Group()
        self._dendrogram_group._object_id = "dendrogram"
        self.scene.add(self._dendrogram_group)

        self._dendrogram_group.add(self._axes_visuals)
        self._dendrogram_group.add(self._dendrogram_visuals)
        self._dendrogram_group.add(*self._leaf_visuals)

    def make_label_lines(self):
        """Generate the lines for the label lines."""
        # Create a group and add to scene
        if not getattr(self, "_label_line_group", None):
            self._label_line_group = gfx.Group()
            self._label_line_group.visible = self.show_label_lines
            self.scene.add(self._label_line_group)

        # Clear the group (we might call this function to update the lines)
        self._label_line_group.clear()

        # Generate a dictionary mapping a unique label to the indices
        labels = {
            l: np.where(self._labels_ordered == l)[0]
            for l in np.unique(self._labels_ordered)
        }

        # For the y-positions we will try to tetris the labels into non-overlapping lines
        # Start with the shortest lines and assign y positions
        lines = []
        pos_matrix = np.zeros((len(labels), len(self)), dtype=bool)
        for l in sorted(labels, key=lambda x: labels[x][-1] - labels[x][0]):
            min_x, max_x = labels[l][0], labels[l][-1]
            for y in range(len(labels)):
                if ~np.any(pos_matrix[y, min_x : max_x + 1]):
                    pos_matrix[y, min_x : max_x + 1] = True
                    break

            # Now generate the line visuals
            # First: the horizontal line (only required if N > 1)
            if min_x != max_x:
                lines += [
                    [min_x * self.x_spacing, -y, 0],
                    [max_x * self.x_spacing, -y, 0],
                    [None, None, None],
                ]

            # Now the vertical little knobs to indicate position of labels
            for ix in labels[l]:
                lines += [
                    [ix * self.x_spacing, -y, 0],
                    [ix * self.x_spacing, -y + 0.7, 0],
                    [None, None, None],
                ]

        # Convert to (N, 3) numpy array
        lines = np.array(lines).astype(np.float32)

        # Last but not least:
        # (1) Add an offset so that the lines slightly below zero
        lines[:, 1] -= 2
        # (2) Adjust height
        lines[:, 1] *= 1.5
        # (3) Shift to the right by 0.5 to center them
        lines[:, 0] += 0.5 * self.x_spacing

        self._label_line_group.add(lines2gfx(lines, color="lightgrey"))

    @update_figure
    def show_labels(self, which=None):
        """Show labels for the leafs.

        Parameters
        ----------
        which : list, optional
                List of indices into the dendrogram (left to right) for which to show
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

            original_ix = self._leafs_order[ix]
            if self._label_visuals[original_ix] is None:
                t = text2gfx(
                    str(self._labels[original_ix]),
                    position=((ix + 0.5) * self.x_spacing, -0.25, 0),
                    font_size=self.font_size,
                    anchor="top-center",
                    pickable=True,
                )

                def _highlight(event, text):
                    self.find_label(text._text, go_to_first=False)

                t.add_event_handler(partial(_highlight, text=t), "double_click")

                # `_label_visuals` is in the same order as `_labels`
                self._label_visuals[original_ix] = t
                self._label_group.add(t)

                # Track where this label is supposed to show up (for scaling)
                t._absolute_position_x = (ix + 0.5) * self.x_spacing
                t._absolute_position_y = -0.25

                # Center the text
                t.text_align = "center"

                # Rotate labels to avoid overlap
                if self._rotate_labels:
                    t.anchor = "topright"
                    t.text_align = "right"
                    t.local.euler_z = (
                        1  # slightly slanted, use `math.pi / 2` for 90 degress
                    )
                    t.local.y = t._absolute_position_y = -1

            self._label_visuals[original_ix].visible = True

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
            original_ix = self._dendrogram["leaves"][ix]
            if self._label_visuals[original_ix] is None:
                continue

            self._label_visuals[original_ix].visible = False

    @update_figure
    def toggle_labels(self):
        """Toggle the visibility of labels."""
        self._label_group.visible = not self._label_group.visible

    @update_figure
    def show_hinge_labels(self, which=None):
        """Show labels for hinges.

        Parameters
        ----------
        which : list, optional
                List of indices into the linkage for which to show
                the labels. If None, all hinge labels are shown.

        """
        if self._hinge_labels is None:
            return

        if which is None:
            which = np.arange(len(self._hinge_labels))
        elif isinstance(which, Number):
            which = np.array([which])
        elif isinstance(which, list):
            which = np.array(which)

        if not isinstance(which, (list, np.ndarray)):
            raise ValueError(f"Expected list or array, got {type(which)}.")

        for ix in which:
            if self._hinge_label_visuals[ix] is None:
                pos = tuple(self._hinge_label_pos[ix])
                t = text2gfx(
                    str(self._hinge_labels[ix]),
                    position=(pos[0], pos[1] * self._dendrogram_group.local.scale_y, 0),
                    font_size=self.font_size * 0.75,
                    color=(1, 1, 1, 1),
                    anchor="bottom-center",
                    pickable=False,
                )

                # Track the label and add to scene
                self._hinge_label_visuals[ix] = t
                self._hinge_label_group.add(t)

                # Track where this label is supposed to show up (for scaling)
                t._absolute_position_x = pos[0]
                t._absolute_position_y = pos[1]

            self._hinge_label_visuals[ix].visible = True

    @update_figure
    def hide_hinge_labels(self, which=None):
        """Hide labels for hinges.

        Parameters
        ----------
        which : list, optional
                List of indices into the linkage for which to hide
                the labels. If None, all hinge labels are hidden.

        """
        if self._hinge_labels is None:
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
            if self._hinge_label_visuals[ix] is None:
                continue
            self._hinge_label_visuals[ix].visible = False

    def show_controls(self):
        """Show controls."""
        if self._is_jupyter:
            print("Currently not supported")
        else:
            if not hasattr(self, "_controls"):
                self._controls = DendrogramControls(
                    self,
                    labels=list(set(self._labels)),
                    datasets=list(set(self._leaf_types)),
                )
            self._controls.show()

    def _toggle_controls(self):
        """Switch controls on and off."""
        if self._is_jupyter:
            if self.widget.toolbar:
                self.widget.toolbar.toggle()
        else:
            if not hasattr(self, "_controls"):
                self.show_controls()
            elif self._controls.isVisible():
                self.hide_controls()
            else:
                self.show_controls()

    @update_figure
    def find_label(
        self, label, regex=False, highlight=True, go_to_first=True, verbose=True
    ):
        """Find and center the dendrogram on a given label.

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
            self.highlight_leafs(ls.indices)

        if verbose:
            self.show_message(f"Found {len(ls)} occurrences of '{label}'", duration=3)

        return ls

    @update_figure
    def highlight_leafs(self, leafs, color="y"):
        """Highlight leafs in the dendrogram.

        Parameters
        ----------
        leafs : str | iterable | None
                Can be either:
                 - a string with a label to highlight.
                 - an iterable with indices (left to right) of leafs to highlight
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
        if leafs is None:
            return

        if isinstance(leafs, str):
            for i, label in enumerate(self._labels):
                if label != leafs:
                    continue

                if self._label_visuals[i] is None:
                    dend_ix = self._leafs_order_inv[i]
                    self.show_labels(dend_ix)
                visual = self._label_visuals[i]

                visual.material._original_color = visual.material.color
                visual.material.color = color
        elif isinstance(leafs, (list, np.ndarray)):
            for ix in leafs:
                # Index in the original order
                ix_org = self._leafs_order[ix]
                if self._label_visuals[ix_org] is None:
                    self.show_labels(ix)
                visual = self._label_visuals[ix_org]

                visual.material._original_color = visual.material.color
                visual.material.color = color

    def close(self):
        """Close the figure."""
        if hasattr(self, "_controls"):
            self._controls.close()
        super().close()

    def sync_viewer(self, viewer):
        """Sync the dendrogram with a neuroglancer viewer."""
        self._ngl_viewer = viewer

        # Activate the neuroglancer controls tab
        if hasattr(self, "_controls"):
            self._controls.tabs.setTabEnabled(2, True)

    def sync_widget(self, widget, callback=None):
        """Connect a widget to the dendrogram.

        Parameters
        ----------
        widget
                The widget to sync.
        callback
                The function to call. If `None`, the widget must implement a
                `.select()` method that takes a list of IDs to select.

        """
        if callback is None:
            assert hasattr(widget, "select") and callable(
                widget.select
            ), "Widget must have a `select` method that takes a list of IDs to select."
            callback = widget.select

        if not hasattr(self, "_synced_widgets"):
            self._synced_widgets = []

        self._synced_widgets.append((widget, callback))

    def set_viewer_colors(self, colors):
        """Set the colors for the neuroglancer viewer.

        Parameters
        ----------
        colors :    dict
                    Dictionary of colors keyed by IDs: {id: color, ...}
        """
        if not hasattr(self, "_ngl_viewer"):
            raise ValueError("No neuroglancer viewer is connected.")

        assert isinstance(colors, dict), "Colors must be a dictionary."
        self._ngl_viewer.set_colors(colors)

    def set_viewer_color_mode(self, mode, palette="seaborn:tab20"):
        """Set the color mode for the neuroglancer viewer.

        Parameters
        ----------
        mode :  "dataset" | "cluster" | "label" | "default"
                The color mode to use.

        """
        if not hasattr(self, "_ngl_viewer"):
            raise ValueError("No neuroglancer viewer is connected.")

        assert mode in ["dataset", "cluster", "label", "default"], "Invalid mode."

        if mode == "cluster":
            # Collect colors for each leaf
            colors = {}
            for vis in self._leaf_visuals:
                this_ids = self._ids_ordered[vis._leaf_ix]
                colors.update(zip(this_ids, vis.geometry.colors.data))
        elif mode == "label":
            labels_unique = np.unique(self._labels_ordered)
            palette = cmap.Colormap(palette)
            colormap = {
                l: c.hex
                for l, c in zip(labels_unique, palette.iter_colors(len(labels_unique)))
            }
            colors = {l: colormap[l] for l in self._labels_ordered}
        elif mode == "dataset":
            palette = cmap.Colormap(palette)
            colormap = {
                i: c.hex
                for i, c in zip(
                    range(len(self._leaf_visuals)),
                    palette.iter_colors(len(self._leaf_visuals)),
                )
            }
            colors = {}
            for i, vis in enumerate(self._leaf_visuals):
                this_ids = self._ids_ordered[vis._leaf_ix]
                this_c = colormap[i]
                colors.update({i: this_c for this_id in this_ids})
        elif mode == "default":
            self._ngl_viewer.set_default_colors()
            return

        self.set_viewer_colors(colors)

    @update_figure
    def set_leaf_label(self, indices, label):
        """Change the label of given leaf(s) in the dendrogram.

        Indices are expected to be in the order of dendrogram left to right!
        """
        if self._labels is None:
            raise ValueError("No labels were provided.")

        if not isinstance(indices, (list, np.ndarray, tuple, set)):
            indices = [indices]

        original_ix = self._leafs_order[indices]

        if isinstance(label, str):
            label = [label] * len(indices)

        self._labels[original_ix] = label
        for ix, lab in zip(indices, label):
            ix_org = self._leafs_order[ix]
            if self._label_visuals[ix_org] is not None:
                self._label_visuals[ix_org].set_text(lab)
                self._label_visuals[ix_org]._text = lab

    @update_figure
    def update_leaf_labels(self):
        """Update the labels of the leafs in the dendrogram."""
        if self._labels is None:
            return

        for i, l in enumerate(self._label_visuals):
            if l is None:
                continue
            l.set_text(self._labels[i])
            l._text = self._labels[i]


class LabelSearch:
    """Class to search for and iterate over dendrogram labels.

    Parameters
    ----------
    dendrogram :    Dendrogram
                    The dendrogram to search in.
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

    def __init__(self, dendrogram, label, rotate=True, go_to_first=True, regex=False):
        self.dendrogram = dendrogram
        self.label = label
        self._ix = None  # start with no index (will be initialized in `next` or `prev`)
        self.regex = regex
        self._rotate = rotate

        if dendrogram._labels is None:
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
        """Search for a label in the dendrogram."""
        if not self.regex:
            return np.where(self.dendrogram._labels_ordered == label)[0]
        else:
            return np.where(
                [
                    re.search(str(label), str(l)) is not None
                    for l in self.dendrogram._labels_ordered
                ]
            )[0]

    def search_ids(self, id):
        """Search for an ID in the dendrogram."""
        return np.where(self.dendrogram._ids[self.dendrogram._leafs_order] == id)[0]

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

        self.dendrogram.camera.local.x = (
            self.indices[self._ix] + 0.5
        ) * self.dendrogram.x_spacing
        self.dendrogram.camera.local.y = 0

        self.dendrogram._render_stale = True

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

        self.dendrogram.camera.local.x = (
            self.indices[self._ix] + 0.5
        ) * self.dendrogram.x_spacing
        self.dendrogram.camera.local.y = 0

        self.dendrogram._render_stale = True
