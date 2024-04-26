import math
import cmap

import pygfx as gfx
import numpy as np

from scipy.cluster.hierarchy import dendrogram as _dendrogram

from .utils import adjust_linkage_colors
from ._selection import SelectionGizmo
from ._figure import Figure
from ._visuals import lines2gfx, points2gfx, text2gfx


class Dendrogram(Figure):
    """A dendrogram plot.

    Parameters
    ----------
    linkage :       ndarray
                    The linkage matrix from `scipy.cluster.hierarchy.linkage`.
    labels :        list, optional
                    Labels for the leafs in the dendrogram.
    clusters :      list, optional
                    List of clusters. Must contain one label for each leaf in the dendrogram.
                    Order of clusters must match the order of labels in the original distance matrix.
    cluster_colors : dict | str, optional
                    Dictionary of cluster colors. If None, a default palette is used. If string,
                    must be the name of a color palette from the `cmap` module.
    hover_info :    list, optional
                    List of hover info for each leaf in the dendrogram. If given, hovering over a leaf
                    will show the corresponding info.
    leaf_types :    list, optional
                    A list of types for each leaf. Each unique type will be assigned a different marker.
    **kwargs :      dict, optional
                    Additional keyword arguments are passed to the `Figure` class.

    """

    x_spacing = 10
    leaf_size = x_spacing / 10
    axes_color = (1, 1, 1, 0.4)
    grid_color = (1, 1, 1, 0.1)

    def __init__(
        self,
        linkage,
        labels=None,
        clusters=None,
        cluster_colors=None,
        hover_info=None,
        leaf_types=None,
        **kwargs,
    ):
        super().__init__(size=(1000, 400), **kwargs)

        self._linkage = np.asarray(linkage)
        self._labels = labels
        self._clusters = clusters
        self._selected = None
        self._hover_info = hover_info
        self._leaf_types = leaf_types

        # Let scipy do the heavy lifting
        self._dendrogram = _dendrogram(
            linkage,
            no_plot=True,
            labels=labels,
            above_threshold_color="w",
            color_threshold=0.2,
        )

        if self._clusters is not None:
            adjust_linkage_colors(self._dendrogram, self._clusters, cluster_colors)

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

        # This group will hold text labels that need to move but not scale with the dendrogram
        self._text_group = gfx.Group()
        self.scene.add(self._text_group)

        # Generate the dendrogram visuals
        self.make_visuals()

        # Generate the labels
        self._label_group = gfx.Group()
        self._label_group.visible = False
        self._text_group.add(self._label_group)
        if labels is not None:  # replace with `if labels`
            for i, t in enumerate(self._dendrogram["ivl"]):
                self._label_group.add(
                    text2gfx(
                        str(t),
                        position=((i + 0.5) * self.x_spacing, -0.25, 0),
                        font_size=1,
                        anchor="topmiddle",
                    )
                )
                # Track where this label is supposed to show up (for scaling)
                self._label_group.children[-1]._absolute_position_x = (
                    i + 0.5
                ) * self.x_spacing
                self._label_group.children[-1]._absolute_position_y = -0.25

                # Center the text
                self._label_group.children[-1].geometry.text_align = "center"

            # Rotate labels to avoid overlap
            rotate = True
            if rotate:
                for i, t in enumerate(self._label_group.children):
                    t.geometry.anchor = "topright"
                    t.geometry.text_align = "right"
                    t.local.euler_z = (
                        1  # slightly slanted, use `math.pi / 2` for 90 degress
                    )
                    t.local.y = t._absolute_position_y = -1

        # Setup hover info
        if hover_info is not None:
            def hover(event):
                if event.type == "pointer_enter":
                    # Translate position to world coordinates
                    pos = self._screen_to_world((event.x, event.y))

                    # Find the closest leaf
                    leafs = self._leaf_visuals.geometry.positions.data
                    dist = np.linalg.norm(leafs[:, :2] - pos[:2], axis=1)
                    closest = np.argmin(dist)

                    # Position and show the hover widget
                    self._hover_widget.local.position = leafs[closest]
                    self._hover_widget.visible = True

                    # Set the text
                    # N.B. there is some funny behaviour where repeatedly setting the same
                    # text will cause the bounding box to increase every time. To avoid this
                    # we have to reset the text to anything but an empty string.
                    self._hover_widget.children[1].geometry.set_text("asfasdfasdfasdf")
                    self._hover_widget.children[1].geometry.set_text(str(hover_info[self._dendrogram["leaves"][closest]]))

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


            self._leaf_visuals.add_event_handler(
                hover, "pointer_enter", "pointer_leave"
            )

            self._hover_widget = self.make_hover_widget()
            self.scene.add(self._hover_widget)


        # Show the dendrogram
        self.camera.show_object(self._dendrogram_group)
        # fig.camera.show_rect(dn.xlim[0], dn.xlim[1], dn.ylim[0], dn.ylim[1])

        # Add some keyboard shortcuts for scaling the dendrogam
        self.key_events["ArrowLeft"] = lambda: self.set_xscale(
            self._dendrogram_group.local.matrix[0, 0] * 0.9
        )
        self.key_events["ArrowRight"] = lambda: self.set_xscale(
            self._dendrogram_group.local.matrix[0, 0] * 1.1
        )
        self.key_events["ArrowUp"] = lambda: self.set_yscale(
            self._dendrogram_group.local.matrix[1, 1] * 1.1
        )
        self.key_events["ArrowDown"] = lambda: self.set_yscale(
            self._dendrogram_group.local.matrix[1, 1] * 0.9
        )
        self.key_events["Escape"] = lambda: self.deselect_all()
        self.key_events["l"] = lambda: self.toggle_labels()

        # def _deselect(event):
        #     print(event.type)

        # self.renderer.add_event_handler(_deselect, "double_click", "pointer_down")

        # Set the initial scale. As a rule of thumb, the dendrogram looks decent when it is
        # about 50 world units high
        self.set_yscale(50 / self.ylim[1])

        def _control_label_vis():
            if self._control_label_vis_every_n % 30:
                self._control_label_vis_every_n += 1
                return

            self._control_label_vis_every_n = 1

            if not self._label_group.visible:
                return
            iv = self.is_visible(self._label_group.children)

            if iv.sum() > 100:
                for i, t in enumerate(self._label_group.children):
                    t.visible = False
            else:
                for i, t in enumerate(self._label_group.children):
                    t.visible = iv[i]

        # Turns out this is too slow to be run every frame - we're throttling it to every 30 frames
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

    def select_leafs(self, bounds, additive=False):
        """Select all selectable objects in the region."""
        # Get the positions of the leaf nodes
        positions = np.copy(self._leaf_visuals.geometry.positions.data)

        # Transform positions into absolute coordinates
        positions_abs = (
            np.hstack((positions, np.zeros(positions.shape[0]).reshape(-1, 1)))
            @ self._leaf_visuals.world.matrix
        )[:, :-1]

        # Check if any of the points are within the selection region
        selected = (
            (positions_abs[:, 0] >= bounds[0, 0])
            & (positions_abs[:, 0] <= bounds[1, 0])
            & (positions_abs[:, 1] >= bounds[0, 1])
            & (positions_abs[:, 1] <= bounds[1, 1])
        )

        if additive and self.selected is not None:
            selected = np.unique(np.concatenate((self.selected, np.where(selected)[0])))

        self.selected = selected

    def deselect_all(self):
        """Deselect all selected leafs."""
        self.selected = None

    @property
    def selected(self):
        """Return the selected leafs in the dendrogram."""
        return self._selected

    @selected.setter
    def selected(self, x):
        """Select given leafs in the dendrogram."""
        if isinstance(x, type(None)):
            x = []
        elif isinstance(x, int):
            x = [x]

        if isinstance(x, np.ndarray) and x.dtype == bool:
            x = np.where(x)[0]

        # Set the selected leafs
        self._selected = np.asarray(x)

        # Clear existing selection
        if hasattr(self, "_selection_visuals"):
            self._selection_visuals.parent.remove(self._selection_visuals)
            del self._selection_visuals

        # Create the new selection visuals
        if len(self._selected) > 0:
            self._selection_visuals = self.make_leafs(
                mask=np.isin(np.arange(len(self)), self._selected)
            )
            self._selection_visuals.material.edge_color = "yellow"
            self._selection_visuals.material.edge_width = 0.2
            self._selection_visuals.material.color = (1, 1, 1, 0)
            self._dendrogram_group.add(self._selection_visuals)

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

    def set_xscale(self, x):
        self._dendrogram_group.local.scale_x = x

        def _set_xscale(t, x):
            if isinstance(t, gfx.Text):
                t.local.x = t._absolute_position_x * x

        self._text_group.traverse(lambda t: _set_xscale(t, x), skip_invisible=False)

    def set_yscale(self, y):
        self._dendrogram_group.local.scale_y = y

        def _set_yscale(t, y):
            if isinstance(t, gfx.Text):
                # Text below the dendrogram does not have to be adjusted
                if t.local.y > 0:
                    t.local.y = t._absolute_position_y * y

        self._text_group.traverse(lambda t: _set_yscale(t, y), skip_invisible=False)

    def make_hover_widget(self, color=(1, 1, 1, 0.5), font_color=(0, 0, 0, 1)):
        """Generate a widget for hover info."""
        widget = gfx.Group()
        widget.visible = False
        widget.local.position = (0, 0, 2)  # this means it's centered and slightly in front

        widget.add(gfx.Mesh(
            gfx.plane_geometry(1, 1),
            gfx.MeshBasicMaterial(color=color),
        ))
        widget.children[0].local.position = (0, 0, 1)  # this means it's centered and slightly in front
        widget.add(text2gfx("Hover info", color=font_color, font_size=1, anchor="center"))
        widget.children[1].local.position = (0, 0, 2)  # this means it's centered and slightly in front

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
                        anchor="middleright",
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

        leafs = []
        colors = []
        for i in range(len(self)):
            if isinstance(mask, (list, np.ndarray)) and not mask[i]:
                continue
            leafs.append([(i + 0.5) * self.x_spacing, 0, 1])

            c = self._dendrogram["leaves_color_list"][i]
            if c in palette:
                colors.append(palette[c])
            else:
                colors.append(tuple(cmap.Color(c)))

        return points2gfx(
            np.array(leafs),
            color=np.array(colors),
            size_space="world",
            marker="circle",
            pick_write=self._hover_info is not None,
            size=self.leaf_size,
        )

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
        self._dendrogram_group.add(self._leaf_visuals)

    def toggle_labels(self):
        """Toggle the visibility of labels."""
        self._label_group.visible = not self._label_group.visible


def dendrogram(
    Z, hanging=False, labels=None, clusters=None, cluster_colors=None, hover_info=None
):
    # Create the Figure
    fig = Dendrogram(
        Z,
        labels=labels,
        clusters=clusters,
        cluster_colors=cluster_colors,
        hover_info=hover_info,
    )

    return fig
