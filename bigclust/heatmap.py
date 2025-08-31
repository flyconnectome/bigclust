import math
import cmap

import pygfx as gfx
import numpy as np
import pylinalg as la

from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform

from ._selection import SelectionGizmo
from ._figure import BaseFigure
from ._visuals import lines2gfx, points2gfx, text2gfx, heatmap2gfx
from .utils import apply_matrix


class Heatmap(BaseFigure):
    """A heatmap.

    Parameters
    ----------
    data :          pandas DataFrame
                    2D data to plot.
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
    leaf_size = x_spacing / 30
    axes_color = (1, 1, 1, 0.4)
    grid_color = (1, 1, 1, 0.1)

    def __init__(
        self,
        data,
        labels=None,
        show=True,
        col_linkage=None,
        row_linkage=None,
        colormap="viridis",
        **kwargs,
    ):
        super().__init__(size=(800, 800), **kwargs)

        self._data = np.asarray(data)
        self._labels = labels
        self._selected = None
        self.col_linkage = col_linkage
        self.row_linkage = row_linkage

        # Set up scenes
        self.canvas_heatmap = gfx.Scene()
        self.canvas_margin_x = gfx.Scene()
        self.canvas_margin_y = gfx.Scene()
        self.background_scene = gfx.Scene()

        # Add the background (from BaseFigure) to the scene
        self.canvas_heatmap.add(gfx.Background(None, self._background))

        # Add a cameras
        self.camera_heatmap = gfx.NDCCamera()
        self.camera_margin_x = gfx.NDCCamera()
        self.camera_margin_y = gfx.NDCCamera()
        self.camera_background = gfx.NDCCamera()

        # Setup viewports
        self.viewport_heatmap = gfx.Viewport(self.renderer)
        self.viewport_margin_x = gfx.Viewport(self.renderer)
        self.viewport_margin_y = gfx.Viewport(self.renderer)
        self.viewport_background = gfx.Viewport(self.renderer)

        # self.renderer.add_event_handler(self._mouse_press, "pointer_down")

        # This group will hold text labels that need to move but not scale with the dendrogram
        # self._text_group = gfx.Group()
        # self.scene.add(self._text_group)

        # Generate the dendrogram visuals
        # self.make_visuals()

        # Generate the labels
        # self._label_group = gfx.Group()
        # self._label_group.visible = False
        # self._text_group.add(self._label_group)
        # if labels is not None:  # replace with `if labels`
        #     for i, t in enumerate(self._dendrogram["ivl"]):
        #         self._label_group.add(
        #             text2gfx(
        #                 str(t),
        #                 position=((i + 0.5) * self.x_spacing, -0.25, 0),
        #                 font_size=1,
        #                 anchor="top-center",
        #             )
        #         )
        #         # Track where this label is supposed to show up (for scaling)
        #         self._label_group.children[-1]._absolute_position_x = (
        #             i + 0.5
        #         ) * self.x_spacing
        #         self._label_group.children[-1]._absolute_position_y = -0.25

        #         # Center the text
        #         self._label_group.children[-1].text_align = "center"

        #     # Rotate labels to avoid overlap
        #     rotate = True
        #     if rotate:
        #         for i, t in enumerate(self._label_group.children):
        #             t.anchor = "topright"
        #             t.text_align = "right"
        #             t.local.euler_z = (
        #                 1  # slightly slanted, use `math.pi / 2` for 90 degress
        #             )
        #             t.local.y = t._absolute_position_y = -1

        # Generate marginal dendrogram
        self.col_marginal = self.make_col_dendrogram()
        self.canvas_margin_x.add(self.col_marginal)

        # Generate heatmap
        # Note: we're making the heatmap 2x2 so it has the same extent as the NDCCamera
        # which is (-1, 1) in both x and y
        self._heatmap = heatmap2gfx(self._data, width=2, height=2, colormap=colormap)
        self._heatmap.local.z = (
            0.9  # this moves the heatmap to the back so we can see the selection gizmo
        )
        self.canvas_heatmap.add(self._heatmap)

        # Generate markers on the margin
        self.row_marginal, self._marker_visuals = self.make_row_marginal()
        self.canvas_margin_y.add(self.row_marginal)

        # Generate hover widget
        # Setup hover info
        if self._labels is not None:

            def hover(event):
                # Note: we could use e.g. shift-hover to show
                # more/different info?
                if event.type == "pointer_enter":
                    # Translate position to world coordinates
                    pos = self._screen_to_world(
                        (event.x, event.y), viewport=self.viewport_margin_y
                    )

                    # Find the closest leaf
                    vis = event.current_target
                    leafs = apply_matrix(vis.geometry.positions.data, vis.world.matrix)
                    dist = np.linalg.norm(leafs[:, :2] - pos[:2], axis=1)
                    closest = np.argmin(dist)

                    # Position and show the hover widget
                    # self._hover_widget.local.position = leafs[closest]
                    self._hover_widget.visible = True

                    # Set the text
                    # N.B. there is some funny behaviour where repeatedly setting the same
                    # text will cause the bounding box to increase every time. To avoid this
                    # we have to reset the text to anything but an empty string.
                    self._hover_widget.children[1].set_text(
                        "asdfgasdfasdfsdafsfasdfasg"
                    )
                    self._hover_widget.children[1].set_text(
                        str(self._labels[self._leafs_order[vis._leaf_ix[closest]]])
                    )

                elif self._hover_widget.visible:
                    self._hover_widget.visible = False

            for vis in self._marker_visuals:
                vis.add_event_handler(hover, "pointer_enter", "pointer_leave")

            self._hover_widget = self.make_hover_widget()
            self.canvas_heatmap.add(self._hover_widget)  # the hover widget is displayed on top of the heatmap

        def layout(event=None):
            w, h = self.renderer.logical_size
            self.viewport_heatmap.rect = 0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h
            self.viewport_margin_x.rect = 0.1 * w, 0, 0.9 * w, 0.1 * h
            self.viewport_margin_y.rect = 0, 0.1 * h, 0.1 * w, 0.9 * h
            self.reset_view()

        layout()
        self.renderer.add_event_handler(layout, "resize")

        # Add a controller
        # self.controller = gfx.PanZoomController(
        #     self.camera, register_events=self.renderer
        # )

        def _set_view(x):
            # print(x.bounds)
            self.set_view(
                x.bounds[0, 0],
                x.bounds[1, 0],
                x.bounds[0, 1],
                x.bounds[1, 1],
            )

        # Add the heatmap selection gizmo
        self.selection_gizmo = SelectionGizmo(
            self.viewport_heatmap,
            self.camera_heatmap,
            self.canvas_heatmap,
            line_width=2,
            force_square=True,
            callback_after=_set_view,
            name="HeatmapSelection",
        )
        # Make sure the Gizmo is above the heatmap and visible
        self.selection_gizmo.local.z = 0

        # Double click will reset the view
        self.renderer.add_event_handler(lambda x: self.reset_view(), "double_click")

        # Add the selection gizmo for the labels
        self.selection_gizmo_y = SelectionGizmo(
            self.viewport_margin_y,
            self.camera_margin_y,
            self.canvas_margin_y,
            line_width=2,
            edge_color="r",
            force_square=False,
            callback_after=lambda x: self.select_leafs(
                x.bounds, additive="Control" in x._event_modifiers
            ),
            name="LabelSelection",
        )

        # self.camera_heatmap.show_object(self._heatmap)
        # self.camera_heatmap.show_rect(-0.5, 0.5, -0.5, 0.5)
        # self.camera_margin_x.show_rect(0, len(self._data) * 10, 0, 1)
        # self.col_marginal.local.scale_y = 200
        self.reset_view()  # set view to default

        # Show the dendrogram
        # self.camera_heatmap.show_object(self._heatmap)
        # fig.camera.show_rect(dn.xlim[0], dn.xlim[1], dn.ylim[0], dn.ylim[1])

        # self.renderer.add_event_handler(_deselect, "double_click", "pointer_down")

        # This starts the animation loop
        if show and not self._is_jupyter:
            self.show()

    def _animate(self):
        """Animate the scene."""
        self._run_user_animations()

        if self._show_fps:
            with self.stats:
                self.viewport_background.render(
                    self.background_scene, self.camera_background, flush=False
                )
            self.stats.render(flush=False)
        else:
            self.viewport_background.render(
                self.background_scene, self.camera_background, flush=False
            )

        self.viewport_heatmap.render(
            self.canvas_heatmap, self.camera_heatmap, flush=False
        )
        self.viewport_margin_x.render(
            self.canvas_margin_x, self.camera_margin_x, flush=False
        )
        self.viewport_margin_y.render(
            self.canvas_margin_y, self.camera_margin_y, flush=False
        )

        self.renderer.flush()

        self.canvas.request_draw()

    def _screen_to_world(self, pos, viewport=None):
        """Translate screen position to world coordinates."""
        if viewport is None:
            viewport = gfx.Viewport.from_viewport_or_renderer(self.renderer)

        print(viewport, pos)

        if not viewport.is_inside(*pos):
            return None

        # Get position relative to viewport
        pos_rel = (
            pos[0] - viewport.rect[0],
            pos[1] - viewport.rect[1],
        )
        vs = viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        # We're using the same NDC cameras for all viewports, so it shouldn't
        # matter which we use here
        pos_ndc += la.vec_transform(
            self.camera_heatmap.world.position, self.camera_heatmap.camera_matrix
        )
        pos_world = la.vec_unproject(pos_ndc[:2], self.camera_heatmap.camera_matrix)

        return pos_world

    def _mouse_press(self, event):
        """For debugging: print event coordinates on Shift-click."""
        if "Shift" not in event.modifiers:
            return
        print(event.x, event.y)

    def __len__(self):
        """Number of original observations in the dendrogram."""
        return self._data.shape[0]

    def select_leafs(self, bounds, additive=False):
        """Select all selectable objects in the region."""
        print(bounds)
        return
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

    def set_view(self, x1, x2, y1, y2):
        """Set the view of the heatmap.

        Parameters
        ----------
        x1,x2,y1,y2 : float
            The bounds of the view. In space of the heatmap visual!
            The way we generated the heatmap it will always be in
            the range (-1, 1) in both x and y. In the future,
            we should change this to something more intuitive like
            data space or at least normalised to 0-1.

        """
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # We need to (1) scale it to fit the viewport and (2) center the view on the selected region

        # (1) Scale the view such that the selected region fits the viewport
        extent = (x2 - x1, y2 - y1)

        if extent[0] == 0 or extent[1] == 0:
            return

        self._heatmap.local.scale_x = 2 / extent[0]
        self._heatmap.local.scale_y = 2 / extent[1]

        # (2) Center the view while taking the new scale into account
        x_center = -(x1 + x2) / 2
        y_center = -(y1 + y2) / 2
        self._heatmap.local.x = x_center * self._heatmap.local.scale_x
        self._heatmap.local.y = y_center * self._heatmap.local.scale_y

        # Adjust the marginal dendrogram
        self.canvas_margin_x.local.x = self._heatmap.local.x
        self.canvas_margin_x.local.scale_x = self._heatmap.local.scale_x

        # Adjust the row marginal
        self.canvas_margin_y.local.y = self._heatmap.local.y
        self.canvas_margin_y.local.scale_y = self._heatmap.local.scale_y

        # self.camera_heatmap.show_rect(x1, x2, y1, y2, up=(0, 1, 0), view_dir=(0, 0, -1))

    def reset_view(self):
        """Reset the view of the heatmap."""
        # Get the (local!) extent of the heatmap
        self._heatmap.local.scale_x = 1
        self._heatmap.local.scale_y = 1
        self._heatmap.local.x = 0
        self._heatmap.local.y = 0

        # Do the same for the marginal dendrogram on the column
        self.canvas_margin_x.local.x = 0
        self.canvas_margin_x.local.scale_x = 1

        self.col_marginal.local.scale_x = 2 / (self._data.shape[1] * 10)
        self.col_marginal.local.scale_y = (
            2 / np.nanmax(self.col_marginal.geometry.positions.data[:, 1]) * 0.95
        )
        self.col_marginal.local.y = -1  # center
        self.col_marginal.local.x = -1  # center

        # Do it for the row marginal as well
        self.canvas_margin_y.local.y = 0
        self.canvas_margin_y.local.scale_y = 1
        self.row_marginal.local.scale_y = 2 / (self._data.shape[1] * 10)
        self.row_marginal.local.y = -1  # center

        # For the hover widget we need to scale the aspect ratio of the text
        ratio = self.renderer.logical_size[1] / self.renderer.logical_size[0]
        if ratio > 1:
            self._hover_widget.children[1].local.scale_x = .2
            self._hover_widget.children[1].local.scale_y = self._hover_widget.children[1].local.scale_x / ratio
        else:
            self._hover_widget.children[1].local.scale_y = .2
            self._hover_widget.children[1].local.scale_x = ratio * self._hover_widget.children[1].local.scale_y


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

    def make_hover_widget(self, color=(1, 1, 1, 0.75), font_color=(0, 0, 0, 1)):
        """Generate a widget for hover info."""
        widget = gfx.Group()
        widget.visible = False  # hide by default

        widget.add(
            gfx.Mesh(
                gfx.plane_geometry(2, 2),
                gfx.MeshBasicMaterial(color=color),
            )
        )
        widget.add(
            text2gfx("Hover info", color=font_color, font_size=1, anchor="middle-center")
        )

        # Because the NDC camera is (-1, 1) in both x and y, we need to scale the text
        widget.children[1].local.scale_y = len(self) / 100

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
                        anchor="center-right",
                    )
                )
                # Track where this label is supposed to show up (for scaling)
                self._text_group.children[-1]._absolute_position_x = -1.5
                self._text_group.children[-1]._absolute_position_y = t

        return group

    def make_col_dendrogram(self, method="ward"):
        """Generate the lines of the dendrogram."""
        if self.col_linkage is None:
            self.col_linkage = linkage(squareform(self._data.T), method=method)
        self.col_dendrogram = dendrogram(
            self.col_linkage, no_plot=True, above_threshold_color="w"
        )

        self._data = self._data[:, self.col_dendrogram["leaves"]]
        self._data = self._data[self.col_dendrogram["leaves"]]
        self._leafs_order = np.array(self.col_dendrogram["leaves"])

        lines = []

        # For colors: translate matplotlib C1-N into actual RGB colors
        cn_colors = [
            c for c in set(self.col_dendrogram["color_list"]) if str(c).startswith("C")
        ]
        palette = dict(
            zip(sorted(cn_colors), cmap.Colormap("tab10").iter_colors(len(cn_colors)))
        )

        # Now compile the lines
        colors = []
        for i in range(len(self.col_dendrogram["icoord"])):
            x = self.col_dendrogram["icoord"][i]
            y = self.col_dendrogram["dcoord"][i]
            for j in range(4):
                lines.append([x[j], y[j], 0])
                c = self.col_dendrogram["color_list"][i]
                if c in palette:
                    colors.append(palette[c])
                else:
                    colors.append(tuple(cmap.Color(c)))
            lines.append([None, None, None])

        return lines2gfx(
            np.array(lines),
            color=np.array(colors),
            linewidth=0.01,
            linewidth_space="world",
        )

    def make_row_marginal(self):
        """Generate row marginal.

        Currently this includes:
         - a marker for each row in the heatmap
        """
        group = gfx.Group()

        marker_pos = np.zeros((len(self), 3))
        marker_pos[:, 0] = 0.5  # move to the right (axis is -1 to 1 by default)
        marker_pos[:, 1] = (np.arange(len(self)) + 0.5) * self.x_spacing

        # For now we will work with uniform markers
        markers = np.array([None] * len(self))

        markers_vis = []
        for m in list(set(markers)):
            ix = markers == m
            markers_vis.append(
                points2gfx(
                    marker_pos[ix],
                    # color=colors[ix],
                    color="w",
                    size_space="world",
                    marker=m,
                    pick_write=True,
                    size=self.leaf_size,
                )
            )
            # This keeps track of the original indices of the leafs
            markers_vis[-1]._leaf_ix = np.where(ix)[0]

            # Move behind the hover visual
            markers_vis[-1].local.z = .1

            group.add(markers_vis[-1])

        return group, markers_vis

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


def heatmap(data):
    # Create the Figure
    fig = Heatmap(
        data,
    )

    return fig
