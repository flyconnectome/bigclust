"""
A transform gizmo to manipulate world objects.
"""

import numpy as np
import pylinalg as la
import pygfx as gfx

from pygfx.objects import WorldObject
from pygfx.utils.viewport import Viewport
from pygfx.utils.transform import AffineTransform

from ._visuals import lines2gfx


# IDEAS:
# - 3d selection gizmo (i.e. a box where we need to click twice)
# - lasso selection gizmo


class SelectionGizmo(WorldObject):
    """Gizmo to Draw a Selection Box.

    To invoke the Gizmo:

    * Shift-click on the canvas and start dragging to draw a selection rectangle

    Parameters
    ----------
    renderer : Renderer | Viewport
        The renderer or viewport to use for the gizmo.
    camera : Camera
        The camera to use for the gizmo.
    edge_color : str, optional
        The color to use for the edge of the selection box.
    fill_color : str, optional
        The color to use for the fill of the selection box.
    modifier : str
        The modifier key to use to activate the gizmo. Default "Shift".
    show_info : bool
        Whether to show a small box with additional infos on the selection box.

    """

    info_font_size = 3
    _fill_opacity = 0.3
    _outline_opacity = 0.7

    def __init__(
        self,
        renderer,
        camera,
        scene,
        edge_color="w",
        fill_color=None,
        modifier="Shift",
        line_width=1,
        line_style="dashed",
        force_square=False,
        show_info=False,
        debug=False,
        leave=False,
        callback_after=None,
        callback_during=None,
    ):
        assert modifier in ("Shift", "Ctrl", "Alt", None)

        super().__init__()

        # We store these as soon as we get a call in ``add_default_event_handlers``
        self._viewport = Viewport.from_viewport_or_renderer(renderer)
        self._camera = camera
        self._scene = scene
        self._ndc_to_screen = None
        self.add_default_event_handlers()  # this sets up the event handlers
        self._scene.add(self)

        # Init
        self._show_info = show_info
        self._line_style = line_style
        self._line_width = line_width
        self._modifier = modifier
        self._edge_color = edge_color
        self._fill_color = fill_color
        self._create_elements()
        self._force_square = force_square
        self.visible = False
        self._active = False
        self.debug = debug
        self._sel = {}
        self._leave = leave
        self._callback_after = callback_after
        self._callback_during = callback_during

    @property
    def bounds(self):
        """Return bounds based on selection."""
        if not self._sel:
            return None
        sel = np.vstack([self._sel["start"], self._sel["end"]])
        return np.vstack([sel.min(axis=0), sel.max(axis=0)])

    def _create_elements(self, clear=False):
        """Create selection elements."""
        # Generate fill
        if self._fill_color:
            self._fill = gfx.Mesh(
                gfx.Geometry(
                    positions=np.array(
                        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)],
                        dtype=np.float32,
                    ),
                    indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
                ),
                gfx.MeshBasicMaterial(color=self._fill_color),
            )
            self._fill.material.opacity = self._fill_opacity
            self.add(self._fill)
        else:
            self._fill = None

        # Generate outline
        if self._edge_color:
            self._outline = lines2gfx(
                np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                self._edge_color,
                dash_pattern=self._line_style,
                linewidth=self._line_width,
            )
            self._outline.material.opacity = self._outline_opacity
            self.add(self._outline)
        else:
            self._outline = None

        # Generate info text
        if self._show_info:
            self._show_info = (
                gfx.Text(
                    gfx.TextGeometry(
                        markdown="",
                        font_size=self.info_font_size,
                        anchor="bottom-right",
                    ),
                    gfx.TextMaterial(
                        color=self._edge_color if self._edge_color else self._fill_color
                    ),
                ),
                gfx.Text(
                    gfx.TextGeometry(
                        markdown="",
                        font_size=self.info_font_size,
                        anchor="top-left",
                    ),
                    gfx.TextMaterial(
                        color=self._edge_color if self._edge_color else self._fill_color
                    ),
                ),
            )
            self.add(self._show_info[0])
            self.add(self._show_info[1])

    def update_gizmo(self, event):
        if event.type != "before_render":
            return

        if self._viewport and self._camera and self._active:
            self._update_ndc_screen_transform()

    def _update_ndc_screen_transform(self):
        # Note: screen origin is at top left corner of NDC with Y-axis pointing down
        x_dim, y_dim = self._viewport.logical_size
        screen_space = AffineTransform()
        screen_space.position = (-1, 1, 0)
        screen_space.scale = (2 / x_dim, -2 / y_dim, 1)
        self._ndc_to_screen = screen_space.inverse_matrix
        self._screen_to_ndc = screen_space.matrix

    def add_default_event_handlers(self):
        """Register Gizmo callbacks."""
        self._viewport.renderer.add_event_handler(
            self.process_event,
            "pointer_down",
            "pointer_move",
            "pointer_up",
        )
        self._viewport.renderer.add_event_handler(self.update_gizmo, "before_render")

    def process_event(self, event):
        """Callback to handle gizmo-related events."""
        # Triage over event type
        has_mod = self._modifier is None or (self._modifier in event.modifiers)
        if event.type == "pointer_down" and has_mod:
            self._start_drag(event)
            self._viewport.renderer.request_draw()
            # self.set_pointer_capture(event.pointer_id, event.root)

        elif event.type == "pointer_up" and self._active:
            self._stop_drag(event)
            self._viewport.renderer.request_draw()

        elif event.type == "pointer_move" and self._active:
            self._move_selection(event)
            self._viewport.renderer.request_draw()

    def _start_drag(self, event):
        """Initialize the drag."""
        # Set the rectangle to visible
        self.visible = True
        self._active = True
        self._event_modifiers = event.modifiers

        # Set the positions of the selection rectangle
        world_pos = self._screen_to_world((event.x, event.y))

        if self._outline:
            self._outline.geometry.positions.data[:, 0] = world_pos[0]
            self._outline.geometry.positions.data[:, 1] = world_pos[1]
            self._outline.geometry.positions.update_range()
        if self._fill:
            self._fill.geometry.positions.data[:, 0] = world_pos[0]
            self._fill.geometry.positions.data[:, 1] = world_pos[1]
            self._fill.geometry.positions.update_range()

        # In debug mode we will add points
        if self.debug:
            print("Starting at ", world_pos)
            self.remove(*[c for c in self.children if isinstance(c, gfx.Points)])
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color="r", size=10),
            )
            self.add(point)

        # Store the selection box coordinates
        self._sel = {
            "start": world_pos,
            "end": world_pos,
        }

        # Update info text (if applicable)
        self._update_info()

    def _stop_drag(self, event):
        """Stop the drag on pointer up."""
        # Set the rectangle to invisible
        self._active = False
        if not self._leave:
            self.visible = False

        if self.debug:
            world_pos = self._screen_to_world((event.x, event.y))
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color="g", size=10),
            )
            self.add(point)
            print("Stopping with Selection box: ", self._sel)

        if self._callback_after:
            self._callback_after(self)

    def _move_selection(self, event):
        """Translate action, either using a translate1 or translate2 handle."""
        # Set the positions of the rectangle
        world_pos = self._screen_to_world((event.x, event.y))

        if self._force_square:
            dx, dy, dz = world_pos - self._sel["start"]
            dmin = min(abs(dx), abs(dy))
            world_pos[0] = self._sel["start"][0] + np.sign(dx) * dmin
            world_pos[1] = self._sel["start"][1] + np.sign(dy) * dmin

        if self._outline:
            # The first and the last point on the line remain on the origin
            # The second point goes to (origin, new_y), the third to (new_x, new_y)
            # The fourth to (new_x, origin)
            self._outline.geometry.positions.data[1, 1] = world_pos[1]
            self._outline.geometry.positions.data[2, 0] = world_pos[0]
            self._outline.geometry.positions.data[2, 1] = world_pos[1]
            self._outline.geometry.positions.data[3, 0] = world_pos[0]
            self._outline.geometry.positions.update_range()

        if self._fill:
            self._fill.geometry.positions.data[1, 1] = world_pos[1]
            self._fill.geometry.positions.data[2, 0] = world_pos[0]
            self._fill.geometry.positions.data[2, 1] = world_pos[1]
            self._fill.geometry.positions.data[3, 0] = world_pos[0]
            self._fill.geometry.positions.update_range()

        # Store the selection box coordinates
        self._sel["end"] = world_pos

        # Update info text (if applicable)
        self._update_info()

        if self.debug:
            print("Moving to ", world_pos)
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color="w", size=10),
            )
            self.add(point)

    def _update_info(self):
        """Update the info text."""
        if not self._show_info:
            return

        self._show_info[0].geometry.set_text(
            f"({self._sel['start'][0]:.2f}, {self._sel['start'][0]:.2f})"
        )
        self._show_info[1].geometry.set_text(
            f"({self._sel['end'][0]:.2f}, {self._sel['end'][0]:.2f})"
        )

        self._show_info[0].local.position = self._sel["start"]
        self._show_info[1].local.position = self._sel["end"]

    def _screen_to_world(self, pos):
        """Translate screen position to world coordinates."""
        if not self._viewport.is_inside(*pos):
            return None

        # Get position relative to viewport
        pos_rel = (
            pos[0] - self._viewport.rect[0],
            pos[1] - self._viewport.rect[1],
        )

        vs = self._viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        pos_ndc += la.vec_transform(
            self._camera.world.position, self._camera.camera_matrix
        )
        pos_world = la.vec_unproject(pos_ndc[:2], self._camera.camera_matrix)

        return pos_world
