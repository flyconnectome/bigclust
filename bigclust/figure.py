import cmap
import time

import pygfx as gfx
import numpy as np
import pylinalg as la

from abc import abstractmethod
from functools import wraps
from rendercanvas.auto import RenderCanvas
from rendercanvas.offscreen import OffscreenRenderCanvas

from . import utils, visuals


def update_figure(func):
    """Decorator to update figure."""

    @wraps(func)
    def inner(*args, **kwargs):
        val = func(*args, **kwargs)
        figure = args[0]

        # Any time we update the viewer, we should set it to stale
        figure._render_stale = True
        figure.canvas.request_draw()

        return val

    return inner


class StateWgpuCanvas(RenderCanvas):
    """WgpuCanvas that emits signals to its parent figure when its moved/resized."""

    def __init__(self, figure, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._figure = figure

    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self._figure, "_on_canvas_state_change"):
            self._figure._on_canvas_state_changed(event="move")

    def resizeEvent(self, event):
        self._save_state()
        super().resizeEvent(event)
        if hasattr(self._figure, "_on_canvas_state_change"):
            self._figure._on_canvas_state_changed(event="resize")


class BaseFigure:
    """Figure.

    Parameters
    ----------
    max_fps :   int, optional
                Maximum frames per second to render.
    size :      tuple, optional
                Size of the viewer window.
    show :      bool, optional
                Whether to immediately show the viewer. Note that this has no
                effect in Jupyter. There you will have to call ``.show()`` manually
                on the last line of a cell for the viewer to appear.
    **kwargs
                Keyword arguments are passed through to ``WgpuCanvas``.

    """

    # Palette used for assigning colors to objects
    palette = "seaborn:tab10"

    def __init__(
        self,
        offscreen=False,
        max_fps=30,
        size=None,
        **kwargs,
    ):
        # Check if we're running in an IPython environment
        if utils._type_of_script() == "ipython":
            ip = get_ipython()  # noqa: F821
            if not ip.active_eventloop:
                # ip.enable_gui('qt6')
                raise ValueError(
                    'IPython event loop not running. Please use e.g. "%gui qt" to hook into the event loop.'
                )

        # Update some defaults as necessary
        defaults = {"max_fps": max_fps, "size": size, "update_mode": "continuous"}
        defaults.update(kwargs)

        if not offscreen:
            self.canvas = RenderCanvas(**defaults)
        else:
            self.canvas = OffscreenRenderCanvas(**defaults)

        # There is a bug in pygfx 0.1.18 that causes the renderer to crash
        # when using a Jupyter canvas without explicitly setting the pixel_ratio.
        # This is already fixed in main but for now:
        if self._is_jupyter:
            self.renderer = gfx.renderers.WgpuRenderer(
                self.canvas, show_fps=False, pixel_ratio=2
            )
        else:
            self.renderer = gfx.renderers.WgpuRenderer(self.canvas, show_fps=False)

        # Set up a default background
        self._background = gfx.BackgroundMaterial((0, 0, 0))

        # Stats
        self.stats = gfx.Stats(self.renderer)
        self._show_fps = False

        # Setup key events
        self.key_events = {}
        self.key_events["f"] = lambda: self._toggle_fps()
        self.key_events["c"] = lambda: self._toggle_controls()

        def _keydown(event):
            """Handle key presses."""
            if not event.modifiers:
                if event.key in self.key_events:
                    self.key_events[event.key]()
            else:
                tup = (event.key, tuple(event.modifiers))
                if tup in self.key_events:
                    self.key_events[tup]()

        # Register events
        self.renderer.add_event_handler(_keydown, "key_down")

        # Finally, setting some variables
        self._animations = []

    @abstractmethod
    def _animate(self):
        """Animate the scene."""
        pass

    def _run_user_animations(self):
        """Run user-defined animations."""
        to_remove = []
        for i, func in enumerate(self._animations):
            try:
                func()
            except BaseException as e:
                print(f"Removing animation function {func} because of error: {e}")
                to_remove.append(i)
        for i in to_remove[::-1]:
            self.remove_animation(i)

    @property
    def size(self):
        """Return size of the canvas."""
        return self.canvas.get_logical_size()

    @size.setter
    @update_figure
    def size(self, size):
        """Set size of the canvas."""
        assert len(size) == 2
        self.canvas.set_logical_size(*size)

    @property
    def max_fps(self):
        """Maximum frames per second to render."""
        return self.canvas._subwidget._BaseRenderCanvas__scheduler._max_fps

    @max_fps.setter
    def max_fps(self, v):
        assert isinstance(v, int)
        self.canvas._subwidget._BaseRenderCanvas__scheduler._max_fps = v

    @property
    def _is_jupyter(self):
        """Check if Viewer is using Jupyter canvas."""
        return "JupyterWgpuCanvas" in str(type(self.canvas))

    @property
    def _is_offscreen(self):
        """Check if Viewer is using offscreen canvas."""
        return isinstance(self.canvas, OffscreenRenderCanvas)

    @update_figure
    def add_animation(self, x):
        """Add animation function to the Viewer.

        Parameters
        ----------
        x :     callable
                Function to add to the animation loop.

        """
        if not callable(x):
            raise TypeError(f"Expected callable, got {type(x)}")

        self._animations.append(x)

    @update_figure
    def remove_animation(self, x):
        """Remove animation function from the Viewer.

        Parameters
        ----------
        x :     callable | int
                Either the function itself or its index
                in the list of animations.

        """
        if callable(x):
            self._animations.remove(x)
        elif isinstance(x, int):
            self._animations.pop(x)
        else:
            raise TypeError(f"Expected callable or index (int), got {type(x)}")

    @update_figure
    def show(self, use_sidecar=False, toolbar=False):
        """Show viewer.

        Parameters
        ----------

        For Jupyter lab only:

        use_sidecar : bool
                      If True, will use the Sidecar extension to display the
                      viewer outside the notebooks. Will throw an error if
                      Sidecar is not installed.
        toolbar :     bool
                      If True, will show a toolbar. You can always show/hide
                      the toolbar with ``viewer.show_controls()`` and
                      ``viewer.hide_controls()``, or the `c` hotkey.

        """
        # Start the animation loop
        self.canvas.request_draw(self._animate)

        # If this is an offscreen canvas, we don't need to show anything
        if isinstance(self.canvas, OffscreenRenderCanvas):
            return
        # In terminal we can just show the window
        elif not self._is_jupyter:
            if hasattr(self.canvas, "show"):
                self.canvas.show()
        # For Jupyter we need to wrap the canvas in a widget
        else:
            # if not hasattr(self, 'widget'):
            from .jupyter import JupyterOutput

            # Construct the widget
            self.widget = JupyterOutput(
                self,
                use_sidecar=use_sidecar,
                toolbar=toolbar,
                sidecar_kwargs={"title": self._title},
            )
            return self.widget

    def close(self):
        """Close the viewer."""
        # Close if not already closed
        if not self.canvas.is_closed():
            self.canvas.close()

    @update_figure
    def set_bgcolor(self, c):
        """Set background color.

        Parameters
        ----------
        c :     tuple | str
                RGB(A) color to use for the background.

        """
        self._background.set_colors(gfx.Color(c).rgba)

    def show_controls(self):
        """Show controls."""
        raise NotImplementedError("Controls not implemented for this viewer.")

    def hide_controls(self):
        """Hide controls."""
        if self._is_jupyter:
            if self.widget.toolbar:
                self.widget.toolbar.hide()
        else:
            if hasattr(self, "_controls"):
                self._controls.hide()

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

    def _toggle_fps(self):
        """Switch FPS measurement on and off."""
        self._show_fps = not self._show_fps


class Figure(BaseFigure):
    """Figure.

    Parameters
    ----------
    max_fps :   int, optional
                Maximum frames per second to render.
    size :      tuple, optional
                Size of the viewer window.
    show :      bool, optional
                Whether to immediately show the viewer. Note that this has no
                effect in Jupyter. There you will have to call ``.show()`` manually
                on the last line of a cell for the viewer to appear.
    **kwargs
                Keyword arguments are passed through to ``WgpuCanvas``.

    """

    # Palette used for assigning colors to objects
    palette = "seaborn:tab10"

    def __init__(
        self,
        offscreen=False,
        max_fps=30,
        size=None,
        show=True,
        **kwargs,
    ):
        super().__init__(offscreen=offscreen, max_fps=max_fps, size=size, **kwargs)

        # Setup the scene
        self._setup_scene()

        # Set the render trigger
        self._render_trigger = "continuous"

        # This starts the animation loop
        if show and not self._is_jupyter:
            self.show()

    def _setup_scene(self):
        # Set up a default scene
        self.scene = gfx.Scene()
        # self.scene.add(gfx.AmbientLight(intensity=1))

        # Add the background (from BaseFigure) to the scene
        self.scene.add(gfx.Background(None, self._background))

        # Add a camera
        self.camera = gfx.OrthographicCamera()

        # Setup overlay
        self.overlay_camera = gfx.NDCCamera()
        self.overlay_scene = gfx.Scene()

        # Add a controller
        self.controller = gfx.PanZoomController(
            self.camera, register_events=self.renderer
        )

    def _animate(self):
        """Run the render loop."""
        rm = self.render_trigger

        if rm == "active_window":
            # Note to self: we need to explore how to do this with different backends / Window managers
            if hasattr(self.canvas, "isActiveWindow"):
                if not self.canvas.isActiveWindow():
                    return
        elif rm == "reactive":
            # If the scene is not stale, we can skip rendering
            if not getattr(self, "_render_stale", False):
                return

        self._run_user_animations()

        # Now render the scene
        if self._show_fps:
            with self.stats:
                self.renderer.render(self.scene, self.camera, flush=False)
                self.renderer.render(
                    self.overlay_scene, self.overlay_camera, flush=False
                )
            self.stats.render()
        else:
            self.renderer.render(self.scene, self.camera, flush=False)
            self.renderer.render(self.overlay_scene, self.overlay_camera)

        # Set stale to False
        self._render_stale = False

        self.canvas.request_draw()

    @property
    def render_trigger(self):
        """Determines when the scene is (re)rendered.

        By default, we leave it to the renderer to decide when to render the scene.
        You can adjust that behaviour by setting render mode to:
         - "continuous" (default): leave it to the renderer to decide when to render the scene
         - "active_window": rendering is only done when the window is active
         - "reactive": rendering is only triggered when the scene changes

        """
        return self._render_trigger

    @render_trigger.setter
    def render_trigger(self, mode):
        valid = ("continuous", "active_window", "reactive")
        if mode not in valid:
            raise ValueError(f"Unknown render mode: {mode}. Must be one of {valid}.")

        # No need to do anything if the value is the same
        if mode == getattr(self, "_render_trigger", None):
            return

        # Add/remove event handlers as necessary
        if mode == "reactive":
            self._set_stale_func = lambda event: setattr(self, "_render_stale", True)
            self.renderer.add_event_handler(
                self._set_stale_func,
                "pointer_down",
                "pointer_move",
                "pointer_up",
                "wheel",
                # "before_render",
            )
        elif self._render_trigger == "reactive":
            self.renderer.remove_event_handler(
                self._set_stale_func,
                "pointer_down",
                "pointer_move",
                "pointer_up",
                "wheel",
                # "before_render",
            )

        self._render_trigger = mode

    @property
    def x_scale(self):
        """The x-scale of the scene."""
        return self.scene.local.matrix[0, 0]

    @x_scale.setter
    @update_figure
    def x_scale(self, x):
        assert x > 0
        mat = np.copy(self.scene.local.matrix)
        mat[0, 0] = x
        self.scene.local.matrix = mat

    @property
    def y_scale(self):
        """The y-scale of the scene."""
        return self.scene.local.matrix[1, 1]

    @y_scale.setter
    @update_figure
    def y_scale(self, y):
        assert y > 0
        mat = np.copy(self.scene.local.matrix)
        mat[1, 1] = y
        self.scene.local.matrix = mat

    @update_figure
    def center_camera(self):
        """Center camera on visuals."""
        if len(self.scene.children):
            self.camera.show_object(
                self.scene, scale=1, view_dir=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0)
            )

    @update_figure
    def clear(self):
        """Clear canvas of objects (expects lights and background)."""
        # Remove everything but the lights and backgrounds
        self.scene.clear()

    def close(self):
        """Close the viewer."""
        # Clear first to free all visuals
        self.clear()

        super().close()

    def is_visible(self, obj, offset=0):
        """Test if object is visible to camera."""
        top_left = self._screen_to_world((0, 0))
        bottom_right = self._screen_to_world(self.size)

        if not isinstance(obj, (list, tuple)):
            obj = [obj]

        is_visible = np.ones(len(obj), dtype=bool)

        for i, o in enumerate(obj):
            if hasattr(o, "get_world_bounding_box"):
                bbox = o.get_world_bounding_box()
                if top_left[0] > bbox[1, 0]:
                    is_visible[i] = False
                elif top_left[1] < bbox[1, 1]:
                    is_visible[i] = False
                elif bottom_right[0] < bbox[0, 0]:
                    is_visible[i] = False
                elif bottom_right[1] > bbox[0, 1]:
                    is_visible[i] = False
            else:
                pos = o.local.position
                if top_left[0] > pos[0]:
                    is_visible[i] = False
                elif top_left[1] < pos[1]:
                    is_visible[i] = False
                elif bottom_right[0] < pos[0]:
                    is_visible[i] = False
                elif bottom_right[1] > pos[1]:
                    is_visible[i] = False

        return is_visible

    def is_visible_pos(self, pos, offset=0):
        """Test if positions are visible to camera."""
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 2
        assert pos.shape[1] == 2

        top_left = self._screen_to_world((0, 0))
        bottom_right = self._screen_to_world(self.size)

        is_visible = np.ones(len(pos), dtype=bool)

        is_visible[pos[:, 0] < top_left[0]] = False
        is_visible[pos[:, 1] > top_left[1]] = False
        is_visible[pos[:, 0] > bottom_right[0]] = False
        is_visible[pos[:, 1] < bottom_right[1]] = False

        return is_visible

    def _screen_to_world(self, pos):
        """Translate screen position to world coordinates."""
        viewport = gfx.Viewport.from_viewport_or_renderer(self.renderer)
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

        pos_ndc += la.vec_transform(
            self.camera.world.position, self.camera.camera_matrix
        )
        pos_world = la.vec_unproject(pos_ndc[:2], self.camera.camera_matrix)

        return pos_world

    @update_figure
    def show_message(
        self, message, position="top-right", font_size=20, color="w", duration=None
    ):
        """Show message on canvas.

        Parameters
        ----------
        message :   str | None
                    Message to show. Set to `None` to remove the existing message.
        position :  "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center"
                    Position of the message on the canvas.
        font_size : int, optional
                    Font size of the message.
        color :     str | tuple, optional
                    Color of the message. If `None`, will use white.
        duration :  int, optional
                    Number of seconds after which to fade the message.

        """
        if message is None and hasattr(self, "_message_text"):
            if self._message_text.parent:
                self.overlay_scene.remove(self._message_text)
            del self._message_text
            return

        _positions = {
            "top-left": (-0.95, 0.95, 0),
            "top-right": (0.95, 0.95, 0),
            "bottom-left": (-0.95, -0.95, 0),
            "bottom-right": (0.95, -0.95, 0),
            "center": (0, 0, 0),
        }
        if position not in _positions:
            raise ValueError(f"Unknown position: {position}")

        if not hasattr(self, "_message_text"):
            self._message_text = visuals.text2gfx(
                message, color="white", font_size=font_size, screen_space=True
            )

        # Make sure the text is in the scene
        if self._message_text not in self.overlay_scene.children:
            self.overlay_scene.add(self._message_text)

        self._message_text.set_text(message)
        self._message_text.font_size = font_size
        self._message_text.anchor = position
        if color is not None:
            self._message_text.material.color = cmap.Color(color).rgba
        self._message_text.material.opacity = 1
        self._message_text.local.position = _positions[position]

        # When do we need to start fading out?
        if duration:
            self._fade_out_time = time.time() + duration

            def _fade_message():
                if not hasattr(self, "_message_text"):
                    self.remove_animation(_fade_message)
                else:
                    if time.time() > self._fade_out_time:
                        # This means the text will fade fade over 1/0.02 = 50 frames
                        self._message_text.material.opacity = max(
                            self._message_text.material.opacity - 0.02, 0
                        )

                    if self._message_text.material.opacity <= 0:
                        if self._message_text.parent:
                            self.overlay_scene.remove(self._message_text)
                        self.remove_animation(_fade_message)

            self.add_animation(_fade_message)

    def restrict_selection(self, restrict_to):
        """Restrict selection to objects from a given class/origin/dataset.

        Parameters
        ----------
        restrict_to :   str | iterable of "strings" | None
                        Name of the object to restrict selection to. As a special
                        case this also accepts "No restrictions" to remove any restrictions.
                        If `None`, will also remove any restrictions.

        """
        if isinstance(restrict_to, str):
            if restrict_to == "No restrictions":
                restrict_to = None
            else:
                restrict_to = [restrict_to]

        if not restrict_to:
            if hasattr(self, "_restrict_selection"):
                del self._restrict_selection
            return

        self._restrict_selection = restrict_to
