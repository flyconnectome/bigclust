import pygfx as gfx
import numpy as np
import pylinalg as la

from wgpu.gui.auto import WgpuCanvas
from wgpu.gui.offscreen import WgpuCanvas as WgpuCanvasOffscreen

from . import utils


class Figure:
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
        # Check if we're running in an IPython environment
        if utils._type_of_script() == "ipython":
            ip = get_ipython()  # noqa: F821
            if not ip.active_eventloop:
                # ip.enable_gui('qt6')
                raise ValueError(
                    'IPython event loop not running. Please use e.g. "%gui qt" to hook into the event loop.'
                )

        # Update some defaults as necessary
        defaults = {"max_fps": max_fps, "size": size}
        defaults.update(kwargs)

        if not offscreen:
            self.canvas = WgpuCanvas(**defaults)
        else:
            self.canvas = WgpuCanvasOffscreen(**defaults)

        # There is a bug in pygfx 0.1.18 that causes the renderer to crash
        # when using a Jupyter canvas without explicitly setting the pixel_ratio.
        # This is already fixed in main but for now:
        if self._is_jupyter:
            self.renderer = gfx.renderers.WgpuRenderer(
                self.canvas, show_fps=False, pixel_ratio=2
            )
        else:
            self.renderer = gfx.renderers.WgpuRenderer(self.canvas, show_fps=False)

        # Set up a default scene
        self.scene = gfx.Scene()
        # self.scene.add(gfx.AmbientLight(intensity=1))

        # Set up a default background
        self._background = gfx.BackgroundMaterial((0, 0, 0))
        self.scene.add(gfx.Background(None, self._background))

        # Add camera
        self.camera = gfx.OrthographicCamera()

        # Add controller
        self.controller = gfx.PanZoomController(
            self.camera, register_events=self.renderer
        )

        # Stats
        self.stats = gfx.Stats(self.renderer)
        self._show_fps = False

        # Setup key events
        self.key_events = {}
        self.key_events["f"] = lambda: self._toggle_fps()

        def _keydown(event):
            """Handle key presses."""
            if event.key in self.key_events:
                self.key_events[event.key]()

        # Register events
        self.renderer.add_event_handler(_keydown, "key_down")

        # Finally, setting some variables
        self._animations = []

        # This starts the animation loop
        if show and not self._is_jupyter:
            self.show()


    def _animate(self):
        """Animate the scene."""
        to_remove = []
        for i, func in enumerate(self._animations):
            try:
                func()
            except BaseException as e:
                print(f"Removing animation function {func} because of error: {e}")
                to_remove.append(i)
        for i in to_remove[::-1]:
            self.remove_animation(i)

        if self._show_fps:
            with self.stats:
                self.renderer.render(self.scene, self.camera, flush=False)
            self.stats.render()
        else:
            self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()

    @property
    def size(self):
        """Return size of the canvas."""
        return self.canvas.get_logical_size()

    @size.setter
    def size(self, size):
        """Set size of the canvas."""
        assert len(size) == 2
        self.canvas.set_logical_size(*size)

    @property
    def max_fps(self):
        """Maximum frames per second to render."""
        return self.canvas._subwidget._max_fps

    @max_fps.setter
    def max_fps(self, v):
        assert isinstance(v, int)
        self.canvas._subwidget._max_fps = v

    @property
    def _is_jupyter(self):
        """Check if Viewer is using Jupyter canvas."""
        return "JupyterWgpuCanvas" in str(type(self.canvas))

    @property
    def _is_offscreen(self):
        """Check if Viewer is using offscreen canvas."""
        return isinstance(self.canvas, WgpuCanvasOffscreen)

    @property
    def x_scale(self):
        return self.scene.local.matrix[0, 0]

    @x_scale.setter
    def x_scale(self, x):
        assert x > 0
        mat = np.copy(self.scene.local.matrix)
        mat[0, 0] = x
        self.scene.local.matrix = mat

    @property
    def y_scale(self):
        return self.scene.local.matrix[1, 1]

    @y_scale.setter
    def y_scale(self, y):
        assert y > 0
        mat = np.copy(self.scene.local.matrix)
        mat[1, 1] = y
        self.scene.local.matrix = mat

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
        if isinstance(self.canvas, WgpuCanvasOffscreen):
            return
        # In terminal we can just show the window
        elif not self._is_jupyter:
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

        # This starts the animation loop
        if show and not self._is_jupyter:
            self.show()

    def _animate(self):
        """Animate the scene."""
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

        self.canvas.request_draw()

    @property
    def x_scale(self):
        """The x-scale of the scene."""
        return self.scene.local.matrix[0, 0]

    @x_scale.setter
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
    def y_scale(self, y):
        assert y > 0
        mat = np.copy(self.scene.local.matrix)
        mat[1, 1] = y
        self.scene.local.matrix = mat

    def center_camera(self):
        """Center camera on visuals."""
        if len(self.scene.children):
            self.camera.show_object(
                self.scene, scale=1, view_dir=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0)
            )

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

    def show_message(
        self, message, position="top-right", font_size=20, color=None, duration=None
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
            self._message_text = _visuals.text2gfx(
                message, color="white", font_size=font_size, screen_space=True
            )

        # Make sure the text is in the scene
        if self._message_text not in self.overlay_scene.children:
            self.overlay_scene.add(self._message_text)

        self._message_text.geometry.set_text(message)
        self._message_text.geometry.font_size = font_size
        self._message_text.geometry.anchor = position
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
                        self._message_text.material.opacity = max(self._message_text.material.opacity - 0.02, 0)

                    if self._message_text.material.opacity <= 0:
                        if self._message_text.parent:
                            self.overlay_scene.remove(self._message_text)
                        self.remove_animation(_fade_message)

            self.add_animation(_fade_message)