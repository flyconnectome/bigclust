import json

from pathlib import Path

from octarine import Viewer
from PySide6.QtWidgets import QWidget

from .dendrogram import Dendrogram
from .scatter import ScatterPlot
from .figure import Figure
from .neuroglancer import NglViewer


class WindowStateManager:
    """Class that saves/loads window states (location & size)."""

    def __init__(self, config_file, shortcut="S", modifiers=("Shift",)):
        self.config_file = config_file
        self.figures = {}
        self.shortcut = shortcut
        self.modifiers = modifiers

    @property
    def config_file(self):
        return self._config_file

    @config_file.setter
    def config_file(self, value):
        path = Path(value).expanduser()

        # Generate parent directory if not exists
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        # Generate file if it doesn't exist
        if not path.exists():
            path.touch()

        self._config_file = path

    @property
    def state(self):
        """Return the current state of the figures."""
        state = {}

        for name, fig in self.figures.items():
            if isinstance(fig, (Figure, Dendrogram, Viewer)):
                window = fig.canvas
            elif isinstance(fig, NglViewer):
                window = fig.viewer.canvas
            elif isinstance(fig, QWidget):
                window = fig

            screen = window.screen()

            state[name] = {
                "x": window.x() / screen.size().width(),
                "y": window.y() / screen.size().height(),
                "width": window.width() / screen.size().width(),
                "height": window.height() / screen.size().height(),
            }

        return state

    @property
    def stored_state(self):
        with open(self.config_file, "r") as f:
            try:
                state = json.load(f)
            except json.JSONDecodeError:
                state = {}
        return state

    @property
    def has_stored_state(self):
        """Whether the config file has stored state."""
        return True if self.stored_state else False

    def add_figures(self, figure, id=None):
        """Add figure to be managed."""
        # Register the keyboard shortcut to save the state
        if isinstance(figure, (Figure, Dendrogram)):
            figure.key_events[(self.shortcut, self.modifiers)] = self.save_state
            name = figure.canvas.windowTitle()
        elif isinstance(figure, NglViewer):
            figure.viewer._key_events[(self.shortcut, self.modifiers)] = self.save_state
            name = figure.viewer.canvas.windowTitle()
        elif isinstance(figure, Viewer):
            figure._key_events[(self.shortcut, self.modifiers)] = self.save_state
            name = figure.canvas.windowTitle()
        elif isinstance(figure, QWidget):
            name = figure.windowTitle()
        else:
            raise TypeError(
                f"Expected Figure, Dendrogram, or NglViewer, got {type(figure)}."
            )

        if id is None:
            id = name

        if id in self.figures:
            raise ValueError(
                f"Figure with ID/title '{id}' already exists."
            )

        self.figures[id] = figure

    def save_state(self):
        """Save the state of all figures to the config file."""
        #Get the currently stored state and update with current state (so that we don't overwrite anything)
        state = self.stored_state
        state.update(self.state)
        with open(self.config_file, "w") as f:
            json.dump(state, f)

        print(f"State for {len(state)} figures saved to {self.config_file}.")

        for fig in self.figures.values():
            if isinstance(fig, (Figure, Dendrogram)):
                fig.show_message("State(s) saved.", duration=3, color="g")
            elif isinstance(fig, NglViewer):
                fig.viewer.show_message("State(s) saved.", duration=3, color="g")

    def restore_state(self):
        state = self.stored_state

        for name, fig in self.figures.items():
            if name not in state:
                print(f"Figure {name} not found in state.")
                continue

            if isinstance(fig, (Figure, Dendrogram, Viewer)):
                window = fig.canvas
            elif isinstance(fig, NglViewer):
                window = fig.viewer.canvas
            elif isinstance(fig, QWidget):
                window = fig

            screen = window.screen()

            window.move(
                state[name]["x"] * screen.size().width(),
                state[name]["y"] * screen.size().height(),
            )
            window.resize(
                state[name]["width"] * screen.size().width(),
                state[name]["height"] * screen.size().height(),
            )

        print(f"State restored from {self.config_file}.")
