import uuid
import cmap

import pandas as pd
import numpy as np
import pygfx as gfx


def text2gfx(
    text,
    position=(0, 0, 0),
    color="w",
    font_size=1,
    anchor="topright",
    screen_space=False,
    markdown=False,
    pickable=False
):
    """Convert text to pygfx visuals.

    Parameters
    ----------
    text :          str
                    Text to plot.
    position :      tuple
                    Position of the text.
    color :         tuple | str
                    Color to use for plotting.
    font_size :     int, optional
                    Font size.
    anchor :        str, optional
                    Anchor point of the text. Can be one of "topleft", "topright",
                    "bottomleft", "bottomright", "center", "topmiddle", "bottommiddle", "middleleft",
                    "middleright".
    screen_space :  bool, optional
                    Whether to use screen space coordinates.
    markdown :      bool, optional
                    Whether the text should be interpreted as markdown.

    Returns
    -------
    text :          gfx.Text
                    Pygfx visual for text.
    """
    assert isinstance(text, str), "Expected string."
    assert isinstance(position, (list, tuple, np.ndarray)), "Expected list or tuple."
    assert len(position) == 3, "Expected (x, y, z) position."

    defaults = {
        "font_size": font_size,
        "anchor": anchor,
        "screen_space": screen_space,
    }
    if markdown:
        defaults["markdown"] = text
    else:
        defaults["text"] = text

    vis = gfx.Text(
        gfx.TextGeometry(**defaults),
        gfx.TextMaterial(color=color, pick_write=pickable),
    )
    vis.local.position = position
    vis.geometry._text = text  # track the actual text
    return vis


def points2gfx(
    points, color, size=2, marker=None, size_space="screen", pick_write=False
):
    """Convert points to pygfx visuals.

    Parameters
    ----------
    points :        (N, 3) array
                    Points to plot.
    color :         tuple | array
                    Color to use for plotting. If multiple colors,
                    must be a list of colors with the same length as
                    the number of points.
    size :          int, optional
                    Marker size.
    marker :        str, optional
                    See gfx.MarkerShape for available markers.
    size_space :    "screen" | "world" | "model", optional
                    Units to use for the marker size. "screen" (default)
                    will keep the line width constant on the screen, while
                    "world" and "model" will keep it constant in world and
                    model coordinates, respectively.

    Returns
    -------
    vis :           gfx.Points
                    Pygfx visual for points.

    """
    # TODOs:
    # - add support for per-vertex sizes and colors
    assert isinstance(points, np.ndarray), "Expected numpy array."
    assert points.ndim == 2, "Expected 2D numpy array."
    assert points.shape[1] == 3, "Expected (N, 3) array."

    points = points.astype(np.float32, copy=False)

    # Make sure coordinates are c-contiguous
    if not points.flags["C_CONTIGUOUS"]:
        points = np.ascontiguousarray(points)

    geometry_kwargs = {}
    material_kwargs = {}
    material_kwargs["pick_write"] = pick_write

    # Parse sizes
    if isinstance(size, (list, np.ndarray)):
        if len(size) != len(points):
            raise ValueError(
                "Expected `size` to be a single value or "
                "an array of the same length as `points`."
            )
        geometry_kwargs["sizes"] = np.asarray(size).astype(np.float32, copy=False)
        material_kwargs["size_mode"] = "vertex"
    else:
        material_kwargs["size"] = size

    # Parse color(s)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        # If colors are provided for each node we have to make sure
        # that we also include `None` for the breaks in the segments
        n_points = len(points)
        if len(color) != n_points:
            raise ValueError(f"Got {len(color)} colors for {n_points} points.")
        color = color.astype(np.float32, copy=False)
        geometry_kwargs["colors"] = color
        material_kwargs["color_mode"] = "vertex"
    else:
        if isinstance(color, np.ndarray):
            color = color.astype(np.float32, copy=False)
        material_kwargs["color"] = color

    if marker is None:
        material = gfx.PointsMaterial(size_space=size_space, **material_kwargs)
    else:
        material = gfx.PointsMarkerMaterial(
            marker=marker, size_space=size_space, **material_kwargs, edge_width=0
        )

    vis = gfx.Points(gfx.Geometry(positions=points, **geometry_kwargs), material)

    # Add custom attributes
    vis._object_type = "points"
    vis._object_id = uuid.uuid4()

    return vis


def lines2gfx(lines, color, linewidth=1, linewidth_space="screen", dash_pattern=None):
    """Convert lines into pygfx visuals.

    Parameters
    ----------
    lines :     list of (N, 3) arrays | (N, 3) array
                Lines to plot. If a list of arrays, each array
                represents a separate line. If a single array,
                each row represents a point in the line. You can
                introduce breaks in the line by inserting NaNs.
    color :     str | tuple, optional
                Color to use for plotting. Can be a single color
                or one for every point in the line(s).
    linewidth : float, optional
                Line width.
    linewidth_space : "screen" | "world" | "model", optional
                Units to use for the line width. "screen" (default)
                will keep the line width constant on the screen, while
                "world" and "model" will keep it constant in world and
                model coordinates, respectively.
    dash_pattern : "solid" | "dashed" | "dotted" | "dashdot" | tuple, optional
                Line style to use. If a tuple, must define the on/off
                sequence.

    Returns
    -------
    vis :           gfx.Line
                    Pygfx visuals for lines.

    """
    if isinstance(lines, np.ndarray):
        assert lines.ndim == 2
        assert lines.shape[1] in (2, 3)
        assert len(lines) > 1

        if lines.shape[1] == 2:
            lines = np.column_stack([lines, np.zeros(len(lines))])
    elif isinstance(lines, list):
        assert all([isinstance(l, np.ndarray) for l in lines])
        assert all([l.ndim == 2 for l in lines])
        assert all([l.shape[1] in (2, 3) for l in lines])
        assert all([len(l) > 1 for l in lines])

        for i, l in enumerate(lines):
            if l.shape[1] == 2:
                lines[i] = np.column_stack([l, np.zeros(len(l))])

        # Convert to the (N, 3) format
        if len(lines) == 1:
            lines = lines[0]
        else:
            lines = np.insert(
                np.vstack(lines),
                np.cumsum([len(l) for l in lines[:-1]]),
                np.nan,
                axis=0,
            )
    else:
        raise TypeError("Expected numpy array or list of numpy arrays.")

    # At this point we expect to have a (N, 3) array
    lines = lines.astype(np.float32, copy=False)

    if dash_pattern is None:
        dash_pattern = ()  # pygfx expects an empty tuple for solid lines
    elif isinstance(dash_pattern, str):
        if dash_pattern == "solid":
            dash_pattern = ()
        elif dash_pattern == "dashed":
            dash_pattern = (5, 2)
        elif dash_pattern == "dotted":
            dash_pattern = (1, 2)
        elif dash_pattern == "dashdot":
            dash_pattern = (5, 2, 1, 2)
        else:
            raise ValueError(f"Unknown dash pattern: {dash_pattern}")

    geometry_kwargs = {}
    material_kwargs = {}

    # Parse color(s)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        # If colors are provided for each node we have to make sure
        # that we also include `None` for the breaks in the segments
        n_points = (~np.isnan(lines[:, 0])).sum()
        if n_points != len(lines):
            # See if we can rescue this
            if len(color) == n_points:
                breaks = np.where(np.isnan(lines[:, 0]))[0]
                offset = np.arange(len(breaks))
                color = np.insert(color, breaks-offset, np.nan, axis=0)
            else:
                raise ValueError(f"Got {len(color)} colors for {n_points} line points.")
        color = color.astype(np.float32, copy=False)
        geometry_kwargs["colors"] = color
        material_kwargs["color_mode"] = "vertex"
    else:
        if isinstance(color, np.ndarray):
            color = color.astype(np.float32, copy=False)
        material_kwargs["color"] = color

    vis = gfx.Line(
        gfx.Geometry(positions=lines, **geometry_kwargs),
        gfx.LineMaterial(
            thickness=linewidth,
            thickness_space=linewidth_space,
            dash_pattern=dash_pattern,
            **material_kwargs,
        ),
    )

    # Add custom attributes
    vis._object_type = "lines"
    vis._object_id = uuid.uuid4()

    return vis


def heatmap2gfx(data, width=1, height=1, clim="data", colormap="viridis"):
    """Convert heatmap data to pygfx visuals.

    Parameters
    ----------
    data :          (N, M) array | pd.DataFrame
                    Data to plot.
    clim :          tuple | str, optional
                    Color limits. If "data", use the min/max of the data.
                    If "datatype", use the min/max of the data type.
    colormap :      str, optional
                    Colormap to use.

    Returns
    -------
    vis :           gfx.Mesh
                    Pygfx visuals for heatmap.

    """
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Find the colormap
    colormap = cmap.Colormap(colormap)

    # Find the potential min/max value of the volume
    if isinstance(clim, str) and clim == "datatype":
        cmin = cmax = "datatype"
    elif isinstance(clim, str) and clim == "data":
        cmin = cmax = "data"
    else:
        cmin, cmax = clim

    if cmin == "datatype":
        cmin = 0
    elif cmin == "data":
        cmin = data.min()

    if cmax == "datatype":
        # If float, assume that the data is normalized
        if data.dtype.kind == "f":
            cmax = 1
        # Otherwise, use the maximum value of the data type
        else:
            cmax = np.iinfo(data.dtype).max
    elif cmax == "data":
        cmax = data.max()

    # Normalize the data
    data = (data - cmin) / (cmax - cmin)

    # Transate values into colors using the colormap
    colors = colormap(data.flatten())

    # We need two colors for each square (top left and bottom right triangle)
    face_colors = np.zeros((data.size * 2, 3), dtype=np.float32)
    face_colors[::2] = face_colors[1::2] = colors[:, :3]

    # Generate a plane with the appropriate number of cells
    p = gfx.plane_geometry(
        width, height, width_segments=data.shape[1], height_segments=data.shape[0]
    )
    vis = gfx.Mesh(
        gfx.Geometry(
            positions=p.positions.data, indices=p.indices.data, colors=face_colors
        ),
        gfx.MeshBasicMaterial(color_mode="face"),
    )

    return vis