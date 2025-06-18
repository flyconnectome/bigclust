import cmap
import colorsys

import numpy as np


def _type_of_script() -> str:
    """Return context (terminal, jupyter, colab, iPython) in which navis is run."""
    try:
        ipy_str = str(type(get_ipython()))  # noqa: F821
        if "zmqshell" in ipy_str:
            return "jupyter"
        elif "colab" in ipy_str:
            return "colab"
        else:  # if 'terminal' in ipy_str:
            return "ipython"
    except BaseException:
        return "terminal"


def is_jupyter() -> bool:
    """Test if navis is run in a Jupyter notebook.

    Also returns True if inside Google colaboratory!

    Examples
    --------
    >>> from navis.utils import is_jupyter
    >>> # If run outside a Jupyter environment
    >>> is_jupyter()
    False

    """
    return _type_of_script() in ("jupyter", "colab")


def adjust_linkage_colors(dendrogram, clusters, cluster_colors=None):
    """Adjust linkage colors based on the given cluster.

    Parameters
    ----------
    dendrogram :    scipy.cluster.hierarchy.dendrogram
                    Scipy dendrogram object.
    clusters :      list
                    List of clusters. Must contain one label for each leaf in the dendrogram.
    cluster_colors : dict | str, optional
                    Dictionary of cluster colors. If None, a default palette is used. If string,
                    must be the name of a color palette from the `cmap` module.

    Returns
    -------
    None
                    The dendrogram is modified in place.

    """
    assert isinstance(dendrogram, dict)
    assert isinstance(clusters, (list, np.ndarray))
    assert len(clusters) == len(dendrogram["ivl"])

    # Make sure clusters are numpy array
    clusters = np.asarray(clusters)

    # We should order the clusters by their appearance in the dendrogram
    # to avoid having the same color assigned to adjacent clusters
    cl, ix = np.unique(clusters[dendrogram["leaves"]], return_index=True)
    cl = cl[np.argsort(ix)]

    # Set a default palette
    if cluster_colors is None:
        cluster_colors = "tab10"

    if isinstance(cluster_colors, str):
        # Get all the colors from the palette - if we ask for more than
        # the palette has, it will start cycling oddly:
        # Imagine a palette with 3 colors, r, g, b. If we ask for 5 colors,
        # we will get r, r, g, g, b instead of r, g, b, r, g.
        colors = list(cmap.Colormap(cluster_colors).iter_colors())
        cluster_colors = {c: colors[i % len(colors)] for i, c in enumerate(cl)}
        cluster_colors = {k: tuple(v) for k, v in cluster_colors.items()}
    elif isinstance(cluster_colors, dict):
        assert all(
            [c in cluster_colors for c in cl]
        ), "Not all clusters have colors assigned."
    else:
        raise ValueError(f"Expected dict or str, got {type(cluster_colors)}.")

    # First we can adjust the leaf colors
    dendrogram["leaves_color_list"] = [
        cluster_colors[c] for c in clusters[dendrogram["leaves"]]
    ]

    # Next, we need to adjust the link colors. For that we will map each "hinge" first to all its leafs
    parents = {}
    top_center_to_ix = {}
    leafs = []
    for i in range(len(dendrogram["icoord"])):
        ic = dendrogram["icoord"][i]
        dc = dendrogram["dcoord"][i]

        top_center = (ic[0] + (ic[-1] - ic[0]) / 2, dc[1])
        bottom_left = (ic[0], dc[0])
        bottom_right = (ic[-1], dc[-1])

        top_center_to_ix[top_center] = i
        parents[bottom_left] = top_center
        parents[bottom_right] = top_center

        # If either of the two nodes are leafs, we will store an integer instead of the position
        if dc[0] == 0:
            leafs.append(int((ic[0] - 5) / 10))
            parents[int((ic[0] - 5) / 10)] = top_center
        if dc[-1] == 0:
            leafs.append(int((ic[-1] - 5) / 10))
            parents[int((ic[-1] - 5) / 10)] = top_center

    ix_to_leaf = dict(zip(range(len(leafs)), dendrogram["leaves"]))
    hinge_to_leafs = {i: [] for i in range(len(dendrogram["icoord"]))}
    for i, l_ix in enumerate(leafs):
        p = parents.get(l_ix, None)
        while p:
            ix = top_center_to_ix[p]
            hinge_to_leafs[ix].append(ix_to_leaf[l_ix])
            p = parents.get(p, None)

    # `hinge_to_leafs` now contains, for a given hinge, all the leafs as indexed into `clusters`
    for i, leafs in hinge_to_leafs.items():
        cl = np.unique(clusters[leafs])
        # If this hinge maps to only one cluster we can give it a color
        if len(cl) == 1:
            dendrogram["color_list"][i] = cluster_colors[cl[0]]
        # If it maps to multiple clusters, we'll give it a gray color
        else:
            dendrogram["color_list"][i] = (0.5, 0.5, 0.5, 1.0)

    return hinge_to_leafs


def apply_matrix(pos, mat):
    """Apply homogeneous transformation matrix to a set of points."""
    assert pos.shape[1] == 3
    assert mat.shape == (4, 4)

    pos = np.hstack([pos, np.ones((pos.shape[0], 1))])
    pos = np.dot(pos, mat.T)
    return pos[:, :3]


def hash_function(state, value):
    """This is a modified murmur hash.
    """
    k1 = 0xCC9E2D51
    k2 = 0x1B873593
    state = state & 0xFFFFFFFF
    value = (value * k1) & 0xFFFFFFFF
    value = ((value << 15) | value >> 17) & 0xFFFFFFFF
    value = (value * k2) & 0xFFFFFFFF
    state = (state ^ value) & 0xFFFFFFFF
    state = ((state << 13) | state >> 19) & 0xFFFFFFFF
    state = ((state * 5) + 0xE6546B64) & 0xFFFFFFFF
    return state


def rgb_from_segment_id(color_seed, segment_id):
    """Return the RGBA for a segment given a color seed and the segment ID."""
    segment_id = int(segment_id)  # necessary since segment_id is 64 bit originally
    result = hash_function(state=color_seed, value=segment_id)
    newvalue = segment_id >> 32
    result2 = hash_function(state=result, value=newvalue)
    c0 = (result2 & 0xFF) / 255.0
    c1 = ((result2 >> 8) & 0xFF) / 255.0
    h = c0
    s = 0.5 + 0.5 * c1
    v = 1.0
    return tuple([v * 255 for v in colorsys.hsv_to_rgb(h, s, v)])
