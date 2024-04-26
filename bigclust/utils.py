import cmap

import numpy as np

def _type_of_script() -> str:
    """Return context (terminal, jupyter, colab, iPython) in which navis is run."""
    try:
        ipy_str = str(type(get_ipython()))  # noqa: F821
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        elif 'colab' in ipy_str:
            return 'colab'
        else:  # if 'terminal' in ipy_str:
            return 'ipython'
    except BaseException:
        return 'terminal'


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
    return _type_of_script() in ('jupyter', 'colab')


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
    assert len(clusters) == len(dendrogram['ivl'])

    # Make sure clusters are numpy array
    clusters = np.asarray(clusters)
    cl = np.unique(clusters)

    # Set a default palette
    if cluster_colors is None:
        cluster_colors = "tab20"

    if isinstance(cluster_colors, str):
        cluster_colors = dict(zip(cl, cmap.Colormap(cluster_colors).iter_colors(len(cl))))
        cluster_colors = {k: tuple(v) for k, v in cluster_colors.items()}
    elif isinstance(cluster_colors, dict):
        assert all([c in cluster_colors for c in cl]), "Not all clusters have colors assigned."
    else:
        raise ValueError(f"Expected dict or str, got {type(cluster_colors)}.")

    # First we can adjust the leaf colors
    dendrogram['leaves_color_list'] = [cluster_colors[c] for c in clusters[dendrogram['leaves']]]

    # Next, we need to adjust the link colors. For that we will map each "hinge" first to allits leafs
    parents = {}
    top_center_to_ix = {}
    leafs = []
    for i in range(len(dendrogram['icoord'])):
        ic = dendrogram['icoord'][i]
        dc = dendrogram['dcoord'][i]

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

    ix_to_leaf = dict(zip(range(len(leafs)), dendrogram['leaves']))
    hinge_to_leafs = {i: [] for i in range(len(dendrogram['icoord']))}
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
            dendrogram['color_list'][i] = cluster_colors[cl[0]]
        # If it maps to multiple clusters, we'll give it a gray color
        else:
            dendrogram['color_list'][i] = (0.5, 0.5, 0.5, 1.0)









    to_map = list(range(len(dendrogram['icoord'])))

    i_to_leaf = {}
    pos_to_leaf = {}
    while to_map:
        i = to_map[0]

        ic = dendrogram['icoord'][i]
        dc = dendrogram['dcoord'][i]

        # Check if this is a leaf (we're assuming leafs are at y=0 and spaced out in intervals of 10 with a +5 offset)
        if dc[0] == 0:
            i_to_leaf[i] = int((ic[0] - 5) / 10)
        elif dc[-1] == 0:
            i_to_leaf[i] = int((ic[-1] - 5) / 10)
        elif (ic[0], dc[0]) in pos_to_leaf:
            i_to_leaf[i] = pos_to_leaf[(ic[0], dc[0])]
        elif (ic[-1], dc[1]) in pos_to_leaf:
            i_to_leaf[i] = pos_to_leaf[(ic[-1], dc[1])]
        else:
            # If we ended up here, we can't yet map this hinge
            continue

        # If we got to here, we found a mapping
        # We track this hinge by its top center position, i.e. where the next
        # higher hinge would attach
        top_center = (ic[0] + (ic[-1] - ic[0]) / 2, dc[1])
        pos_to_leaf[top_center] = i_to_leaf[i]
        to_map.pop(0)


