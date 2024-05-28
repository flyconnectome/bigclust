from ._dendrogram import Dendrogram
from ._heatmap import heatmap
from ._figure import Figure

from . import _visuals
from . import _neuroglancer
from . import _selection
from . import _dendrogram
from . import _heatmap
from . import _figure
from . import utils

def test():
    import numpy as np
    import bigclust as bc
    import pandas as pd

    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, dendrogram

    d = pd.read_feather(
        "/Users/philipps/Github/flywire_retyping/distances/AN_dists.feather"
    ).set_index("index")
    Z = linkage(squareform(d), method="single")

    # fig = bc.Figure()
    fig = bc.dendrogram(Z)

    return fig
