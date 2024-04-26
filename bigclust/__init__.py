from ._dendrogram import dendrogram
from ._figure import Figure


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
