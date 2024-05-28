# BigClust

`BigClust` is a set of tools for interactively exploring large clusterings via dendrograms or heatmaps.
For that we are making use of the `pygfx` WGPU-based rendering engine.

## Installation

For now the recommended way of installing this package is this:

1. Clone the repository:
   ```bash
   git clone https://github.com/flyconnectome/bigclust.git
   ```
2. Install in "editable" mode:
   ```bash
   cd bigclust
   pip install -e .
   ```

In the future, you just need to `git pull` to update the package.

## Usage

I imagine the typical usage will be to run a big one-off
clustering on a remote cluster node and then
load that clustering into `bigclust` on a local machine. Therefore, `bigclust` is designed to work with data artefacts
rather than run the clustering itself.

Here, we will illustrate the usage with a simple toy example using the
[`cocoa`](https://github.com/flyconnectome/cocoa) package:

#### Step 1: Run a co-clustering

```python
import cocoa as cc
import numpy as np

# Co-cluster two cell types in the male CNS left vs right
cl = cc.generate_clustering(mcns=['DA1_lPN', 'DA2_lPN']).compile()
```

### Step 2: Get the data artifacts

```python
# Save the neuron IDs as they appear in the distance matrix
np.save("dist_index.npy", cl.dists_.index, allow_pickle=False)

# Get the linkage
Z = cl.get_linkage(method='ward')
np.save("linkage.npy", Z, allow_pickle=False)

# Prepare a table with details we can use for
t = cl.to_table(cl.extract_homogeneous_clusters(max_dist=2, min_dist=.1, linkage=Z), linkage=Z)

# This is optional:
# Which neurons have a mapping that was used to co-cluster?
mappings = cc.GraphMapper()._mappings[('MaleCNS',)]
t['label_used'] = t.id.isin(mappings)
t['mapping'] = t.id.map(mappings)

t.to_feather("cosine_table.feather")
```

_*feel free to use more sensible file names_

### Step 3: Write a start-up script

Open a new python script - call it e.g. `run_bigclust.py`:

```python
import sys

import pandas as pd
import numpy as np
import trimesh as tm
import bigclust as bc

from wgpu.gui.auto import run
from bigclust._neuroglancer import NglViewer


if __name__ == "__main__":
    # Parse arguments
    update_labels = "--update-labels" in sys.argv

    print("Loading data...", flush=True, end="")

    # Load the linkage matrix
    Z = np.load("linkage.npy")

    # Load the index of the distance matrix (i.e. the neuron IDs)
    # (we need that so we know which neuron is which leaf in the matrix)
    index = np.load("dist_index.npy")

    # This is the table with the neuron information, including the homogeneous clusters
    table = (
        pd.read_feather("cosine_table.feather")
        # This shouldn't be necessary but just in case:
        # make sure the table is in the same order as the linkage matrix
        .set_index("id")
        .reindex(index)
    ).reset_index(drop=False)

    # Update labels if this was requested via the --update-labels flag
    # Note: this will require having set the appropriate credentials
    # (see https://github.com/flyconnectome/cocoa for details)
    if update_labels:
        import cocoa as cc

        mcns = cc.MaleCNS()
        mcns_types = mcns.get_labels(None, backfill=True)
        table["label"] = table.id.map(mcns_types).values

    # Here we define the actual leaf labels in the dendrogram
    table["dend_label"] = table.label.fillna("untyped")

    # Add cell type counts to the label as "{label}({count})"
    ct_counts = table.label.value_counts()
    table.loc[table.label.isin(ct_counts.index), "dend_label"] += (
        "("
        + table.loc[table.label.isin(ct_counts.index)].label.map(ct_counts).astype(str)
        + ")"
    )

    # Add asterisk for labels that were used in the co-clustering
    table.loc[table.label_used, "dend_label"] = (
        table.loc[table.label_used, "dend_label"] + "*"
    )

    # Here we define the hover information for the dendrogram
    # (hover over the leafs to show)
    table["hover_info"] = table.id.astype(str) + "\n" + table.mapping

    # Add source information -> we need this to load the neuron meshes in the viewer
    # You can get that info by going to e.g. Clio or NeuPrint and checking the segmentation
    # "source" in the neuroglancer
    table["source"] = "dvid://SOURCE_URL"

    # Here we define the default colors for the neurons
    table["color"] = table.dataset.map(
        {"McnsR": "cyan", "McnsL": "lightskyblue"}
    )

    print(" Done.", flush=True)

    print("Making dendrogram...", flush=True, end="")
    # Now we will instantiate the dendrogram
    fig = bc.Dendrogram(
        Z,
        labels=table.dend_label,
        leaf_types=table.dataset,
        clusters=table.cluster,
        ids=table.id,
        hover_info=table.hover_info,
    )
    fig.show()

    # Some tweaks:
    fig.size = (fig.canvas.screen().size().width(), 300)  # make the dendrogram fill the width of the screen
    fig.canvas.move(0, 0)  # nove it into the top left corner
    fig.font_size = 6  # slight larger font size
    fig.label_vis_limit = 300  # show more labels before hiding all
    fig.leaf_size = 3  # slightly larger leaf size
    fig.set_yscale(100)  # make the dendrogram a bit taller

    print(".", flush=True, end="")

    # Load the neuropil mesh for the maleCNS from Github
    # We will add this to the viewer to make navigation easier
    # You could download and put this locally if you want to
    neuropil_mesh = tm.load_remote(
        "https://github.com/navis-org/navis-flybrains/raw/main/flybrains/meshes/JRCFIB2022M.ply"
    )

    # Instantiate the viewer
    ngl = NglViewer(table, neuropil_mesh=neuropil_mesh)
    ngl.viewer.size = (ngl.viewer.canvas.screen().size().width(), 500)
    ngl.viewer.canvas.move(0, 400)

    # Tell the dendrogram to sync with the viewer
    fig.sync_viewer(ngl)

    print(" Done!", flush=True)

    # Run the app
    run()
```

### Step 4: Fire up `bigclust`

Make sure you have all the data artifacts (`linkage.npy`, etc.) in the same folder as
the `run_bigclust.py` script. Then:

```bash
python run_bigclust.py
```

Use the optional `--update-labels` flag to update the labels from Clio/NeuPrint.


#### Step 5: Using the app

You should now be seeing something like this:

![overview screenshot](_static/overview.png)

The top window contains the dendrogram, the bottom is your 3D viewer.

Dendrogram controls:
- scroll up/down to zoom in/out
- mouse drag to move around
- shift+drag to select neurons (they should appear in the viewer)
- hover over a dendrogram leaf to show extra information (see `hover_info` parameter)

3D viewer controls:
- scroll up/down to zoom in/out
- mouse drag to rotate
- two-finger (or middle mouse button) mouse drag to pan
- `1`/`2`/`3` to center the view to frontal/dorsal/lateral

In addition to the above, you can press `C` while either the viewer or the dendrogram window
is active to bring up a GUI control panel.

<center><img src="_static/controls.png" width="400"></center>

