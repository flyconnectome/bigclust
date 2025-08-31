# BigClust

`bigclust` is a set of tools for interactively exploring clusterings via dendrograms or scatterplots with several 100k's data points.

While the focus of `bigclust` is on large connectomic datasets, its modular design allows for easy adaptation to other domains.

Highlights:
- **Interactive dendrograms and scatter plots**: Explore large clusterings interactively with zoom, pan, and selection.
- **Neuroglancer-like 3D viewer**: Visualize neuron morphology in a 3D viewer.
- **Connectivity widget**: Explore connectivity between neurons.


https://github.com/flyconnectome/bigclust/assets/7161148/f1a0ddcb-522d-4655-ad85-158a810348e8

Additional notes:
- `bigclust` is designed to work with data artefacts (distance matrices, linkages, embeddings and such) rather than
  running clusterings itself
- we recommend running `bigclust` via a script (see the real-world example below) but you can also use it interactive from
  a Python shell; in the latter case, you should generally be fine just calling the viewers/widgets' `.show()` method but you may
  have to contend with the event loop to run the app non-blocking and keep the Python REPL responsive

## Installation

For now, the recommended way of installing this package is this:

1. Clone the repository:
   ```bash
   git clone https://github.com/flyconnectome/bigclust.git
   ```
2. Install in "editable" mode:
   ```bash
   cd bigclust
   pip install -e .
   ```

With this setup, you can just `git pull` to update the package.

## Usage

`bigclust` provides several widgets that you can mix and match to fit your needs. The main ones are:

### Dendrograms
To generate a dendrogram, you will minimally need two data artefacts:
1. A [linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) describing the dendrogram (see `scipy.cluster.hierarchy.linkage`)
2. A `pandas` DataFrame with meta data (labels, cluster assigment, etc.) for the original observations.

```python
import pandas as pd
import bigclust as bc
from scipy.cluster.hierarchy import linkage

# Some toy data (re-used from the scipy documentation)
X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
# Generate a linkage matrix defining the dendrogram
Z = linkage(X, 'ward')

# Generate some mock meta data
table = pd.DataFrame({
    'id': range(len(X)),
    'label': [f'Neuron {i}' for i in range(len(X))],
    'cluster': [0, 1, 0, 1, 0, 1, 1, 0],
    'dataset': ['A', 'A', 'B', 'B', 'A', 'B', 'B', 'A']
})

# Instantiate the dendrogram
dend = bc.Dendrogram(
    linkage=Z,
    table=table,
    labels='label',  # column in the table to use as labels
    leaf_types='dataset',  # column in the table to use for leaf markers
    clusters='cluster',  # column in the table to use for cluster coloring
    hover_info="ID: {id}\nLabel: {label}\nCluster: {cluster}",  # info to show on hover
    )

# Show the dendrogram (may not actually be necessary)
dend.show()
```

### Scatter plots
For scatter plots, you really only need a `pandas` DataFrame with meta data and columns for `x` and `y` coordinates. Alternatively, you can also
provide an observation vector or distance matrix, in which case `bigclust` will compute a 2D embedding for you using UMAP.

```python
import numpy as np

# Generate some toy x/y coordinates
x = np.random.rand(100)
y = np.random.rand(100)

# Generate some mock meta data
table = pd.DataFrame({
    'id': range(len(x)),
    'label': [f'Neuron {i}' for i in range(len(x))],
    'dataset': ['A' if i % 2 == 0 else 'B' for i in range(len(x))],
    'color': ['red' if i % 3 == 0 else 'green' for i in range(len(x))],
    'x': x,
    'y': y,
})

# Instantiate the scatter plot
scatter = bc.ScatterPlot(
    data=table,
    labels='label',  # column in the table to use as labels
    ids="id",  # column in the table to use as IDs
    colors="color",
    markers="dataset",  # column in the table to use for markers
    hover_info="ID: {id}\nLabel: {label}\nDataset: {dataset}",  # info to show on hover
)

# Show the scatter plot (may not actually be necessary)
scatter.show()
```

### 3D viewer

To visualize neuron morphology in a 3D viewer, you can use the `NglViewer` class. It is not designed as a standalone widget, but rather
as a widget that can be synced with a dendrogram or scatter plot.

The viewer requires a `pandas` DataFrame with an `id` and a `source` column. The latter must specify a neuroglancer-compatible data source, e.g. something along the lines of `precomputed://gs://my-bucket/my-data`.

Importantly, the `id` is expected to be in the same order as the meta data used for the dendrogram/scatterplot. So typically, you will just use the same `table` DataFrame you already passed to the dendrogram/scatterplot.

```python
# Starting from the previous example, we will add a 3d viewer
# Important: this is just mock data, you will have to use a real neuroglancer-compatible data source!
table["source"] = "precomputed://gs://my-bucket/my-data"  # specify the data source

# Instantiate the 3D viewer
ngl = bc.NglViewer(data=table)

# Show the viewer (may not actually be necessary)
ngl.show()

# Next we have to sync the viewer to either the dendrogram or the scatter plot
scatter.sync_viewer(ngl)  # or dend.sync_viewer(ngl)
```

### Real World Example
Let's illustrate the usage with a small real-world example using the [`cocoa`](https://github.com/flyconnectome/cocoa) package for comparative connectomic analyses. To run this example, in addition to bigclust you will need to have:
- `cocoa` installed (see link above)
- follow the instructions in [this tutorial](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html) to set up your API token for the FlyWire dataset

### Step 1: Run a co-clustering

```python
import cocoa as cc
import numpy as np

# Co-cluster two cell types in hemibrain
cl = cc.generate_clustering(fw=['DA1_lPN', 'DA2_lPN', ]).compile()
```

### Step 2: Store the data artifacts

```python
# Get the linkage (this is a simple scipy linkage)
Z = cl.get_linkage(method='ward')
# Save the linkage matrix to a file
np.save("linkage.npy", Z, allow_pickle=False)

# Prepare a table with details we can use as e.g. labels in the dendrogram
t = cl.to_table(
    cl.extract_homogeneous_clusters(max_dist=2, min_dist=.1, linkage=Z),
    linkage=Z
)

# Save and make sure the order is the same as in our distance matrix
t.set_index("id").reindex(cl.dists_.index).reset_index(drop=False).to_feather("cosine_table.feather")
```

*Feel free to use more sensible file names. If you do, you have to adjust the code below accordingly.*

### Step 3: Write a start-up script

Open a new python script - name it e.g. `run_bigclust.py`:

```python
import pandas as pd
import numpy as np
import trimesh as tm
import bigclust as bc

from wgpu.gui.auto import run


if __name__ == "__main__":
    print("Loading data...", flush=True, end="")

    # Load the linkage matrix
    Z = np.load("linkage.npy")

    # This is the table with the neuron information, including the clusters
    table = pd.read_feather("cosine_table.feather")

    # Add source information -> we need this to load the neuron meshes in the Neuroglancer viewer
    table["source"] = "precomputed://gs://flywire_v141_m783"

    # Here we define the default colors for the neurons
    table["color"] = table.dataset.map(
        {"FwR": "cyan", "FWL": "lightskyblue"}
    )

    print(" Done.", flush=True)

    print("Making dendrogram...", flush=True, end="")
    # Now we will instantiate the dendrogram
    fig = bc.Dendrogram(
        Z,
        table=table,
        labels='label',
        leaf_types='dataset',
        clusters='cluster',
        ids='id',
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

    # Load the neuropil mesh for FlyWire from Github
    # We will add this to the viewer to make navigation easier
    # You could download and store it locally if you want to
    neuropil_mesh = tm.load_remote(
        "https://github.com/navis-org/navis-flybrains/raw/main/flybrains/meshes/FLYWIRE.ply"
    )

    # Instantiate the viewer
    ngl = bc.NglViewer(table, neuropil_mesh=neuropil_mesh)
    ngl.viewer.size = (ngl.viewer.canvas.screen().size().width(), 500)
    ngl.viewer.canvas.move(0, 400)

    # Tell the dendrogram to sync with the viewer
    fig.sync_viewer(ngl)

    print(" Done!", flush=True)

    # Run the app
    # Note: this is only necessary if we're running bigclust from a script
    run()
```

*Make sure to adjust the filepaths if necessary.*

### Step 4: Run the script

Make sure you have all the data artifacts (`linkage.npy` and `cosine_table.feather`) in the same folder as
the `run_bigclust.py` script. Then:

```bash
python run_bigclust.py
```

### Step 5: Using the app

You should now be seeing something like this:

![overview screenshot](_static/overview.png)

The top window contains the dendrogram, the bottom is your 3D viewer.

Dendrogram controls:
- scroll to zoom in/out
- mouse drag to move around
- shift + mouse drag to select neurons (selected neurons should appear in the viewer)
- `escape` to clear the selection
- hover over a dendrogram leaf to show extra information (see `hover_info` parameter)

3D viewer controls:
- scroll up/down to zoom in/out
- mouse drag to rotate
- two-finger (or middle mouse button) mouse drag to pan
- `1`/`2`/`3` to center the view to frontal/dorsal/lateral

In addition to the above, you can press `C` while either the viewer or the dendrogram/scatter window
is active to bring up a GUI control panel.

<center><img src="_static/controls.png" width="400"></center>

