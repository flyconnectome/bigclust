import cmap
import inspect
import requests

import dvid as dv
import numpy as np
import pygfx as gfx
import octarine as oc
import nglscenes as ngl
import cloudvolume as cv

import navis

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor


class NglViewer:
    """Viewer for neurons.

    Parameters
    ----------
    data :  pandas.DataFrame
            A DataFrame in the same order as neurons in original distance matrix.
            Must, for each neuron, contain a 'segment_id' or 'id' column, and a
            `source` column that contains a URL for CloudVolume. Optionally, you
            can add a 'color' column that contains a color for the neuron.

    """

    def __init__(
        self,
        data,
        neuropil_mesh=None,
        neuropil_source=None,
        debug=False,
        max_threads=20,
        title="Octarine Viewer",
    ):
        self.debug = debug

        # To avoid collisions when the same ID exists in multiple datasets, we will
        # use a multi-index of (id, dataset).
        self.data = data.copy()

        # In some older DataFrames we used `segment_id` instead of `id`
        if "segment_id" in self.data.columns and "id" not in self.data.columns:
            self.data.rename(columns={"segment_id": "id"}, inplace=True)

        self.data["id"] = self.data["id"].astype(int)  # make sure IDs are integers

        if "dataset" in self.data.columns:
            self.data.set_index(
                ["id", "dataset"], inplace=True, drop=False, verify_integrity=True
            )
        else:
            self.data.set_index(["id"], inplace=True, drop=False, verify_integrity=True)
            self.data["dataset"] = "default"

        self.set_default_colors()

        # Parse cloudvolumes
        self.volumes = {}
        for url in self.data.source.unique():
            self.volumes[url] = self.make_volume(url)

        self.viewer = oc.Viewer(title=title)
        self._centered = False

        self._neuropil_mesh = neuropil_mesh
        self._neuropil_source = neuropil_source
        if neuropil_mesh:
            self.viewer.add_mesh(
                neuropil_mesh, color=(0.8, 0.8, 0.8, 0.01), name="neuropil"
            )
            self.viewer.center_camera()
            self._centered = True

        # Holds the futures for requested data
        self.futures = {}
        self.n_failed = 0  # track the number of failed requests
        self.pool = ThreadPoolExecutor(max_threads)

        # Tracks which neurons we've already loaded
        self._segments = {}

        # Optional cache
        self.cache = {}

        self.register()

    def __del__(self):
        if hasattr(self, "pool"):
            self.pool.shutdown(cancel_futures=True, wait=True)
        if hasattr(self, "viewer"):
            self.unregister()

    @property
    def use_cache(self):
        return getattr(self, "_use_cache", False)

    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value
        self.report(f"Cache set to {value}", flush=True)

    def set_default_colors(self):
        # Check for color column
        if "color" in self.data.columns:
            self._colors = self.data.color.to_dict()
            self._colors = {k: cmap.Color(v).hex for k, v in self._colors.items()}
        else:
            self._colors = {}

    def set_colors(self, colors):
        """Set the colors for the neurons."""
        assert isinstance(colors, dict)
        self._colors = {k: cmap.Color(v).hex for k, v in colors.items()}

    def update_colors(self, colors):
        """Update the colors for the neurons."""
        assert isinstance(colors, dict)
        self._colors.update({k: cmap.Color(v).hex for k, v in colors.items()})

    def make_volume(self, url, mip=0):
        """Make a volume from a URL."""
        if url.startswith("dvid://"):
            return DVIDVolume(url)
        else:
            try:
                return cv.CloudVolume(url, progress=False)
            except KeyError as e:
                # Try with a simple wrapper - CloudVolume is really finicky
                if url.startswith("precomputed://"):
                    return PrecomputedVolume(url)
                else:
                    raise e

    def close(self):
        self.viewer.close()

    def register(self):
        self.viewer.add_animation(self.check_futures, run_every=20, req_render=False)

    def unregister(self):
        self.viewer.remove_animation(self.check_futures)

    def report(self, *args, **kwargs):
        """Print a message if in debug mode."""
        if self.debug:
            print(*args, **kwargs)

    def show(self, ids, datasets=None, lod=-1, add_as_group=False, **kwargs):
        """Add data to the viewer.

        Parameters
        ----------
        ids :       iterable
                    The IDs of neurons to show.
        datasets :  iterable, optional
                    If provided, only show neurons where ID and dataset match.
                    Must be of the same length as `ids`.
        kwargs :    dict
                    Keyword arguments to pass to the viewer.

        """
        self.report(f"Asked to show {len(ids)} ids:", ids, flush=True)

        if not len(ids):
            self.report("Clearing viewer", flush=True)
            self.clear()
            return

        if datasets is None:
            to_show = self.data.loc[ids]
            datasets = to_show.dataset.values
        else:
            if len(ids) != len(datasets):
                raise ValueError("IDs and datasets must be of the same length.")
            to_show = self.data.loc[zip(ids, datasets)]

        miss = ~np.isin(ids, to_show.id.values)
        if np.any(miss):
            raise ValueError(f"IDs {ids[miss]} not found in the data.")

        self.report(
            f"Showing {len(to_show)} neuron(s): ",
            to_show.id.values.tolist(),
            flush=True,
        )

        # Remove those segments we don't want
        self.remove_objects(
            [
                x for x in self._segments if x not in to_show.index.values.tolist()
            ]  # do not remove the .tolist() here!
        )

        # Cancel all futures we don't need anymore
        # Note: we need to use list() because we're potentially
        # modifying the dict inside the loop
        for (id, _), future in list(self.futures.items()):
            if id not in to_show.index.values.tolist():
                self.report("  Cancelling future for", id, flush=True)

                # Cancel future
                future.cancel()

                # Remove from futures
                self.futures.pop(id, None)

        # Now drop those already on display
        to_show = to_show[~to_show.index.isin(self._segments)]

        for _, row in to_show.iterrows():
            id, dataset = row.name
            self.report(f"  Adding {id} ({dataset})", flush=True)

            # Skip if we're already loading this segment
            if (id, dataset) in self.futures:
                continue

            if not add_as_group:
                name = f"{row.label} ({row.id})"
            else:
                first_id = to_show.index.values[0][0]  # remember this is a multiindex
                name = f"group_{first_id}"

            if (id, dataset) in self.cache:
                self.report(f"  Using cached visual for {id} ({dataset})", flush=True)
                # Get the cached visual
                visual = self.cache[(id, dataset)]

                # It's possible that the color scheme changed or that
                # the user changed the color manually. We will need to
                # reset the color before adding the visual to the
                # scene.
                color = kwargs.get("color", None)
                if color is None:
                    if (id, dataset) in self._colors:
                        color = self._colors[(id, dataset)]
                    else:
                        color = self.viewer._next_color()
                visual.material.color = gfx.Color(color)

                # We also need to make sure that the visual is visible
                visual.visible = True

                self.viewer.add(visual, name=str(name), center=False)
                self._segments[(id, dataset)] = visual
            else:
                self.report(f"  Loading visual for {id} ({dataset})", flush=True)
                self.futures[((id, dataset), name)] = self.pool.submit(
                    self._load_mesh,
                    (id, dataset),
                    self.volumes[row.source],
                    lod=lod,
                    **kwargs,
                )

    def remove_objects(self, objects):
        """Remove objects from the viewer."""
        self.report(f"  Removing {len(objects)} neurons: ", objects, flush=True)
        for x in objects:
            if x not in self._segments:
                raise ValueError(f"Segment {x} not found in viewer.")
            self.viewer.remove_objects([self._segments[x]])
            self._segments.pop(x, None)

    def clear(self):
        """Clear the viewer of selected segments."""
        self.report("Clearing viewer", flush=True)
        self.viewer.remove_objects(list(self._segments.values()))
        self._segments.clear()

        self.report(f"  Canceling {len(self.futures)} futures", flush=True)
        for future in self.futures.values():
            future.cancel()

        self.futures.clear()

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()

    def neuroglancer_scene(self, group_by="source", use_colors=True):
        """Generate neuroglancer scene for the current state.

        Parameters
        ----------
        groupby :   "source" | "color" | "label"
                    Logic for how to group the segments. If "color" or "label" we
                    will try combine different sources by using sub-sources. This
                    will not work if IDs exist in multiple sources!
        use_colors : bool
                    Whether to use the colors from the viewer for the neuroglancer
                    scene. If False, neuroglancer will determine the colors itself.
        """
        layers = []
        id2source = self.data.source.to_dict()
        if group_by == "source":
            for source in self.data.source.unique():
                layer = ngl.SegmentationLayer(source)
                layer["source"] = {
                    "url": fix_dvid_source(source),
                    "subsources": {
                        "default": True,
                        "meshes": True,
                        "bounds": False,
                        "skeletons": False,
                    },
                }
                # Collect all IDs for this source
                ids = [i for i, _ in self._segments.items() if id2source[i] == source]
                ids += [i for i, _ in self.futures.keys() if id2source[i] == source]

                # IDs can be a list of (id, dataset) tuples or just a list of IDs
                if len(ids) and isinstance(ids[0], (list, tuple)):
                    ids_flat = [x[0] for x in ids]
                else:
                    ids_flat = ids

                layer["segments"] = ids_flat

                # Set colors for the segments
                if use_colors:
                    layer.set_colors(
                        {i: self._colors.get(ii, "w") for i, ii in zip(ids_flat, ids)}
                    )

                layers.append(layer)
        elif group_by == "color":
            # Group by color
            for color in list(set(self._colors.values())):
                layer = ngl.SegmentationLayer(color)

                # Collect all IDs for this color
                ids = [
                    i
                    for i, _ in self._segments.items()
                    if self._colors.get(i, None) == color
                ]
                ids += [
                    i
                    for i, _ in self.futures.keys()
                    if self._colors.get(i, None) == color
                ]

                # If there are no IDs with this color we can skip this color
                if len(ids) == 0:
                    continue

                # Collect the sources for this colors
                sources = list(set([id2source[i] for i in ids]))

                # Sort sources such that the DVID source comes first
                sources = sorted(
                    sources, key=lambda x: x.startswith("dvid://"), reverse=True
                )
                layer["source"] = []
                for source in sources:
                    # Add the source to the layer
                    layer["source"].append(
                        {
                            "url": fix_dvid_source(source),
                            "subsources": {
                                "default": True,
                                "meshes": True,
                                "bounds": False,
                                "skeletons": False,
                            },
                        }
                    )

                # IDs can be a list of (id, dataset) tuples or just a list of IDs
                if isinstance(ids[0], (list, tuple)):
                    ids_flat = [x[0] for x in ids]
                else:
                    ids_flat = ids
                layer["segments"] = ids_flat

                # Set the color for the layer
                if use_colors:
                    layer.set_colors(color)

                layers.append(layer)
        elif group_by == "label":
            # Map the labels in the viewer's legend to IDs
            object2id = {v: k for k, v in self._segments.items()}
            label2id = {
                label: [object2id[o] for o in objects if o in object2id]
                for label, objects in self.viewer.objects.items()
            }
            label2id = {k: v for k, v in label2id.items() if len(v) > 0}
            for label, ids in label2id.items():
                layer = ngl.SegmentationLayer(label)

                sources = list(set([id2source[i] for i in ids]))
                # Sort sources such that the DVID source comes first
                sources = sorted(
                    sources, key=lambda x: x.startswith("dvid://"), reverse=True
                )
                layer["source"] = []
                for source in sources:
                    # Add the source to the layer
                    layer["source"].append(
                        {
                            "url": fix_dvid_source(source),
                            "subsources": {
                                "default": True,
                                "meshes": True,
                                "bounds": False,
                                "skeletons": False,
                            },
                        }
                    )

                # IDs can be a list of (id, dataset) tuples or just a list of IDs
                if isinstance(ids[0], (list, tuple)):
                    ids_flat = [x[0] for x in ids]
                else:
                    ids_flat = ids
                layer["segments"] = ids_flat

                # Set colors for the segments
                if use_colors:
                    layer.set_colors(
                        {i: self._colors.get(ii, "w") for i, ii in zip(ids_flat, ids)}
                    )

                layers.append(layer)
        else:
            raise ValueError(
                f"Unknown group_by value: {group_by}. Must be one of 'source', 'color', or 'label'."
            )

        # Generate the scene
        s = ngl.Scene()
        s["layout"] = "3d"
        s["dimensions"] = {"x": [1e-9, "m"], "y": [1e-9, "m"], "z": [1e-9, "m"]}
        s["position"] = [385865.5, 248967.5, 123749.5]  # Default position
        s["projectionScale"] = 495090.6406803968

        # Sort layers such that the DVID source comes first
        # (this is a hack to make sure neuroglancer has bounds to work with)
        def is_dvid_source(layer):
            if isinstance(layer["source"], list):
                return layer["source"][0]["url"].startswith("dvid://")
            else:
                return layer["source"]["url"].startswith("dvid://")

        layers = sorted(layers, key=lambda x: is_dvid_source(x), reverse=True)
        for layer in layers:
            # Add the layer to the scene
            s.add_layers(layer)

        if self._neuropil_source is not None:
            id, source = self._neuropil_source.split("@")
            layer = ngl.SegmentationLayer(source, name="neuropil")
            layer["segments"] = [int(id)]
            layer["segmentDefaultColor"] = "#ffffff"
            layer["meshSilhouetteRendering"] = 3
            layer["objectAlpha"] = 0.4
            layer["source"] = {
                "url": source,
                "subsources": {
                    "default": True,
                    "meshes": True,
                    "bounds": False,
                    "skeletons": False,
                },
            }
            s.add_layers(layer)

        if self.debug:
            print("Neuroglancer scene:")
            from pprint import pprint

            pprint(s.state)
            for layer in s.layers:
                pprint(layer.state)

        return s

    def _load_mesh(self, x, vol, lod=-1, **kwargs):
        """Load a single mesh."""
        if isinstance(x, tuple):
            x, dataset = x
        else:
            dataset = None
        x = int(x)

        # Get the color before we load the mesh
        if "color" not in kwargs:
            if (x, dataset) in self._colors:
                kwargs["color"] = self._colors[(x, dataset)]
            elif x in self._colors:
                kwargs["color"] = self._colors[x]
            else:
                kwargs["color"] = self.viewer._next_color()

        try:
            if "lod" in inspect.signature(vol.mesh.get).parameters:
                m = vol.mesh.get(x, lod=lod)[x]
            else:
                m = vol.mesh.get(x)[x]
        except BaseException as e:
            import traceback

            print(f"Error loading mesh for {x} ({dataset}):")
            traceback.print_exc()
            return e

        return oc.visuals.mesh2gfx(m, **kwargs)

    def check_futures(self):
        """Check if any futures are done."""
        # Keep track of whether we had any futures at the beginning
        has_futures = len(self.futures) > 0

        for ((id, dataset), name), future in self.futures.items():
            if not future.done():
                continue
            visual = future.result()

            # If there is no mesh, skip
            if isinstance(visual, BaseException):
                self.n_failed += 1
                self.report(f"  Failed to load {id} ({dataset}): {visual}", flush=True)
                continue

            self.report(f"  Adding {id} ({dataset}) as '{name}'", flush=True)
            self.viewer.add(visual, name=str(name), center=False)
            self._segments[(id, dataset)] = visual

            # Center on the first neuron
            if not self._centered:
                self.viewer.center_camera()
                self._centered = True

            # Populate cache if necessary
            if self.use_cache:
                self.report(f"  Caching visual for {id}", flush=True)
                self.cache[(id, dataset)] = visual

        # Show progress message
        if has_futures and len(self.futures) > 0:
            msg = f"Loading {len(self.futures):,} neurons"

            if self.n_failed:
                msg += f" ({self.n_failed} failed)"

            self.viewer.show_message(
                msg,
                duration=2,
                position="top-right",
                color="w" if not self.n_failed else "r",
            )

        # Remove completed futures
        self.futures = {k: v for k, v in self.futures.items() if not v.done()}

        # If all futures completed
        if has_futures and len(self.futures) == 0:
            if not self.n_failed:
                self.viewer.show_message(
                    "All neurons loaded", duration=2, position="top-right", color="w"
                )
            else:
                self.viewer.show_message(
                    f"{self.n_failed} neurons failed to load",
                    duration=2,
                    position="top-right",
                    color="r",
                )
                # Reset the number of failed requests
                self.n_failed = 0


class DVIDVolume:
    """Helper class for loading DVID neurons."""

    def __init__(self, url):
        url = url.replace("dvid://", "")

        # Remove any parameters
        self.url = url.split("?")[0]

        self.server, self.node = self.url.replace("https://", "").split("/")[:2]
        self.server = f"https://{self.server}"

        # Translate the node name if necessary
        if ":master" in self.node:
            self.node = get_master_node(self.node.replace(":master", ""), self.server)
            self.url = f"{self.server}/{self.node}"

        self.mesh = DVIDMesh(self.server, self.node)


class DVIDMesh:
    def __init__(self, server, node):
        self.server = server
        self.node = node

    def get(self, x, lod=None):
        try:
            m = dv.get_meshes(
                x,
                progress=False,
                output="trimesh",
                server=self.server,
                node=self.node,
                on_error="raise",
            )[0]
        except requests.exceptions.HTTPError:
            print(f" Failed to load mesh for {x} - falling back to alternative service")
            # If the mesh is missing, try the alternative service
            # This is slower but should mesh the neuron from scratch
            url = f"https://ngsupport-bmcp5imp6q-uk.a.run.app/small-mesh?dvid=https://emdata-mcns.janelia.org&uuid={self.node}&body={x}&segmentation=segmentation"
            r = requests.get(url)
            r.raise_for_status()
            m = dv.decode.read_ngmesh(r.content)

        return {int(x): m}


class PrecomputedVolume:
    """Helper class for loading precomputed meshes using navis."""

    def __init__(self, url):
        self.url = url
        self.mesh = PrecomputedMesh(url)


class PrecomputedMesh:
    def __init__(self, url):
        if url.endswith("/"):
            url = url[:-1]

        url = url.replace("precomputed://", "")

        self.url = url

    def get(self, x, lod=None):
        return {
            int(x): navis.read_precomputed(
                f"{self.url}/{x}", progress=False, datatype="mesh"
            )
        }


@lru_cache
def get_master_node(node, server):
    """Cached function to get the master node name."""
    return dv.get_master_node(node, server)


def fix_dvid_source(source):
    """Potentially fix :master in DVID sources."""
    if source.startswith("dvid://") and ":master" in source:
        url = source.replace("dvid://", "")
        url = url.split("?")[0]
        server, node = url.replace("https://", "").split("/")[:2]
        # Get the master node
        master_node = get_master_node(node.replace(":master", ""), f"https://{server}")
        # Replace the node with the master node
        source = source.replace(node, master_node)

    return source
