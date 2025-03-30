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
        self.data = (
            data.rename({"segment_id": "id"}, axis=1)
            .astype({"id": int})
            .set_index("id")
        )
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
        return getattr(self, '_use_cache', False)

    @use_cache.setter
    def use_cache(self, value):
        self._use_cache = value
        self.report(f"Cache set to {value}", flush=True)

    def set_default_colors(self):
        # Check for color column
        if "color" in self.data.columns:
            self._colors = self.data.color.to_dict()
            self._colors = {int(k): cmap.Color(v).hex for k, v in self._colors.items()}
        else:
            self._colors = {}

    def set_colors(self, colors):
        """Set the colors for the neurons."""
        assert isinstance(colors, dict)
        self._colors = colors

    def update_colors(self, colors):
        """Update the colors for the neurons."""
        assert isinstance(colors, dict)
        self._colors.update(colors)

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

    def show(self, ids, lod=-1, add_as_group=False, **kwargs):
        """Add data to the viewer.

        Parameters
        ----------
        ids :       iterable
                    The IDs of neurons to show.
        kwargs :    dict
                    Keyword arguments to pass to the viewer.

        """
        if not len(ids):
            self.report("Clearing viewer", flush=True)
            self.clear()
            return

        miss = ~np.isin(ids, self.data.index.values)
        if np.any(miss):
            raise ValueError(f"IDs {ids[miss]} not found in the data.")

        to_show = self.data.loc[ids]
        self.report(
            f"Showing {len(to_show)} neurons: ",
            to_show.index.values.tolist(),
            flush=True,
        )

        # Remove those segments we don't want
        self.remove_objects([x for x in self._segments if x not in to_show.index.values])

        # Cancel all futures we don't need anymore
        # Note: we need to use list() because we're potentially
        # modifying the dict inside the loop
        for (id, _), future in list(self.futures.items()):
            if id not in to_show.index.values:
                self.report("  Cancelling future for", id, flush=True)

                # Cancel future
                future.cancel()

                # Remove from futures
                self.futures.pop(id, None)

        # Now drop those already on display
        to_show = to_show[~to_show.index.isin(self._segments)]

        for _, row in to_show.iterrows():
            self.report("  Adding", row.name, flush=True)

            # Skip if we're already loading this segment
            if row.name in self.futures:
                continue

            if not add_as_group:
                name = f"{row.label} ({row.name})"
            else:
                name = f"group_{to_show.index.values[0]}"

            if row.name in self.cache:
                self.report("  Using cached visual for", row.name, flush=True)
                # Get the cached visual
                visual =self.cache[row.name]

                # It's possible that the color scheme changed or that
                # the user changed the color manually. We will need to
                # reset the color before adding the visual to the
                # scene.
                color = kwargs.get("color", None)
                if color is None:
                    if int(row.name) in self._colors:
                        color = self._colors[int(row.name)]
                    else:
                        color = self.viewer._next_color()
                visual.material.color = gfx.Color(color)

                # We also need to make sure that the visual is visible
                visual.visible = True

                self.viewer.add(visual, name=str(name), center=False)
                self._segments[row.name] = visual
            else:
                self.report("  Loading visual for", row.name, flush=True)
                self.futures[(row.name, name)] = self.pool.submit(
                    self._load_mesh,
                    row.name,
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

    def neuroglancer_scene(self):
        """Generate neuroglancer scene for the current state."""
        # Go over all visible segments and get their layers
        ids_per_layer = {source: [] for source in self.data.source.unique()}
        id2source = self.data.source.to_dict()

        # Add already loaded IDs
        for (id, visual) in self._segments.items():
            ids_per_layer[id2source[id]].append(id)

        # Add IDs that are currently being loaded
        for (id, visual), future in self.futures.items():
            ids_per_layer[id2source[id]].append(id)

        # Generate the scene
        s = ngl.Scene()
        s["layout"] = "3d"
        s["dimensions"] = {"x": [1e-9, "m"], "y": [1e-9, "m"], "z": [1e-9, "m"]}
        s["position"] = [385865.5, 248967.5, 123749.5]  # Default position
        s["projectionScale"] = 495090.6406803968

        # Sort the layers such that the DVID source comes first
        # (this is a hack to make sure neuroglancer has bounds to work with)
        sources = sorted(
            ids_per_layer.keys(), key=lambda x: x.startswith("dvid://"), reverse=True
        )

        for source in sources:
            ids = ids_per_layer[source]
            # Translate "{node}:master" to the actual master node
            if source.startswith("dvid://") and ":master" in source:
                url = source.replace("dvid://", "")
                url = url.split("?")[0]
                server, node = url.replace("https://", "").split("/")[:2]
                # Get the master node
                master_node = get_master_node(
                    node.replace(":master", ""), f"https://{server}"
                )
                # Replace the node with the master node
                source = source.replace(node, master_node)

            layer = ngl.SegmentationLayer(source)
            layer["segments"] = ids
            layer["source"] = {
                "url": source,
                "subsources": {
                    "default": True,
                    "meshes": True,
                    "bounds": False,
                    "skeletons": False,
                },
            }

            # Set colors
            layer.set_colors({i: self._colors.get(i, "w") for i in ids})

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

        return s

    def _load_mesh(self, x, vol, lod=-1, **kwargs):
        """Load a single mesh."""
        x = int(x)

        # Get the color before we load the mesh
        if "color" not in kwargs:
            if int(x) in self._colors:
                kwargs["color"] = self._colors[int(x)]
            else:
                kwargs["color"] = self.viewer._next_color()

        try:
            if "lod" in inspect.signature(vol.mesh.get).parameters:
                m = vol.mesh.get(x, lod=lod)[x]
            else:
                m = vol.mesh.get(x)[x]
        except BaseException as e:
            import traceback

            print(f"Error loading mesh for {x}:")
            traceback.print_exc()
            return e

        return oc.visuals.mesh2gfx(m, **kwargs)

    def check_futures(self):
        """Check if any futures are done."""
        # Keep track of whether we had any futures at the beginning
        has_futures = len(self.futures) > 0

        for (id, name), future in self.futures.items():
            if not future.done():
                continue
            visual = future.result()

            # If there is no mesh, skip
            if isinstance(visual, BaseException):
                self.n_failed += 1
                self.report(f"Failed to load {id}: {visual}", flush=True)
                continue

            self.report(f"Loaded {id}", flush=True)
            self.viewer.add(visual, name=str(name), center=False)
            self._segments[id] = visual

            # Center on the first neuron
            if not self._centered:
                self.viewer.center_camera()
                self._centered = True

            # Populate cache if necessary
            if self.use_cache:
                self.report(f"Caching visual for {id}", flush=True)
                self.cache[id] = visual

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
