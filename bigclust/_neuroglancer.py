import cmap
import requests

import octarine as oc
import cloudvolume as cv
import dvid as dv

import navis

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

    def __init__(self, data, neuropil_mesh=None, max_threads=20):
        self.data = data.rename({"segment_id": "id"}, axis=1).astype({"id": int})

        self.set_default_colors()

        # Parse cloudvolumes
        self.volumes = {}
        for url in self.data.source.unique():
            self.volumes[url] = self.make_volume(url)

        self.viewer = oc.Viewer()
        self._centered = False

        self._neuropil_mesh = neuropil_mesh
        if neuropil_mesh:
            self.viewer.add_mesh(
                neuropil_mesh, color=(0.8, 0.8, 0.8, 0.05), name="neuropil"
            )
            self._centered = True

        # Holds the futures for requested data
        self.futures = {}
        self.pool = ThreadPoolExecutor(max_threads)

        # Tracks which neurons we've already loaded
        self._segments = set()

        self.register()

    def __del__(self):
        if hasattr(self, "pool"):
            self.pool.shutdown(cancel_futures=True, wait=True)
        if hasattr(self, "viewer"):
            self.unregister()

    def set_default_colors(self):
        # Check for color column
        if "color" in self.data.columns:
            self._colors = self.data.set_index("id").color.to_dict()
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
        self.viewer.add_animation(self.check_futures)

    def unregister(self):
        self.viewer.remove_animation(self.check_futures)

    def show(self, indices, lod=-1, add_as_group=False, **kwargs):
        """Add data to the viewer.

        Parameters
        ----------
        indices :   iterable
                    The indices of neurons to show.
        kwargs :    dict
                    Keyword arguments to pass to the viewer.

        """
        if not len(indices):
            self.clear()
            return

        to_show = self.data.iloc[indices]
        # print(f"Showing {len(to_show)} neurons: ", to_show.id.values.tolist(), flush=True)

        # Remove those segments we don't want
        to_remove = [str(x) for x in (self._segments - set(to_show.id.values))]
        self.viewer.remove_objects(to_remove)
        # print(f"Removing {len(to_remove)} neurons: ", to_remove, flush=True)

        # Cancel all futures we don't need anymore
        # Note: we need to use list() because we're potentially
        # modifying the dict inside the loop
        for id, future in list(self.futures.items()):
            if id not in to_show.id.values:
                # print('Cancelling futur for', id, flush=True)

                # Cancel future
                future.cancel()

                # Remove from futures
                self.futures.pop(id, None)

        # Now drop those already on display
        to_show = to_show[~to_show.id.isin(self._segments)]

        for _, row in to_show.iterrows():
            # print("Loading", row.id, flush=True)

            # Skip if we're already loading this segment
            if row.id in self.futures:
                continue

            self.futures[row.id] = self.pool.submit(
                self._load_mesh,
                row.id,
                self.volumes[row.source],
                lod=lod,
                #add_as_group=add_as_group,
                **kwargs,
            )
            self._segments.add(row.id)

    def clear(self):
        """Clear the viewer of selected segments."""
        self.viewer.remove_objects([str(x) for x in self._segments])
        self._segments.clear()

        for future in self.futures.values():
            future.cancel()

        self.futures.clear()

    def _load_mesh(self, x, vol, lod=-1, **kwargs):
        """Load a single mesh."""
        x = int(x)
        try:
            m = vol.mesh.get(x, lod=lod)[x]
        except:
            return None

        if "color" not in kwargs:
            if int(x) in self._colors:
                kwargs["color"] = self._colors[int(x)]
            else:
                kwargs["color"] = self.viewer._next_color()

        return oc.visuals.mesh2gfx(m, **kwargs)

    def check_futures(self):
        """Check if any futures are done."""
        has_futures = len(self.futures) > 0

        for id, future in self.futures.items():
            if not future.done():
                continue
            data = future.result()

            # If there is no mesh, skip
            if data is None:
                continue

            self.viewer.add(data, name=str(id), center=False)
            self._segments.add(id)

            # Center on the first neuron
            if not self._centered:
                self.viewer.center_camera()
                self._centered = True

        # Show progress message
        if has_futures and len(self.futures) > 0:
            self.viewer.show_message(
                f"Loading {len(self.futures):,} neurons",
                duration=2,
                position="top-right",
                color="w",
            )

        # Remove completed futures
        self.futures = {k: v for k, v in self.futures.items() if not v.done()}

        # If all futures completed
        if has_futures and len(self.futures) == 0:
            self.viewer.show_message(
                "All neurons loaded", duration=2, position="top-right", color="w"
            )


class DVIDVolume:
    """Helper class for loading DVID neurons."""

    def __init__(self, url):
        url = url.replace("dvid://", "")

        # Remove any parameters
        self.url = url.split("?")[0]

        self.server, self.node = self.url.replace("https://", "").split("/")[:2]
        self.server = f"https://{self.server}"

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
