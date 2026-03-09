# network/zarr_utils.py
import os
import zarr


def open_zarr_group(path: str) -> zarr.Group:
    if not os.path.exists(path):
        raise FileNotFoundError(f"zarr path not found: {path}")
    return zarr.open_group(path, mode="r")

def open_zarr_array(path: str) -> zarr.Array:
    if not os.path.exists(path):
        raise FileNotFoundError(f"zarr path not found: {path}")
    return zarr.open_array(path, mode="r")


def get_transitions_dataset(g: zarr.Group) -> zarr.Array:
    if "transitions" in g:
        return g["transitions"]
    # fallback 1-level deep
    for k in list(g.group_keys()):
        sub = g[k]
        if isinstance(sub, zarr.Group) and "transitions" in sub:
            return sub["transitions"]
    raise KeyError("could not find 'transitions' dataset in the zarr group")