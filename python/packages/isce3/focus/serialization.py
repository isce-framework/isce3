import h5py
from isce3.core import Linspace, LookSide
from isce3.focus import PolarGrid
import numpy as np

# FIXME Not sure where to put this stuff.  Maybe monkey patch the classes?

def overwrite(group: h5py.Group, key: str, value):
    if key in group:
        del group[key]
    group.create_dataset(key, data=value)


def save_linspace_to_h5(x: Linspace, group: h5py.Group):
    for key in ("first", "spacing", "size"):
        val = getattr(x, key)
        overwrite(group, key, val)


def save_polar_grid_to_h5(grid: PolarGrid, group: h5py.Group):
    for key in ("aztime_start", "aztime_end", "origin", "axis"):
        val = getattr(grid, key)
        overwrite(group, key, val)
    for key in ("range", "sin_squint"):
        g = group.require_group(key)
        val = getattr(grid, key)
        save_linspace_to_h5(val, g)
    side_str = str(grid.look_side).split(".")[1]
    overwrite(group, "look_side", np.bytes_(side_str))


def save_polar_image_to_h5(z: np.ndarray, grid: PolarGrid, group: h5py.Group):
    overwrite(group, "image", z)
    g = group.require_group("polar_grid")
    save_polar_grid_to_h5(grid, g)