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


def load_linspace_from_h5(group: h5py.Group) -> Linspace:
    args = [group[key][()] for key in ("first", "spacing", "size")]
    return Linspace(*args)


def save_lookside_to_h5(side: LookSide, group: h5py.Group):
    side_str = str(side).split(".")[1]
    overwrite(group, "look_side", np.bytes_(side_str))


def load_lookside_from_h5(group: h5py.Group) -> LookSide:
    side_str_lower = group["look_side"][()].decode("utf-8").lower()
    valid_sides = {"left": LookSide.Left, "right": LookSide.Right}
    return valid_sides[side_str_lower]


def save_polar_grid_to_h5(grid: PolarGrid, group: h5py.Group):
    for key in ("aztime_start", "aztime_end", "origin", "axis"):
        val = getattr(grid, key)
        overwrite(group, key, val)
    for key in ("range", "sin_squint"):
        g = group.require_group(key)
        val = getattr(grid, key)
        save_linspace_to_h5(val, g)
    save_lookside_to_h5(grid.look_side, group)


def load_polar_grid_from_h5(group: h5py.Group) -> PolarGrid:
    keys = ("aztime_start", "aztime_end", "origin", "axis")
    args = [group[key][()] for key in keys]
    args.append(load_linspace_from_h5(group["range"]))
    args.append(load_linspace_from_h5(group["sin_squint"]))
    args.append(load_lookside_from_h5(group))
    return PolarGrid(*args)


def save_polar_image_to_h5(z: np.ndarray, grid: PolarGrid, group: h5py.Group):
    overwrite(group, "image", z)
    g = group.require_group("polar_grid")
    save_polar_grid_to_h5(grid, g)


def load_polar_image_from_h5(group: h5py.Group) -> tuple[np.ndarray, PolarGrid]:
    z = group["image"][:]
    grid = load_polar_grid_from_h5(group["polar_grid"])
    return (z, grid)