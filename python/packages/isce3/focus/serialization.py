from dataclasses import dataclass
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

@dataclass(frozen=True)
class NonUniformFFTParameters:
    zero_padding_factor: float = 2.0
    kernel_halfwidth: int = 2

    def __post_init__(self):
        if self.zero_padding_factor <= 1.0:
            raise ValueError("require NFFT zero_padding_factor > 1.0")
        if self.kernel_halfwidth < 1:
            raise ValueError("require NFFT kernel_halfwidth >= 1")

    @classmethod
    def from_dict(cls, d: dict):
        default = cls()
        return cls(
            float(d.get("zero_padding_factor", default.zero_padding_factor)),
            int(d.get("kernel_halfwidth", default.kernel_halfwidth)))

@dataclass(frozen=True)
class NonUniformFFT2DParameters:
    range: NonUniformFFTParameters = NonUniformFFTParameters()
    azimuth: NonUniformFFTParameters = NonUniformFFTParameters()

    @classmethod
    def from_dict(cls, d: dict):
        default = cls()
        T = NonUniformFFTParameters
        rg = T.from_dict(d["range"]) if "range" in d else default.range
        az = T.from_dict(d["azimuth"]) if "azimuth" in d else default.azimuth
        return cls(rg, az)

@dataclass(frozen=True)
class BackprojectionStageParameters:
    size: int = 1
    oversample_range: float = 1.2
    oversample_azimuth: float = 1.2
    interpolation: NonUniformFFT2DParameters = NonUniformFFT2DParameters()

    def __post_init__(self):
        if self.size < 1:
            raise ValueError("require at least 1 pulse/subaperture per stage")
        if self.oversample_range < 1.0:
            raise ValueError("must sample range at or above Nyquist limit")
        if self.oversample_azimuth < 1.0:
            raise ValueError("must sample azimuth at or above Nyquist limit")

    @classmethod
    def from_dict(cls, d: dict):
        default = cls()
        key, T = "interpolation", NonUniformFFT2DParameters
        return cls(
            d.get("size", default.size),
            d.get("oversample_range", default.oversample_range),
            d.get("oversample_azimuth", default.oversample_azimuth),
            T.from_dict(d[key]) if key in d else default.interpolation)