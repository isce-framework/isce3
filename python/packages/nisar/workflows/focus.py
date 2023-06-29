#!/usr/bin/env python3
from bisect import bisect_left, bisect_right
from functools import reduce
import h5py
import json
import logging
import os
from nisar.mixed_mode import (PolChannel, PolChannelSet, Band,
    find_overlapping_channel)
from nisar.products.readers.Raw import Raw, open_rrsd
from nisar.products.writers import SLC
from nisar.types import to_complex32, read_c4_dataset_as_c8
import nisar
import numpy as np
import isce3
from isce3.core import DateTime, LUT2d, Attitude, Orbit
from isce3.io.gdal import Raster, GDT_CFloat32
from isce3.product import RadarGridParameters
from nisar.workflows.yaml_argparse import YamlArgparse
import nisar.workflows.helpers as helpers
from ruamel.yaml import YAML
import sys
import tempfile
from typing import List, Union, Optional, Callable, Iterable, Dict, Tuple
from isce3.io import Raster as RasterIO


# TODO some CSV logger
log = logging.getLogger("focus")

# https://stackoverflow.com/a/6993694/112699
class Struct(object):
    "Convert nested dict to object, assuming keys are valid identifiers."
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


def load_config(yaml):
    "Load default runconfig, override with user input, and convert to Struct"
    parser = YAML(typ='safe')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg = parser.load(open(f'{dir_path}/defaults/focus.yaml', 'r'))
    with open(yaml) as f:
        user = parser.load(f)
    helpers.deep_update(cfg, user)
    return Struct(cfg)


def dump_config(cfg: Struct, filename):
    def struct2dict(s: Struct):
        d = s.__dict__.copy()
        for k in d:
            if isinstance(d[k], Struct):
                d[k] = struct2dict(d[k])
        return d
    parser = YAML()
    parser.indent = 4
    with open(filename, 'w') as f:
        d = struct2dict(cfg)
        parser.dump(d, f)


def validate_config(x):
    # TODO
    log.warning("Skipping input validation.")
    return x


def cosine_window(n: int, pedestal: float):
    if not (0.0 <= pedestal <= 1.0):
        raise ValueError(f"Expected pedestal between 0 and 1, got {pedestal}.")
    b = 0.5 * (1 - pedestal)
    t = np.arange(n) - (n - 1) / 2.0
    return (1 - b) + b * np.cos(2 * np.pi * t / (n - 1))


def apply_window(kind: str, shape: float, z: np.ndarray) -> np.ndarray:
    """Return product of an array with a window of the same length.
    """
    n = len(z)
    if kind == 'kaiser':
        return np.kaiser(n, shape) * z
    elif kind == 'cosine':
        return cosine_window(n, shape) * z
    raise NotImplementedError(f"window {kind} not in (Kaiser, Cosine).")


def check_window_input(win: Struct, msg='') -> Tuple[str, float]:
    """Check user input window kind and shape, log both, and
    return (kind, shape).
    """
    kind = win.kind.lower()
    if kind == 'kaiser':
        log.info(f"{msg}Kaiser(beta={win.shape})")
        if win.shape < 0.0:
            raise ValueError("Require positive Kaiser window shape")
    elif kind == 'cosine':
        log.info(f'{msg}Cosine(pedestal_height={win.shape})')
        if not (0.0 <= win.shape <= 1.0):
            raise ValueError("Require Cosine window parameter in [0, 1]")
    else:
        raise NotImplementedError(
            f"window '{kind}' not in ('Kaiser', 'Cosine').")
    return kind, win.shape


def get_max_chirp_duration(cfg: Struct):
    """Return maximum chirp duration (in seconds) among all sidebands,
    polarizations, and files referenced in the runconfig.
    """
    maxlen = 0.0
    for filename in cfg.input_file_group.input_file_path:
        raw = open_rrsd(filename)
        for freq, polarizations in raw.polarizations.items():
            for pol in polarizations:
                _, _, _, T = raw.getChirpParameters(freq, pol[0])
                maxlen = max(maxlen, T)
    return maxlen


def parse_rangecomp_mode(mode: str):
    lut = {"full": isce3.focus.RangeComp.Mode.Full,
           "same": isce3.focus.RangeComp.Mode.Same,
           "valid": isce3.focus.RangeComp.Mode.Valid}
    mode = mode.lower()
    if mode not in lut:
        raise ValueError(f"Invalid RangeComp mode {mode}")
    return lut[mode]


def get_orbit(cfg: Struct):
    xml = cfg.dynamic_ancillary_file_group.orbit
    if xml is not None:
        log.info("Loading orbit from external XML file.")
        return nisar.products.readers.orbit.load_orbit_from_xml(xml)
    log.info("Loading orbit from L0B file.")
    rawfiles = cfg.input_file_group.input_file_path
    if len(rawfiles) > 1:
        raise NotImplementedError("Can't concatenate orbit data.")
    raw = open_rrsd(rawfiles[0])
    return raw.getOrbit()


def get_attitude(cfg: Struct):
    xml = cfg.dynamic_ancillary_file_group.pointing
    if xml is not None:
        log.info("Loading attitude from external XML file")
        return nisar.products.readers.attitude.load_attitude_from_xml(xml)
    log.info("Loading attitude from L0B file.")
    rawfiles = cfg.input_file_group.input_file_path
    if len(rawfiles) > 1:
        raise NotImplementedError("Can't concatenate attitude data.")
    raw = open_rrsd(rawfiles[0])
    return raw.getAttitude()


def get_total_grid_bounds(rawfiles: List[str]):
    times, ranges = [], []
    for fn in rawfiles:
        raw = open_rrsd(fn)
        for frequency, pols in raw.polarizations.items():
            for pol in pols:
                ranges.append(raw.getRanges(frequency, tx=pol[0]))
                times.append(raw.getPulseTimes(frequency, tx=pol[0]))
    rmin = min(r[0] for r in ranges)
    rmax = max(r[-1] for r in ranges)
    dtmin = min(epoch + isce3.core.TimeDelta(t[0]) for (epoch, t) in times)
    dtmax = max(epoch + isce3.core.TimeDelta(t[-1]) for (epoch, t) in times)
    epoch = min(epoch for (epoch, t) in times)
    tmin = (dtmin - epoch).total_seconds()
    tmax = (dtmax - epoch).total_seconds()
    return epoch, tmin, tmax, rmin, rmax


def get_total_grid(rawfiles: List[str], dt, dr):
    epoch, tmin, tmax, rmin, rmax = get_total_grid_bounds(rawfiles)
    nt = int(np.ceil((tmax - tmin) / dt))
    nr = int(np.ceil((rmax - rmin) / dr))
    t = isce3.core.Linspace(tmin, dt, nt)
    r = isce3.core.Linspace(rmin, dr, nr)
    return epoch, t, r


def squint(t, r, orbit, attitude, side, angle=0.0, dem=None, **kw):
    """Find squint angle given imaging time and range to target.
    """
    assert orbit.reference_epoch == attitude.reference_epoch
    p, v = orbit.interpolate(t)
    R = attitude.interpolate(t).to_rotation_matrix()
    axis = R[:,1]
    # In NISAR coordinate frames (see D-80882 and REE User Guide) left/right is
    # implemented as a 180 yaw flip, so the left/right flag can just be
    # inferred by the sign of axis.dot(v). Verify this convention.
    inferred_side = "left" if axis.dot(v) > 0 else "right"
    if side.lower() != inferred_side:
        raise ValueError(f"Requested side={side.lower()} but "
                         f"inferred side={inferred_side} based on orientation "
                         f"(Y_RCS.dot(V) = {axis.dot(v)})")
    if dem is None:
        dem = isce3.geometry.DEMInterpolator()
    # NOTE Here "left" means an acute, positive look angle by right-handed
    # rotation about `axis`.  Since axis will flip sign, always use "left" to
    # get the requested side in the sense of velocity vector.
    xyz = isce3.geometry.rdr2geo_cone(p, axis, angle, r, dem, "left", **kw)
    look = (xyz - p) / np.linalg.norm(xyz - p)
    vhat = v / np.linalg.norm(v)
    return np.arcsin(look.dot(vhat))


def squint_to_doppler(squint, wvl, vmag):
    return 2.0 / wvl * vmag * np.sin(squint)


def convert_epoch(t: List[float], epoch_in, epoch_out):
    TD = isce3.core.TimeDelta
    return [(epoch_in - epoch_out + TD(ti)).total_seconds() for ti in t]


def get_dem(cfg: Struct):
    dem = isce3.geometry.DEMInterpolator(
        height=cfg.processing.dem.reference_height,
        method=cfg.processing.dem.interp_method)
    fn = cfg.dynamic_ancillary_file_group.dem_file
    if fn:
        log.info(f"Loading DEM {fn}")
        log.info("Out-of-bound DEM values will be set to "
                 f"{cfg.processing.dem.reference_height} (m).")
        dem.load_dem(RasterIO(fn))
    else:
        log.warning("No DEM given, using ref height "
                    f"{cfg.processing.dem.reference_height} (m).")
    return dem


def make_doppler_lut(rawfiles: List[str],
        az: float = 0.0,
        orbit: Optional[isce3.core.Orbit] = None,
        attitude: Optional[isce3.core.Attitude] = None,
        dem: Optional[isce3.geometry.DEMInterpolator] = None,
        azimuth_spacing: float = 1.0,
        range_spacing: float = 1e3,
        interp_method: str = "bilinear",
        epoch: Optional[DateTime] = None,
        **rdr2geo):
    """Generate Doppler look up table (LUT).

    Parameters
    ----------
    rawfiles
        List of NISAR L0B format raw data files.
    az : optional
        Complement of the angle between the along-track axis of the antenna and
        its electrical boresight, in radians.  Zero for non-scanned, flush-
        mounted antennas like ALOS-1.
    orbit : optional
        Path of antenna phase center.  Defaults to orbit in first L0B file.
    attitude : optional
        Orientation of antenna.  Defaults to attitude in first L0B file.
    dem : optional
        Digital elevation model, height in m above WGS84 ellipsoid. Default=0 m.
    azimuth_spacing : optional
        LUT grid spacing in azimuth, in seconds.  Default=1 s.
    range_spacing : optional
        LUT grid spacing in range, in meters.  Default=1000 m.
    interp_method : optional
        LUT interpolation method. Default="bilinear".
    epoch : isce3.core.DateTime, optional
        Time reference for output table.  Defaults to orbit.reference_epoch
    threshold : optional
    maxiter : optional
    extraiter : optional
        See rdr2geo

    Returns
    -------
    fc
        Center frequency, in Hz, assumed for Doppler calculation.
        It is among those found in the input raw data files.
    LUT
        Look up table of Doppler = f(r,t)
    """
    # Input wrangling.
    assert len(rawfiles) > 0, "Need at least one L0B file."
    assert (azimuth_spacing > 0.0) and (range_spacing > 0.0)
    raw = open_rrsd(rawfiles[0])
    if orbit is None:
        orbit = raw.getOrbit()
    if attitude is None:
        attitude = raw.getAttitude()
    if dem is None:
        dem = isce3.geometry.DEMInterpolator()
    if epoch is None:
        epoch = orbit.reference_epoch
    # Ensure consistent time reference (while avoiding side effects).
    if orbit.reference_epoch != epoch:
        orbit = orbit.copy()
        orbit.update_reference_epoch(epoch)
    if attitude.reference_epoch != epoch:
        attitude = attitude.copy()
        attitude.update_reference_epoch(epoch)

    side = require_constant_look_side(open_rrsd(fn) for fn in rawfiles)
    # Use a nominal center frequency, which we'll return for user reference.
    frequency = next(iter(raw.polarizations))
    fc = raw.getCenterFrequency(frequency)

    # Now do the actual calculations.
    wvl = isce3.core.speed_of_light / fc
    epoch_in, t, r = get_total_grid(rawfiles, azimuth_spacing, range_spacing)
    t = convert_epoch(t, epoch_in, epoch)
    dop = np.zeros((len(t), len(r)))
    for i, ti in enumerate(t):
        _, v = orbit.interpolate(ti)
        vi = np.linalg.norm(v)
        for j, rj in enumerate(r):
            sq = squint(ti, rj, orbit, attitude, side, angle=az, dem=dem,
                        **rdr2geo)
            dop[i,j] = squint_to_doppler(sq, wvl, vi)
    lut = LUT2d(np.asarray(r), t, dop, interp_method, False)
    return fc, lut


def make_doppler(cfg: Struct, epoch: Optional[DateTime] = None):
    log.info("Generating Doppler LUT from pointing")
    orbit = get_orbit(cfg)
    attitude = get_attitude(cfg)
    dem = get_dem(cfg)
    opt = cfg.processing.doppler
    az = np.radians(opt.azimuth_boresight_deg)
    rawfiles = cfg.input_file_group.input_file_path

    fc, lut = make_doppler_lut(rawfiles,
                               az=az, orbit=orbit, attitude=attitude,
                               dem=dem, azimuth_spacing=opt.spacing.azimuth,
                               range_spacing=opt.spacing.range,
                               interp_method=opt.interp_method,  epoch=epoch,
                               **vars(opt.rdr2geo))

    log.info(f"Made Doppler LUT for fc={fc} Hz with mean={lut.data.mean()} Hz")
    return fc, lut


def zero_doppler_like(dop: LUT2d):
    x = np.zeros_like(dop.data)
    # Assume we don't care about interp method or bounds when all values == 0.
    method, check_bounds = "nearest", False
    return LUT2d(dop.x_start, dop.y_start, dop.x_spacing, dop.y_spacing, x,
                 method, check_bounds)


def scale_doppler(dop: LUT2d, c: float):
    if dop.have_data:
        x = c * dop.data
        return LUT2d(dop.x_start, dop.y_start, dop.x_spacing, dop.y_spacing, x,
                    dop.interp_method, dop.bounds_error)
    if dop.ref_value == 0.0:
        return LUT2d()
    raise NotImplementedError("No way to scale Doppler with nonzero ref_value")


def make_output_grid(cfg: Struct,
                     epoch: DateTime, t0: float, t1: float, max_prf: float,
                     r0: float, r1: float, dr: float,
                     side: Union[str, isce3.core.LookSide],
                     orbit: Orbit,
                     fc_ref: float, doppler: LUT2d,
                     chirplen_meters: float,
                     dem: isce3.geometry.DEMInterpolator) -> RadarGridParameters:
    """
    Given the available raw data extent (in slow time and slant range) figure
    out a reasonable output extent that:
        * Accounts for the reskew to zero-Doppler,
        * Takes care not to exceed the available ephemeris data, and
        * Excludes the coherent processing interval in range and azimuth.
    These extents are then overridden by any user-provided runconfig data and
    used to construct a RadarGridParameters object suitable for focusing.

    Parameters
    ----------
    cfg : Struct
        RSLC runconfig data.
    epoch : DateTime
        Reference for all time tags.
    t0 : float
        First pulse time available in all raw data, in seconds since epoch.
    t1 : float
        Last pulse time available in all raw data, in seconds since epoch.
    max_prf : float
        The highest PRF used in all raw data, in Hz.
    r0 : float
        Minimum one-way range among all raw data, in meters.
    r1 : float
        Maximum one-way range among all raw data, in meters.
    dr : float
        Desired range spacing of output grid, in meters.
    side : {"left", "right"} or isce3.core.LookSide
        Radar look direction
    orbit : Orbit
        Radar orbit
    fc_ref : float
        Radar center frequency corresponding to `doppler` object.  Also
        used to determine CPI and populate wavelength in output grid object.
    doppler : LUT2d
        Doppler centroid of the raw data.
    chirplen_meters : float
        Maximum chirp length among raw data, expressed as range in meters.
        Used to help determine the region that can be fully focused.
    dem : isce3.geometry.DEMInterpolator
        Digital elevation model containing height above WGS84 ellipsoid,
        in meters.

    Returns
    -------
    grid : RadarGridParameters
        Zero-Doppler grid suitable for focusing.
    """
    assert orbit.reference_epoch == epoch
    ac = cfg.processing.azcomp
    wavelength = isce3.core.speed_of_light / fc_ref

    # Calc approx synthetic aperture duration (coherent processing interval).
    tmid = 0.5 * (t0 + t1)
    cpi = isce3.focus.get_sar_duration(tmid, r1, orbit, isce3.core.Ellipsoid(),
                                       ac.azimuth_resolution, wavelength)
    log.debug(f"Approximate synthetic aperture duration is {cpi} s.")

    # Crop to fully focused region, ignoring range-dependence of CPI.
    # Ignore sampling of convolution kernels, accepting possibility of a few
    # pixels that are only 99.9% focused.
    # CPI is symmetric about Doppler centroid.
    t0 = t0 + cpi / 2
    t1 = t1 - cpi / 2
    # Range delay is defined relative to _start_ of TX pulse.
    r1 = r1 - chirplen_meters

    # Output grid is zero Doppler, so reskew the four corners and assume they
    # enclose the image.  Take extrema as default processing box.
    # Define a capture to save some typing
    zerodop = isce3.core.LUT2d()
    def reskew_to_zerodop(t, r):
        return isce3.geometry.rdr2rdr(t, r, orbit, side, doppler, wavelength,
            dem, doppler_out=zerodop,
            rdr2geo_params=vars(ac.rdr2geo), geo2rdr_params=vars(ac.geo2rdr))

    # One annoying case is where the orbit data covers the raw pulse times
    # and nothing else.  The code can crash when trying to compute positions on
    # the output zero-Doppler grid because the reskew time offset causes it to
    # run off the end of the available orbit data.  Also the Newton solvers need
    # some wiggle room. As a workaround, let's nudge the default bounds until
    # we're sure the code doesn't crash.
    def reskew_near_far_with_nudge(t, r0, r1, step, tstop):
        assert (tstop - t) * step > 0, "Sign of step must bring t towards tstop"
        offset = 0.0
        # Give up when we've nudged t past tstop (nudging forwards or
        # backwards).
        while (tstop - (t + offset)) * step > 0:
            try:
                ta, ra = reskew_to_zerodop(t + offset, r0)
                tb, rb = reskew_to_zerodop(t + offset, r1)
                return offset, ta, ra, tb, rb
            except RuntimeError:
                offset += step
        raise RuntimeError("No valid geometry.  Invalid orbit data?")

    # Solve for points at near range (a) and far range (b) at start time.
    offset0, ta, ra, tb, rb = reskew_near_far_with_nudge(t0, r0, r1, 0.1, t1)
    log.debug(f"offset0 = {offset0}")
    if abs(offset0) > 0.0:
        log.warning(f"Losing up to {offset0} seconds of image data at start due"
            " to insufficient orbit data.")
    # Solve for points at near range (c) and far range (d) at end time.
    offset1, tc, rc, td, rd = reskew_near_far_with_nudge(t1, r0, r1, -0.1, t0)
    log.debug(f"offset1 = {offset1}")
    if abs(offset1) > 0.0:
        log.warning(f"Losing up to {offset1} seconds of image data at end due"
            " to insufficient orbit data.")

    # "z" for zero Doppler.  Reskew varies with range, so take most conservative
    # bounding box to ensure fully focused data everywhere.
    t0z = max(ta, tb)
    r0z = max(ra, rc)
    t1z = min(tc, td)
    r1z = min(rb, rd)
    log.debug(f"Reskew time offset at start {t0z - t0 - offset0} s")
    log.debug(f"Reskew time offset at end {t1z - t1 - offset1} s")
    log.debug(f"Reskew range offset at start {r0z - r0} m")
    log.debug(f"Reskew range offset at end {r1z - r1} m")

    dt0 = epoch + isce3.core.TimeDelta(t0z)
    dt1 = epoch + isce3.core.TimeDelta(t1z)
    log.info(f"Approximate fully focusable time interval is [{dt0}, {dt1}]")
    log.info(f"Approximate fully focusable range interval is [{r0z}, {r1z}]")

    # TODO snap start time to standard interval

    p = cfg.processing.output_grid
    if p.start_time:
        t0z = (DateTime(p.start_time) - epoch).total_seconds()
    if p.end_time:
        t1z = (DateTime(p.end_time) - epoch).total_seconds()
    r0z = p.start_range if (p.start_range is not None) else r0z
    r1z = p.end_range if (p.end_range is not None) else r1z
    # Usually for NISAR grid PRF should be specified as 1520 in the runconfig.
    # If not then take max() as most conservative choice.
    prf = p.output_prf if (p.output_prf is not None) else max_prf

    nr = round((r1z - r0z) / dr)
    nt = round((t1z - t0z) * prf)
    assert (nr > 0) and (nt > 0)
    return RadarGridParameters(t0z, wavelength, prf, r0z, dr, side, nt, nr,
                               epoch)


Selection2d = Tuple[slice, slice]
TimeBounds = Tuple[float, float]
BlockPlan = List[Tuple[Selection2d, TimeBounds]]

def plan_processing_blocks(cfg: Struct, grid: RadarGridParameters,
                           doppler: LUT2d, dem: isce3.geometry.DEMInterpolator,
                           orbit: Orbit, pad: float = 0.1) -> BlockPlan:
    """
    Subdivide output grid into processing blocks and find time bounds of raw
    data needed to focus each one.

    This has the added benefit of coarsely checking that we can calculate the
    geometry over the requested domain.  By default the synthetic aperture
    length is padded by 10% to make the bounds a bit conservative, but this can
    be adjusted via the `pad` parameter.
    """
    assert pad >= 0.0
    ac = cfg.processing.azcomp
    nr = ac.block_size.range
    na = ac.block_size.azimuth

    if nr < 1:
        nr = grid.width

    blocks = []
    for i in range(0, grid.length, na):
        imax = min(i + na, grid.length)
        for j in range(0, grid.width, nr):
            jmax = min(j + nr, grid.width)
            blocks.append(np.s_[i:imax, j:jmax])

    zerodop = isce3.core.LUT2d()
    results = []
    for rows, cols in blocks:
        raw_times = []
        # Compute zero-to-native Doppler reskew times for four corners.
        # NOTE if this eats up a lot of time it can be sped up 4x by
        # computing each vertex and later associating them with blocks.
        for (u, v) in ((rows.start, cols.start), (rows.stop-1, cols.start),
                       (rows.start, cols.stop-1), (rows.stop-1, cols.stop-1)):
            t = grid.sensing_start + u / grid.prf
            r = grid.starting_range + v * grid.range_pixel_spacing
            try:
                traw, _ = isce3.geometry.rdr2rdr(t, r, orbit, grid.lookside,
                    zerodop, grid.wavelength, dem, doppler_out=doppler,
                    rdr2geo_params=vars(ac.rdr2geo),
                    geo2rdr_params=vars(ac.geo2rdr))
            except RuntimeError as e:
                dt = epoch + isce3.core.TimeDelta(t)
                log.error(f"Reskew zero-to-native failed at t={dt} r={r}")
                raise RuntimeError("Could not compute imaging geometry") from e
            raw_times.append(traw)
        sub_grid = grid[rows, cols]
        cpi = isce3.focus.get_sar_duration(sub_grid.sensing_mid,
                                    sub_grid.end_range, orbit,
                                    isce3.core.Ellipsoid(),
                                    ac.azimuth_resolution, sub_grid.wavelength)
        cpi *= 1.0 + pad
        raw_begin = min(raw_times) - cpi / 2
        raw_end = max(raw_times) + cpi / 2
        results.append(((rows, cols), (raw_begin, raw_end)))
    return results


def total_bounds(blocks_bounds: BlockPlan) -> TimeBounds:
    begin = min(t0 for _, (t0, t1) in blocks_bounds)
    end = max(t1 for _, (t0, t1) in blocks_bounds)
    return (begin, end)


def is_overlapping(a, b, c, d):
    assert (b >= a) and (d >= c)
    return (d >= a) and (c <= b)

def get_kernel(cfg: Struct):
    # TODO
    opt = cfg.processing.azcomp.kernel
    if opt.type.lower() != 'knab':
        raise NotImplementedError("Only Knab kernel implemented.")
    n = 1 + 2 * opt.halfwidth
    kernel = isce3.core.KnabKernel(n, 1 / 1.2)
    assert opt.fit.lower() == "table"
    table = isce3.core.TabulatedKernelF32(kernel, opt.fit_order)
    return table


def verify_uniform_pri(t: np.ndarray, atol=0.0, rtol=0.001):
    """Return True if pulse interval is uniform (within given absolute and
    relative tolerances).
    """
    assert atol >= 0.0
    assert rtol >= 0.0
    dt = np.diff(t)
    pri = np.mean(dt)
    return np.allclose(dt, pri, atol=atol, rtol=rtol)


# Work around for fact that slices are not hashable and can't be used as
# dictionary keys or entries in sets
# https://bugs.python.org/issue1733184
def unpack_slices(slices: Tuple[slice, slice]):
    rows, cols = slices
    return ((rows.start, rows.stop, rows.step),
            (cols.start, cols.stop, cols.step))


class BackgroundWriter(isce3.io.BackgroundWriter):
    """
    Compute statistics and write RSLC data in a background thread.

    Parameters
    ----------
    range_cor : np.ndarray
        1D range correction to apply to data before writing, e.g., phasors that
        shift the frequency to baseband.  Length should match number of columns
        (range bins) in `dset`.
    dset : h5py.Dataset
        HDF5 dataset to write blocks to.  Shape should be 2D (azimuth, range).
        Data type is complex32 (pairs of float16).

    Attributes
    ----------
    stats : isce3.math.StatsRealImagFloat32
        Statistics of all data that have been written.
    """
    def __init__(self, range_cor, dset, **kw):
        self.range_cor = range_cor
        self.dset = dset
        # Keep track of which blocks have been written and the image stats
        # for each one.
        self._visited_blocks = dict()
        super().__init__(**kw)

    @property
    def stats(self):
        total_stats = isce3.math.StatsRealImagFloat32()
        for block_stats in self._visited_blocks.values():
            total_stats.update(block_stats)
        return total_stats

    def write(self, z, block):
        """
        Scale `z` by `range_cor` (in-place) then write to file and accumulate
        statistics.  If the block has been written already, then the current
        data will be added to the existing results (dset[block] += ...).

        Parameters
        ----------
        z : np.ndarray
            An arbitrary 2D chunk of data to store in `dset`.
        block : tuple[slice]
            Pair of slices describing the (azimuth, range) selection of `dset`
            where the chunk should be stored.

        Notes
        -----
        For efficiency, no check is made for partially overlapping block
        selections.  In those cases data will be written directly without
        accumulating previous results.
        """
        # scale and deramp
        z *= self.range_cor[None, block[1]]
        # Expect in mixed-mode cases that each file will contribute partially
        # focused blocks at mode-change boundaries.  In that case accumulate
        # data, but avoid slow reads if possible by keeping track of which
        # blocks we've already visited.
        # XXX Slices aren't hashable, so convert them to a type that is so that
        # we can still do O(1) lookups.
        key = unpack_slices(block)
        if key in self._visited_blocks:
            log.debug("reading back SLC data at block %s", block)
            z += read_c4_dataset_as_c8(self.dset, block)
        # Calculate block stats.  Don't accumulate since image is mutable in
        # mixed-mode case.
        s = isce3.math.StatsRealImagFloat32(z)
        amax = np.max(np.abs([s.real.max, s.real.min, s.imag.max, s.imag.min]))
        log.debug(f"scaled max component = {amax}")
        self._visited_blocks[key] = s
        # convert to float16 and write to HDF5
        zf = to_complex32(z)
        self.dset.write_direct(zf, dest_sel=block)


def resample(raw: np.ndarray, t: np.ndarray,
             grid: RadarGridParameters, swaths: np.ndarray,
             orbit: isce3.core.Orbit, doppler: isce3.core.LUT2d, L=12.0,
             fn="regridded.c8"):
    """
    Fill gaps and resample raw data to uniform grid using BLU method.

    Parameters
    ----------
    raw : array-like [complex float32, rows=pulses, cols=range bins]
        Decoded raw data.
    t : np.ndarray [float64]
        Pulse times (seconds since orbit/grid epoch).
    grid : isce3.product.RadarGridParameters
        Raw data grid.  Output will have same parameters.
    swaths : np.ndarray [int]
        Valid subswath samples, dims = (ns, nt, 2) where ns is the number of
        sub-swaths, nt is the number of pulses, and the trailing dimension is
        the [start, stop) indices of the sub-swath.
    orbit : isce3.core.Orbit
        Orbit.  Used to determine velocity for scaling autocorrelation function.
    doppler : isce3.core.LUT2d [double]
        Raw data Doppler look up table.  Must be valid over entire grid.
    L : float
        Antenna azimuth dimension, in meters.  Used for scaling sinc antenna
        pattern model for azimuth autocorrelation function.
    fn : string, optional
        Filename for output memory map.

    Returns
    -------
    regridded : array-like [complex float32, rows=pulses, cols=range bins]
        Gridded, gap-free raw data.
    """
    assert raw.shape == (grid.length, grid.width)
    assert len(t) == raw.shape[0]
    assert grid.ref_epoch == orbit.reference_epoch
    # Compute uniform time samples for given raw data grid
    out_times = t[0] + np.arange(grid.length) / grid.prf
    # Ranges are the same.
    r = grid.starting_range + grid.range_pixel_spacing * np.arange(grid.width)
    regridded = np.memmap(fn, mode="w+", shape=grid.shape, dtype=np.complex64)
    for i, tout in enumerate(out_times):
        # Get velocity for scaling autocorrelation function.  Won't change much
        # but update every pulse to avoid artifacts across images.
        v = np.linalg.norm(orbit.interpolate(tout)[1])
        acor = isce3.core.AzimuthKernel(L / v)
        # Figure out what pulses are in play by computing weights without mask.
        # TODO All we really need is offset and len(weights)... maybe refactor.
        offset, weights = isce3.focus.get_presum_weights(acor, t, tout)
        nw = len(weights)
        # Compute valid data mask (transposed).
        # NOTE Could store the whole mask instead of recomputing blocks.
        mask = np.zeros((grid.width, nw), dtype=bool)
        for iw in range(nw):
            it = offset + iw
            for swath in swaths:
                start, end = swath[it]
                mask[start:end, iw] = True
        # The pattern of missing samples in any given column can change
        # depending on the gap structure.  Recomputing weights is expensive,
        # though, so compute a hash we can use to cache the unique weight
        # vectors.
        twiddle = 1 << np.arange(nw)
        ids = mask.dot(twiddle)  # NOTE in C++ you'd just OR the bits
        # Compute weights for each unique mask pattern.
        lut = dict()
        for uid in np.unique(ids):
            # Invert the hash to get the mask back
            valid = (uid & twiddle).astype(bool)
            # Pull out valid times for this mask config and compute weights.
            tj = t[offset:offset+nw][valid]
            joff, jwgt = isce3.focus.get_presum_weights(acor, tj, tout)
            assert joff == 0
            # Now insert zeros where data is invalid to get full-length weights.
            jwgt_full = np.zeros_like(weights)
            jwgt_full[valid] = jwgt
            lut[uid] = jwgt_full
        # Fill weights for entire block using look up table.
        w = isce3.focus.fill_weights(ids, lut)
        # Read raw data.
        block = np.s_[offset:offset+nw, :]
        x = raw[block]
        # Compute Doppler deramp.  Zero phase at tout means no need to re-ramp.
        trel = t[offset:offset+nw] - tout
        fd = doppler.eval(tout, r)
        deramp = np.exp(-2j * np.pi * trel[:, None] * fd[None, :])
        # compute weighted sum of deramped pulses.
        regridded[i, :] = (w * deramp * x).sum(axis=0)
    return regridded


def delete_safely(filename):
    # Careful to avoid race on file deletion.  Use pathlib in Python 3.8+
    try:
        os.unlink(filename)
    except FileNotFoundError:
        pass


def get_range_deramp(grid: RadarGridParameters) -> np.ndarray:
    """Compute the phase ramp required to shift a backprojected grid to
    baseband in range.
    """
    r = grid.starting_range + grid.range_pixel_spacing * np.arange(grid.width)
    return np.exp(-1j * 4 * np.pi / grid.wavelength * r)


def require_ephemeris_overlap(ephemeris: Union[Attitude, Orbit],
                              t0: float, t1: float, name: str = "Ephemeris"):
    """Raise exception if ephemeris doesn't fully overlap time interval [t0, t1]
    """
    if ephemeris.contains(t0) and ephemeris.contains(t1):
        return
    dt0 = ephemeris.reference_epoch + isce3.core.TimeDelta(t0)
    dt1 = ephemeris.reference_epoch + isce3.core.TimeDelta(t1)
    msg = (f"{name} time span "
        f"[{ephemeris.start_datetime}, {ephemeris.end_datetime}] does not fully"
        f"overlap required time span [{dt0}, {dt1}]")
    log.error(msg)
    raise ValueError(msg)


def require_frequency_stability(rawlist: Iterable[Raw]) -> None:
    """Check that center frequency doesn't depend on TX polarization since
    this is assumed in RSLC Doppler metadata.
    """
    for raw in rawlist:
        for frequency, polarizations in raw.polarizations.items():
            fc_set = {raw.getCenterFrequency(frequency, pol[0])
                for pol in polarizations}
            if len(fc_set) > 1:
                raise NotImplementedError("TX frequency agility not supported")


def require_constant_look_side(rawlist: Iterable[Raw]) -> str:
    side_set = {raw.identification.lookDirection for raw in rawlist}
    if len(side_set) > 1:
        raise ValueError("Cannot combine left- and right-looking data.")
    return side_set.pop()


def get_common_mode(rawlist: List[Raw]) -> PolChannelSet:
    assert len(rawlist) > 0
    modes = [PolChannelSet.from_raw(raw) for raw in rawlist]
    return reduce(lambda mode1, mode2: mode1.intersection(mode2), modes)


def get_bands(mode: PolChannelSet) -> Dict[str, Band]:
    assert mode == mode.regularized()
    bands = dict()
    for channel in mode:
        bands[channel.freq_id] = channel.band
    return bands


def get_max_prf(rawlist: Iterable[Raw]) -> float:
    """Calculate the average PRF in each Raw file and return the largest one.
    """
    prfs = []
    for raw in rawlist:
        freq, pols = next(iter(raw.polarizations.items()))
        tx = pols[0][0]
        _, grid = raw.getRadarGrid(frequency=freq, tx=tx)
        prfs.append(grid.prf)
    return max(prfs)


def prep_rangecomp(cfg, raw, raw_grid, channel_in, channel_out):
    """Setup range compression.

    Parameters
    ----------
    cfg : Struct
        RSLC runconfig data
    raw : Raw
        NISAR L0B reader object
    raw_grid : RadarGridParameters
        Grid parameters for the raw data that will be compressed
    channel_in : PolChannel
        Input polarimetric channel info
    channel_out : PolChannel
        Output polarimetric channel info, different in mixed-mode case

    Returns
    -------
    rc : RangeComp
        Range compression execution object
    rc_grid : RadarGridParameters
        Grid parameters for the output range-compressed data
    shift : float
        Frequency shift in rad/sample required to shift range-compressed data
        to desired center frequency.
    deramp : np.ndarray[np.complex64]
        Phase array (1D) that can be multiplied into the output data to shift
        to the desired center frequency.  Zero phase is referenced to range=0.
    """
    log.info("Generating chirp")
    tx = channel_in.pol[0]
    chirp = raw.getChirp(channel_in.freq_id, tx)
    log.info(f"Chirp length = {len(chirp)}")
    win_kind, win_shape = check_window_input(cfg.processing.range_window)

    if channel_in.band != channel_out.band:
        log.info("Filtering chirp for mixed-mode processing.")
        fs = raw.getChirpParameters(channel_in.freq_id, tx)[1]
        # NOTE In mixed-mode case window is part of the filter design.
        design = cfg.processing.range_common_band_filter
        cb_filt, shift = nisar.mixed_mode.get_common_band_filter(
                                        channel_in.band, channel_out.band, fs,
                                        attenuation=design.attenuation,
                                        width=design.width,
                                        window=(win_kind, win_shape))
        log.info("Common-band filter length = %d", len(cb_filt))
        if len(cb_filt) > len(chirp):
            log.warning("Common-band filter is longer than chirp!  "
                        "Consider relaxing the filter design parameters.")
        chirp = np.convolve(cb_filt, chirp, mode="full")
    else:
        # In nominal case window the time-domain LFM chirp.
        chirp = apply_window(win_kind, win_shape, chirp)
        cb_filt, shift = [1.0], 0.0

    log.info("Normalizing chirp to unit white noise gain.")
    chirp *= 1.0 / np.linalg.norm(chirp)

    rcmode = parse_rangecomp_mode(cfg.processing.rangecomp.mode)
    log.info(f"Preparing range compressor with mode={rcmode}")
    nr = raw_grid.shape[1]
    na = cfg.processing.rangecomp.block_size.azimuth
    rc = isce3.focus.RangeComp(chirp, nr, maxbatch=na, mode=rcmode)

    # Rangecomp modifies range grid.  Also update wavelength.
    # Careful that common-band filter delay is only half the filter
    # length but RangeComp bookkeeps the entire length of the ref. function.
    rc_grid = raw_grid.copy()
    rc_grid.starting_range -= rc_grid.range_pixel_spacing * (
        rc.first_valid_sample - (len(cb_filt) - 1) / 2)
    rc_grid.width = rc.output_size
    rc_grid.wavelength = isce3.core.speed_of_light / channel_out.band.center

    r = rc_grid.starting_range + (
        rc_grid.range_pixel_spacing * np.arange(rc_grid.width))
    deramp = np.exp(1j * shift / rc_grid.range_pixel_spacing * r)

    return rc, rc_grid, shift, deramp


def focus(runconfig):
    # Strip off two leading namespaces.
    cfg = runconfig.runconfig.groups
    rawnames = cfg.input_file_group.input_file_path
    if len(rawnames) <= 0:
        raise IOError("need at least one raw data file")

    rawlist = [open_rrsd(os.path.abspath(name)) for name in rawnames]
    dem = get_dem(cfg)
    zerodop = isce3.core.LUT2d()
    azres = cfg.processing.azcomp.azimuth_resolution
    atmos = cfg.processing.dry_troposphere_model or "nodelay"
    kernel = get_kernel(cfg)
    scale = cfg.processing.encoding_scale_factor

    require_frequency_stability(rawlist)
    common_mode = get_common_mode(rawlist)
    log.info(f"output mode = {common_mode}")

    use_gpu = isce3.core.gpu_check.use_gpu(cfg.worker.gpu_enabled, cfg.worker.gpu_id)
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg.worker.gpu_id)
        isce3.cuda.core.set_device(device)

        log.info(f"Processing using CUDA device {device.id} ({device.name})")

        backproject = isce3.cuda.focus.backproject
    else:
        backproject = isce3.focus.backproject

    # Generate output grids.
    grid_epoch, t0, t1, r0, r1 = get_total_grid_bounds(rawnames)
    log.info(f"Raw data time spans [{t0}, {t1}] seconds since {grid_epoch}.")
    log.info(f"Raw data range swath spans [{r0}, {r1}] meters.")
    orbit = get_orbit(cfg)
    attitude = get_attitude(cfg)
    # Use same epoch for all objects for consistency and correctness.
    # Especially important since Doppler LUT does not carry its own epoch.
    orbit.update_reference_epoch(grid_epoch)
    attitude.update_reference_epoch(grid_epoch)
    # Need orbit and attitude over whole raw domain in order to generate
    # Doppler LUT.  Check explicitly in order to provide a sensible error.
    log.info("Verifying ephemeris covers time span of raw data.")
    require_ephemeris_overlap(orbit, t0, t1, "Orbit")
    require_ephemeris_overlap(attitude, t0, t1, "Attitude")
    fc_ref, dop_ref = make_doppler(cfg, epoch=grid_epoch)

    max_chirplen = get_max_chirp_duration(cfg) * isce3.core.speed_of_light / 2
    max_bandwidth = max([channel.band.width for channel in common_mode])
    dr = isce3.core.speed_of_light / (2 * 1.2 * max_bandwidth)
    max_prf = get_max_prf(rawlist)
    side = require_constant_look_side(rawlist)
    ref_grid = make_output_grid(cfg, grid_epoch, t0, t1, max_prf, r0, r1, dr,
                                side, orbit, fc_ref, dop_ref, max_chirplen, dem)

    # Frequency A/B specific setup for output grid, doppler, and blocks.
    ogrid, dop, blocks_bounds = dict(), dict(), dict()
    for frequency, band in get_bands(common_mode).items():
        # Ensure aligned grids between A and B by just using an integer skip.
        # Sample rate of A is always an integer multiple of B for NISAR.
        rskip = int(np.round(max_bandwidth / band.width))
        ogrid[frequency] = ref_grid[:, ::rskip]
        ogrid[frequency].wavelength = isce3.core.speed_of_light / band.center
        log.info("Output grid %s is %s", frequency, ogrid[frequency])
        # Doppler depends on center frequency.
        dop[frequency] = scale_doppler(dop_ref, band.center / fc_ref)
        blocks_bounds[frequency] = plan_processing_blocks(cfg, ogrid[frequency],
                                        dop[frequency], dem, orbit)

    proc_begin, proc_end = total_bounds(next(iter(blocks_bounds.values())))
    log.info(f"Need to process raw data time span [{proc_begin}, {proc_end}]"
             f" seconds since {grid_epoch} to produce requested output grid.")

    polygon = isce3.geometry.get_geo_perimeter_wkt(ref_grid, orbit,
                                                   zerodop, dem)

    output_slc_path = os.path.abspath(cfg.product_path_group.sas_output_file)

    output_dir = os.path.dirname(output_slc_path)
    os.makedirs(output_dir, exist_ok=True)

    product = cfg.primary_executable.product_type
    log.info(f"Creating output {product} product {output_slc_path}")
    slc = SLC(output_slc_path, mode="w", product=product)
    slc.set_orbit(orbit) # TODO acceleration, orbitType
    slc.set_attitude(attitude)
    og = next(iter(ogrid.values()))
    slc.copy_identification(rawlist[0], polygon=polygon,
        track=cfg.geometry.relative_orbit_number,
        frame=cfg.geometry.frame_number,
        start_time=og.sensing_datetime(0),
        end_time=og.sensing_datetime(og.length - 1))

    # store metadata for each frequency
    for frequency, band in get_bands(common_mode).items():
        slc.set_parameters(dop[frequency], orbit.reference_epoch, frequency)
        og = ogrid[frequency]
        slc.update_swath(og, orbit, band.width, frequency)

        # add calibration section for each polarization
        pols = [chan.pol for chan in common_mode if chan.freq_id == frequency]
        for pol in pols:
            slc.add_calibration_section(frequency, pol, og.sensing_times,
                                        orbit.reference_epoch, og.slant_ranges)

    freq = next(iter(get_bands(common_mode)))
    slc.set_geolocation_grid(orbit, ogrid[freq], dop[freq],
                             epsg=4326, dem=dem,
                             **vars(cfg.processing.azcomp.geo2rdr))

    # Scratch directory for intermediate outputs
    scratch_dir = os.path.abspath(cfg.product_path_group.scratch_path)
    os.makedirs(scratch_dir, exist_ok=True)

    def temp(suffix):
        return tempfile.NamedTemporaryFile(dir=scratch_dir, suffix=suffix,
            delete=cfg.processing.delete_tempfiles)

    dump_height = (cfg.processing.debug_dump_height and
                   not cfg.processing.delete_tempfiles)

    # main processing loop
    for channel_out in common_mode:
        frequency, pol = channel_out.freq_id, channel_out.pol
        log.info(f"Processing frequency{channel_out.freq_id} {channel_out.pol}")
        acdata = slc.create_image(frequency, pol, shape=ogrid[frequency].shape)
        deramp_ac = get_range_deramp(ogrid[frequency])
        writer = BackgroundWriter(scale * deramp_ac, acdata)

        for raw in rawlist:
            channel_in = find_overlapping_channel(raw, channel_out)
            log.info("Using raw data channel %s", channel_in)

            # NOTE In some cases frequency != channel_in.freq_id, for example
            # 80 MHz (A) being mixed with 5 MHz sideband (B).
            rawdata = raw.getRawDataset(channel_in.freq_id, pol)
            log.info(f"Raw data shape = {rawdata.shape}")
            raw_times, raw_grid = raw.getRadarGrid(channel_in.freq_id,
                                                   tx=pol[0], epoch=grid_epoch)

            pulse_begin = bisect_left(raw_times, proc_begin)
            pulse_end = bisect_right(raw_times, proc_end)
            log.info(f"Using pulses [{pulse_begin}, {pulse_end}]")
            if pulse_begin >= pulse_end:
                log.info("Output does not depend on file %s", raw.filename)
                continue
            raw_times = raw_times[pulse_begin:pulse_end]
            raw_grid = raw_grid[pulse_begin:pulse_end, :]

            na = cfg.processing.rangecomp.block_size.azimuth
            nr = rawdata.shape[1]
            swaths = raw.getSubSwaths(channel_in.freq_id, tx=pol[0])
            log.info(f"Number of sub-swaths = {swaths.shape[0]}")

            rawfd = temp("_raw.c8")
            log.info(f"Decoding raw data to memory map {rawfd.name}.")
            raw_mm = np.memmap(rawfd, mode="w+", shape=raw_grid.shape,
                               dtype=np.complex64)
            for i in range(0, raw_grid.shape[0], na):
                pulse = i + pulse_begin
                nblock = min(na, rawdata.shape[0] - pulse, raw_mm.shape[0] - i)
                block_in = np.s_[pulse:pulse+nblock, :]
                block_out = np.s_[i:i+nblock, :]
                z = rawdata[block_in]
                # Remove NaNs.  TODO could incorporate into gap mask.
                z[np.isnan(z)] = 0.0
                raw_mm[block_out] = z

            if verify_uniform_pri(raw_times):
                log.info("Uniform PRF, using raw data directly.")
                regridded, regridfd = raw_mm, rawfd
            else:
                regridfd = temp("_regrid.c8")
                log.info(f"Resampling non-uniform raw data to {regridfd.name}.")
                regridded = resample(raw_mm, raw_times, raw_grid, swaths, orbit,
                                    dop[frequency], fn=regridfd,
                                    L=cfg.processing.nominal_antenna_size.azimuth)

            del raw_mm, rawfd

            # Do range compression.
            rc, rc_grid, shift, deramp_rc = prep_rangecomp(cfg, raw, raw_grid,
                                        channel_in, channel_out)

            fd = temp("_rc.c8")
            log.info(f"Writing range compressed data to {fd.name}")
            rcfile = Raster(fd.name, rc.output_size, rc_grid.shape[0], GDT_CFloat32)
            log.info(f"Range compressed data shape = {rcfile.data.shape}")

            for pulse in range(0, rc_grid.shape[0], na):
                log.info(f"Range compressing block at pulse {pulse}")
                block = np.s_[pulse:pulse+na, :]
                rc.rangecompress(rcfile.data[block], regridded[block])
                if abs(shift) > 0.0:
                    log.info("Shifting mixed-mode data to baseband")
                    rcfile.data[block] *= deramp_rc[np.newaxis,:]

            del regridded, regridfd

            if dump_height:
                fd_hgt = temp(f"_height_{frequency}{pol}.f4")
                shape = ogrid[frequency].shape
                hgt_mm = np.memmap(fd_hgt, mode="w+", shape=shape, dtype='f4')
                log.debug(f"Dumping height to {fd_hgt.name} with shape {shape}")

            # Do azimuth compression.
            igeom = isce3.container.RadarGeometry(rc_grid, orbit, dop[frequency])

            for block, (t0, t1) in blocks_bounds[frequency]:
                description = f"(i, j) = ({block[0].start}, {block[1].start})"
                if not cfg.processing.is_enabled.azcomp:
                    continue
                if not is_overlapping(t0, t1,
                                    rc_grid.sensing_start, rc_grid.sensing_stop):
                    log.info(f"Skipping inactive azcomp block at {description}")
                    continue
                log.info(f"Azcomp block at {description}")
                bgrid = ogrid[frequency][block]
                ogeom = isce3.container.RadarGeometry(bgrid, orbit, zerodop)
                z = np.zeros(bgrid.shape, 'c8')
                hgt = hgt_mm[block] if dump_height else None
                err = backproject(z, ogeom, rcfile.data, igeom, dem,
                            channel_out.band.center, azres,
                            kernel, atmos, vars(cfg.processing.azcomp.rdr2geo),
                            vars(cfg.processing.azcomp.geo2rdr), height=hgt)
                if err:
                    log.warning("azcomp block contains some invalid pixels")
                writer.queue_write(z, block)

            # Raster/GDAL creates a .hdr file we have to clean up manually.
            hdr = fd.name.replace(".c8", ".hdr")
            if cfg.processing.delete_tempfiles:
                delete_safely(hdr)
            del rcfile

            if dump_height:
                del fd_hgt, hgt_mm

        writer.notify_finished()
        log.info(f"Image statistics {frequency} {pol} = {writer.stats}")
        slc.write_stats(frequency, pol, writer.stats)

    log.info("All done!")


def configure_logging():
    log_level = logging.DEBUG
    log.setLevel(log_level)
    # Format from L0B PGE Design Document, section 9.  Kludging error code.
    msgfmt = ('%(asctime)s.%(msecs)03d, %(levelname)s, RSLC, %(module)s, '
        '999999, %(pathname)s:%(lineno)d, "%(message)s"')
    fmt = logging.Formatter(msgfmt, "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    for friend in ("Raw", "SLCWriter"):
        l = logging.getLogger(friend)
        l.setLevel(log_level)
        l.addHandler(sh)


def main(argv):
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    configure_logging()
    cfg = validate_config(load_config(args.run_config_path))
    echofile = cfg.runconfig.groups.product_path_group.sas_config_file
    if echofile:
        log.info(f"Logging configuration to file {echofile}.")
        dump_config(cfg, echofile)
    focus(cfg)


if __name__ == '__main__':
    main(sys.argv[1:])
