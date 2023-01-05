#!/usr/bin/env python3
import h5py
import json
import logging
import os
from nisar.products.readers.Raw import Raw, open_rrsd
from nisar.products.writers import SLC
from nisar.types import to_complex32
import nisar
import numpy as np
import isce3
from isce3.core import DateTime, LUT2d, Attitude, Orbit
from isce3.io.gdal import Raster, GDT_CFloat32
from nisar.workflows.yaml_argparse import YamlArgparse
import nisar.workflows.helpers as helpers
from ruamel.yaml import YAML
import sys
import tempfile
from typing import List, Union, Optional
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


def get_window(win: Struct, msg=''):
    """Return window function f(n) that return a window of length n
    given runconfig group describing the window.
    """
    kind = win.kind.lower()
    if kind == 'kaiser':
        log.info(f"{msg}Kaiser(beta={win.shape})")
        return lambda n: np.kaiser(n, win.shape)
    elif kind == 'cosine':
        log.info(f'{msg}Cosine(pedestal_height={win.shape})')
        return lambda n: cosine_window(n, win.shape)
    raise NotImplementedError(f"window {kind} not in (Kaiser, Cosine).")


def get_chirp(cfg: Struct, raw: Raw, frequency: str, tx: str):
    if cfg.dynamic_ancillary_file_group.waveform:
        log.warning("Ignoring input waveform file.  Using analytic chirp.")
    chirp = raw.getChirp(frequency, tx)
    log.info(f"Chirp length = {len(chirp)}")
    window = get_window(cfg.processing.range_window, msg="Range window: ")
    chirp *= window(len(chirp))
    log.info("Normalizing chirp to unit white noise gain.")
    return chirp / np.linalg.norm(chirp)


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


def get_total_grid_bounds(rawfiles: List[str], frequency='A'):
    times, ranges = [], []
    for fn in rawfiles:
        raw = open_rrsd(fn)
        pol = raw.polarizations[frequency][0]
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


def get_total_grid(rawfiles: List[str], dt, dr, frequency='A'):
    epoch, tmin, tmax, rmin, rmax = get_total_grid_bounds(rawfiles, frequency)
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
        frequency: str = "A",
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
    frequency : {"A", "B"}, optional
        Band to use.  Default="A"
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
    # Assume look side and center frequency constant across files.
    side = raw.identification.lookDirection
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


def make_doppler(cfg: Struct, frequency='A', epoch: Optional[DateTime] = None):
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
                               frequency=frequency,
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


def make_output_grid(cfg: Struct, igrid: isce3.product.RadarGridParameters,
                     orbit: Orbit, doppler: LUT2d, chirplen_meters: float,
                     dem: isce3.geometry.DEMInterpolator):
    # Default to processing all data.
    t0 = igrid.sensing_start
    t1 = t0 + (igrid.length - 1) / igrid.prf
    r0 = igrid.starting_range
    r1 = r0 + (igrid.width - 1) * igrid.range_pixel_spacing
    dr = igrid.range_pixel_spacing

    ac = cfg.processing.azcomp

    # Calc approx synthetic aperture duration (coherent processing interval).
    tmid = 0.5 * (t0 + t1)
    cpi = isce3.focus.get_sar_duration(tmid, r1, orbit, isce3.core.Ellipsoid(),
                                       ac.azimuth_resolution, igrid.wavelength)
    log.debug(f"Approximate synthetic aperture duration is {cpi} s.")

    # Crop to fully focused region, ignoring range-dependence of CPI.
    # Ignore sampling of convolution kernels, accepting possibility of a few
    # pixels that are only 99.9% focused.
    # CPI is symmetric about Doppler centroid.
    t0 += cpi / 2
    t1 -= cpi / 2
    # Range delay is defined relative to _start_ of TX pulse.
    r1 -= chirplen_meters

    # Output grid is zero Doppler, so reskew the four corners and assume they
    # enclose the image.  Take extrema as default processing box.
    # Define a capture to save some typing
    zerodop = isce3.core.LUT2d()
    def reskew_to_zerodop(t, r):
        return isce3.geometry.rdr2rdr(t, r, orbit, igrid.lookside, doppler,
            igrid.wavelength, dem, doppler_out=zerodop,
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

    dt0 = igrid.ref_epoch + isce3.core.TimeDelta(t0z)
    dt1 = igrid.ref_epoch + isce3.core.TimeDelta(t1z)
    log.info(f"Approximate fully focusable time interval is [{dt0}, {dt1}]")
    log.info(f"Approximate fully focusable range interval is [{r0z}, {r1z}]")

    # TODO snap start time to standard interval

    p = cfg.processing.output_grid
    if p.start_time:
        t0z = (DateTime(p.start_time) - igrid.ref_epoch).total_seconds()
    if p.end_time:
        t1z = (DateTime(p.end_time) - igrid.ref_epoch).total_seconds()
    r0z = p.start_range or r0z
    r1z = p.end_range or r1z
    prf = p.output_prf or igrid.prf

    # User may have updated our precious start/end bounds, so check again that
    # we're actually able to compute the geometry everywhere.  Also
    # backprojection actually runs rdr2rdr in the other direction
    # (zero-to-native) and the solver isn't necessarily symmetric with respect
    # to how it converges to the solution.  So it's good to check here and
    # error out early rather than wait for azcomp to crash and waste the user's
    # time.  Don't bother nudging since we don't want to modify user input.
    log.info("Verifying requested processing domain.")
    for (t, r) in [(t0z, r0z), (t0z, r1z), (t1z, r0z), (t1z, r1z)]:
        try:
            isce3.geometry.rdr2rdr(t, r, orbit, igrid.lookside, zerodop,
                igrid.wavelength, dem, doppler_out=doppler,
                rdr2geo_params=vars(ac.rdr2geo),
                geo2rdr_params=vars(ac.geo2rdr))
        except RuntimeError:
            dt = igrid.ref_epoch + isce3.core.TimeDelta(t)
            log.error(f"Reskew zero-to-native failed at t={dt} r={r}")
            raise RuntimeError("Could not compute imaging geometry")

    nr = round((r1z - r0z) / dr)
    nt = round((t1z - t0z) * prf)
    assert (nr > 0) and (nt > 0)
    ogrid = isce3.product.RadarGridParameters(t0z, igrid.wavelength, prf, r0z,
                                              dr, igrid.lookside, nt, nr,
                                              igrid.ref_epoch)
    return ogrid



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
    """
    def __init__(self, range_cor, dset, **kw):
        self.range_cor = range_cor
        self.dset = dset
        self.stats = isce3.math.StatsRealImagFloat32()
        super().__init__(**kw)

    def write(self, z, block):
        """
        Compute `range_cor * z` then write it to file and accumulate statistics.

        Parameters
        ----------
        z : np.ndarray
            An arbitrary 2D chunk of data to store in `dset`.
        block : tuple[slice]
            Pair of slices describing the (azimuth, range) selection of `dset`
            where the chunk should be stored.
        """
        # scale and deramp
        z *= self.range_cor[None, block[1]]
        # accumulate stats
        s = isce3.math.StatsRealImagFloat32(z)
        amax = np.max(np.abs([s.real.max, s.real.min, s.imag.max, s.imag.min]))
        log.debug(f"scaled max component = {amax}")
        self.stats.update(s)
        # convert to float16 and write to HDF5
        zf = to_complex32(z)
        self.dset.write_direct(zf, dest_sel=block)


def resample(raw: np.ndarray, t: np.ndarray,
             grid: isce3.product.RadarGridParameters, swaths: np.ndarray,
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


def get_range_deramp(grid: isce3.product.RadarGridParameters) -> np.ndarray:
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


def focus(runconfig):
    # Strip off two leading namespaces.
    cfg = runconfig.runconfig.groups
    rawfiles = cfg.input_file_group.input_file_path
    if len(rawfiles) <= 0:
        raise IOError("need at least one raw data file")
    if len(rawfiles) > 1:
        raise NotImplementedError("mixed-mode processing not yet supported")

    input_raw_path = os.path.abspath(rawfiles[0])
    raw = open_rrsd(input_raw_path)
    dem = get_dem(cfg)
    zerodop = isce3.core.LUT2d()
    azres = cfg.processing.azcomp.azimuth_resolution
    atmos = cfg.processing.dry_troposphere_model or "nodelay"
    kernel = get_kernel(cfg)
    scale = cfg.processing.encoding_scale_factor

    # Check that center frequency doesn't depend on TX polarization since
    # this is assumed in RSLC Doppler metadata.
    for frequency, polarizations in raw.polarizations.items():
        fc_list = [raw.getCenterFrequency(frequency, pol[0])
            for pol in polarizations]
        if len(set(fc_list)) > 1:
            raise NotImplementedError("TX frequency agility not supported")

    use_gpu = isce3.core.gpu_check.use_gpu(cfg.worker.gpu_enabled, cfg.worker.gpu_id)
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg.worker.gpu_id)
        isce3.cuda.core.set_device(device)

        log.info(f"Processing using CUDA device {device.id} ({device.name})")

        backproject = isce3.cuda.focus.backproject
    else:
        backproject = isce3.focus.backproject

    # Generate reference output grid based on highest bandwidth, always A.
    log.info(f"Available polarizations: {raw.polarizations}")
    txref = raw.polarizations["A"][0][0]
    pulse_times, raw_grid = raw.getRadarGrid(frequency="A", tx=txref)

    orbit = get_orbit(cfg)
    attitude = get_attitude(cfg)
    # Use same epoch for all objects for consistency and correctness.
    # Especially important since Doppler LUT does not carry its own epoch.
    grid_epoch = raw_grid.ref_epoch
    orbit.update_reference_epoch(grid_epoch)
    attitude.update_reference_epoch(grid_epoch)
    # Need orbit and attitude over whole raw domain in order to generate
    # Doppler LUT.  Check explicitly in order to provide a sensible error.
    log.info("Verifying ephemeris covers time span of raw data.")
    require_ephemeris_overlap(orbit, pulse_times[0], pulse_times[-1], "Orbit")
    require_ephemeris_overlap(attitude, pulse_times[0], pulse_times[-1],
        "Attitude")
    fc_ref, dop_ref = make_doppler(cfg, epoch=grid_epoch)

    log.info(f"len(pulses) = {len(pulse_times)}")
    log.info("Raw grid is %s", raw_grid)
    # Different grids for frequency A and B.
    ogrid = dict()
    max_chirplen = get_max_chirp_duration(cfg) * isce3.core.speed_of_light / 2
    ogrid["A"] = make_output_grid(cfg, raw_grid, orbit, dop_ref, max_chirplen, dem)
    log.info("Output grid A is %s", ogrid["A"])
    assert ogrid["A"].ref_epoch == grid_epoch
    if "B" in raw.frequencies:
        # Ensure aligned grids between A and B by just using an integer skip.
        # Sample rate of A is always an integer multiple of B.
        # Don't assume A and B have same polarization, e.g., Quasi-Quad-Pol
        tx = raw.polarizations["B"][0][0]
        rskip = int(np.round(raw.getRanges("B", tx).spacing
            / raw.getRanges("A", txref).spacing))
        ogrid["B"] = ogrid["A"][:, ::rskip]
        fc = raw.getCenterFrequency("B")
        ogrid["B"].wavelength = isce3.core.speed_of_light / fc
        log.info("Output grid B is %s", ogrid["B"])

    polygon = isce3.geometry.get_geo_perimeter_wkt(ogrid["A"], orbit,
                                                   zerodop, dem)

    output_slc_path = os.path.abspath(cfg.product_path_group.sas_output_file)

    output_dir = os.path.dirname(output_slc_path)
    os.makedirs(output_dir, exist_ok=True)

    product = cfg.primary_executable.product_type
    log.info(f"Creating output {product} product {output_slc_path}")
    slc = SLC(output_slc_path, mode="w", product=product)
    slc.set_orbit(orbit) # TODO acceleration, orbitType
    slc.set_attitude(attitude)
    slc.copy_identification(raw, polygon=polygon,
        track=cfg.geometry.relative_orbit_number,
        frame=cfg.geometry.frame_number,
        start_time=ogrid["A"].sensing_datetime(0),
        end_time=ogrid["A"].sensing_datetime(ogrid["A"].length - 1))

    # store metadata for each frequency
    dop = dict()
    for frequency in raw.frequencies:
        # TODO Find center frequencies after mode intersection.
        fc = raw.getCenterFrequency(frequency)
        dop[frequency] = scale_doppler(dop_ref, fc / fc_ref)
        slc.set_parameters(dop[frequency], orbit.reference_epoch, frequency)
        og = ogrid[frequency]
        t = og.sensing_start + np.arange(og.length) / og.prf
        r = og.starting_range + np.arange(og.width) * og.range_pixel_spacing
        slc.update_swath(t, og.ref_epoch, r, fc, frequency)

        # add calibration section
        for pol in raw.polarizations[frequency]:
            slc.add_calibration_section(frequency, pol, t,
                                        orbit.reference_epoch, r)

    freq = raw.frequencies[0]
    slc.set_geolocation_grid(orbit, ogrid[freq], dop[freq],
                             epsg=4326, dem=dem,
                             **vars(cfg.processing.azcomp.geo2rdr))

    # Scratch directory for intermediate outputs
    scratch_dir = os.path.abspath(cfg.product_path_group.scratch_path)
    os.makedirs(scratch_dir, exist_ok=True)

    # main processing loop
    channels = [(f, p) for f in raw.polarizations for p in raw.polarizations[f]]
    for frequency, pol in channels:
        log.info(f"Processing frequency{frequency} {pol}")
        rawdata = raw.getRawDataset(frequency, pol)
        log.info(f"Raw data shape = {rawdata.shape}")
        raw_times, raw_grid = raw.getRadarGrid(frequency, tx=pol[0])
        fc = raw.getCenterFrequency(frequency, pol[0])
        na = cfg.processing.rangecomp.block_size.azimuth
        nr = rawdata.shape[1]
        swaths = raw.getSubSwaths(frequency, tx=pol[0])
        log.info(f"Number of sub-swaths = {swaths.shape[0]}")

        def temp(suffix):
            return tempfile.NamedTemporaryFile(dir=scratch_dir, suffix=suffix,
                delete=cfg.processing.delete_tempfiles)

        rawfd = temp("_raw.c8")
        log.info(f"Decoding raw data to memory map {rawfd.name}.")
        raw_mm = np.memmap(rawfd, mode="w+", shape=rawdata.shape,
            dtype=np.complex64)
        for pulse in range(0, rawdata.shape[0], na):
            block = np.s_[pulse:pulse+na, :]
            z = rawdata[block]
            # Remove NaNs.  TODO could incorporate into gap mask.
            z[np.isnan(z)] = 0.0
            raw_mm[block] = z

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

        log.info("Generating chirp")
        chirp = get_chirp(cfg, raw, frequency, pol[0])

        rcmode = parse_rangecomp_mode(cfg.processing.rangecomp.mode)
        log.info(f"Preparing range compressor with {rcmode}")
        rc = isce3.focus.RangeComp(chirp, nr, maxbatch=na, mode=rcmode)

        # Rangecomp modifies range grid.  Also update wavelength.
        rc_grid = raw_grid.copy()
        rc_grid.starting_range -= (
            rc_grid.range_pixel_spacing * rc.first_valid_sample)
        rc_grid.width = rc.output_size
        rc_grid.wavelength = isce3.core.speed_of_light / fc
        igeom = isce3.container.RadarGeometry(rc_grid, orbit, dop[frequency])

        fd = temp("_rc.c8")
        log.info(f"Writing range compressed data to {fd.name}")
        rcfile = Raster(fd.name, rc.output_size, rawdata.shape[0], GDT_CFloat32)
        log.info(f"Range compressed data shape = {rcfile.data.shape}")

        for pulse in range(0, rawdata.shape[0], na):
            log.info(f"Range compressing block at pulse {pulse}")
            block = np.s_[pulse:pulse+na, :]
            rc.rangecompress(rcfile.data[block], regridded[block])

        del regridded, regridfd

        acdata = slc.create_image(frequency, pol, shape=ogrid[frequency].shape)

        nr = cfg.processing.azcomp.block_size.range
        na = cfg.processing.azcomp.block_size.azimuth

        if nr < 1:
            nr = ogrid[frequency].width

        if not cfg.processing.is_enabled.azcomp:
            continue

        deramp = get_range_deramp(ogrid[frequency])
        writer = BackgroundWriter(scale * deramp, acdata)

        for i in range(0, ogrid[frequency].length, na):
            # h5py doesn't follow usual slice rules, raises exception
            # if dest_sel slices extend past dataset shape.
            imax = min(i + na, ogrid[frequency].length)
            for j in range(0, ogrid[frequency].width, nr):
                jmax = min(j + nr, ogrid[frequency].width)
                block = np.s_[i:imax, j:jmax]
                log.info(f"Azcomp block at (i, j) = ({i}, {j})")
                bgrid = ogrid[frequency][block]
                ogeom = isce3.container.RadarGeometry(bgrid, orbit, zerodop)
                z = np.zeros(bgrid.shape, 'c8')
                backproject(z, ogeom, rcfile.data, igeom, dem, fc, azres,
                            kernel, atmos, vars(cfg.processing.azcomp.rdr2geo),
                            vars(cfg.processing.azcomp.geo2rdr))
                writer.queue_write(z, block)

        writer.notify_finished()
        slc.write_stats(frequency, pol, writer.stats)

        # Raster/GDAL creates a .hdr file we have to clean up manually.
        hdr = fd.name.replace(".c8", ".hdr")
        if cfg.processing.delete_tempfiles:
            delete_safely(hdr)

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
