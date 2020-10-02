#!/usr/bin/env python3
import argparse
import h5py
import json
import logging
import os
from pybind_nisar.products.readers.Raw import Raw
from pybind_nisar.products.writers import SLC
from pybind_nisar.workflows import defaults
from pybind_nisar.types import to_complex32
import numpy as np
import pybind_isce3 as isce
from pybind_isce3.core import DateTime, LUT2d
from pybind_isce3.io.gdal import Raster, GDT_CFloat32
from ruamel.yaml import YAML
import sys
import tempfile
from typing import List

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


# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_config(yaml):
    "Load default runconfig, override with user input, and convert to Struct"
    parser = YAML(typ='safe')
    cfg = parser.load(defaults.focus.runconfig)
    with open(yaml) as f:
        user = parser.load(f)
    deep_update(cfg, user)
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


def check_valid_cuda_device(cfg: Struct):
    """Validate that the requested CUDA device is supported."""
    from pybind_isce3.cuda.core import Device, min_compute_capability
    device = Device(cfg.worker.gpu_id)
    return device.compute_capability >= min_compute_capability()


def check_gpu_opts(cfg: Struct):
    """Validate the specified GPU processing configuration options.

    Returns
    -------
    logical
        Whether to use GPU processing

    Raises
    ------
    ValueError
        If GPU processing was requested but not available or if an invalid CUDA
        device was requested
    """
    # Check if GPU processing was requested.
    gpu_requested = cfg.worker.gpu_enabled

    # Check if CUDA support is enabled.
    cuda_available = lambda : hasattr(isce, "cuda")

    # If unspecified, use GPU processing if supported. Otherwise, fall back to
    # CPU processing.
    if gpu_requested is None:
        return cuda_available() and check_valid_cuda_device(cfg)

    # If GPU processing was requested, raise an error if CUDA support is not
    # available or if the specified device is not valid.
    if gpu_requested:
        if not cuda_available():
            # XXX logging an error does not halt execution
            errmsg = "GPU processing was requested but not available"
            log.error(errmsg)
            raise ValueError(errmsg)

        if not check_valid_cuda_device(cfg):
            errmsg = "The requested CUDA device has insufficient compute " \
                     "capability"
            log.error(errmsg)
            raise ValueError(errmsg)

        return True

    # GPU processing was not requested.
    return False


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


def get_chirp(cfg: Struct, raw: Raw, frequency: str):
    if cfg.DynamicAncillaryFileGroup.Waveform:
        log.warning("Ignoring input waveform file.  Using analytic chirp.")
    chirp = raw.getChirp(frequency)
    log.info(f"Chirp length = {len(chirp)}")
    window = get_window(cfg.processing.range_window, msg="Range window: ")
    chirp *= window(len(chirp))
    log.info("Normalizing chirp to unit white noise gain.")
    return chirp / np.linalg.norm(chirp)**2


def parse_rangecomp_mode(mode: str):
    lut = {"full": isce.focus.RangeComp.Mode.Full,
           "same": isce.focus.RangeComp.Mode.Same,
           "valid": isce.focus.RangeComp.Mode.Valid}
    mode = mode.lower()
    if mode not in lut:
        raise ValueError(f"Invalid RangeComp mode {mode}")
    return lut[mode]


def get_orbit(cfg: Struct):
    log.info("Loading orbit")
    if cfg.DynamicAncillaryFileGroup.Orbit:
        log.warning("Ignoring input orbit file.  Using L0B orbits.")
    rawfiles = cfg.InputFileGroup.InputFilePath
    if len(rawfiles) > 1:
        raise NotImplementedError("Can't concatenate orbit data.")
    raw = Raw(hdf5file=rawfiles[0])
    return raw.getOrbit()


def get_attitude(cfg: Struct):
    log.info("Loading attitude")
    if cfg.DynamicAncillaryFileGroup.Pointing:
        log.warning("Ignoring input pointing file.  Using L0B attitude.")
    rawfiles = cfg.InputFileGroup.InputFilePath
    if len(rawfiles) > 1:
        raise NotImplementedError("Can't concatenate attitude data.")
    raw = Raw(hdf5file=rawfiles[0])
    return raw.getAttitude()


def get_total_grid_bounds(rawfiles: List[str], frequency='A'):
    times, ranges = [], []
    for fn in rawfiles:
        raw = Raw(hdf5file=fn)
        pol = raw.polarizations[frequency][0]
        ranges.append(raw.getRanges(frequency))
        times.append(raw.getPulseTimes(frequency, pol[0]))
    rmin = min(r[0] for r in ranges)
    rmax = max(r[-1] for r in ranges)
    dtmin = min(epoch + isce.core.TimeDelta(t[0]) for (epoch, t) in times)
    dtmax = max(epoch + isce.core.TimeDelta(t[-1]) for (epoch, t) in times)
    epoch = min(epoch for (epoch, t) in times)
    tmin = (dtmin - epoch).total_seconds()
    tmax = (dtmax - epoch).total_seconds()
    return epoch, tmin, tmax, rmin, rmax


def get_total_grid(rawfiles: List[str], dt, dr, frequency='A'):
    epoch, tmin, tmax, rmin, rmax = get_total_grid_bounds(rawfiles, frequency)
    nt = int(np.ceil((tmax - tmin) / dt))
    nr = int(np.ceil((rmax - rmin) / dr))
    t = isce.core.Linspace(tmin, dt, nt)
    r = isce.core.Linspace(rmin, dr, nr)
    return epoch, t, r


def squint(t, r, orbit, attitude, side, angle=0.0, dem=None, **kw):
    """Find squint angle given imaging time and range to target.
    """
    p, v = orbit.interpolate(t)
    R = attitude.interpolate(t).to_rotation_matrix()
    axis = R[:,0]
    if dem is None:
        dem = isce.geometry.DEMInterpolator()
    xyz = isce.geometry.rdr2geo_cone(p, axis, angle, r, dem, side, **kw)
    look = (xyz - p) / np.linalg.norm(xyz - p)
    vhat = v / np.linalg.norm(v)
    return np.arcsin(look.dot(vhat))


def squint_to_doppler(squint, wvl, vmag):
    return 2.0 / wvl * vmag * np.sin(squint)


def convert_epoch(t: List[float], epoch_in, epoch_out):
    TD = isce.core.TimeDelta
    return [(epoch_in - epoch_out + TD(ti)).total_seconds() for ti in t]


def get_dem(cfg: Struct):
    dem = isce.geometry.DEMInterpolator(
        height=cfg.processing.dem.reference_height,
        method=cfg.processing.dem.interp_method)
    fn = cfg.DynamicAncillaryFileGroup.DEMFile
    if fn:
        log.info(f"Loading DEM {fn}")
        dem.load_dem(fn)
    else:
        log.warning("No DEM given, using height=0.")
    return dem


def make_doppler_lut(rawfiles: List[str],
        az: float = 0.0,
        orbit: isce.core.Orbit = None,
        attitude: isce.core.Attitude = None,
        dem: isce.geometry.DEMInterpolator = None,
        azimuth_spacing: float = 1.0,
        range_spacing: float = 1e3,
        frequency: str = "A",
        interp_method: str = "bilinear",
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
    raw = Raw(hdf5file=rawfiles[0])
    if orbit is None:
        orbit = raw.getOrbit()
    if attitude is None:
        attitude = raw.getAttitude()
    if dem is None:
        dem = isce.geometry.DEMInterpolator()
    # Assume look side and center frequency constant across files.
    side = raw.identification.lookDirection
    fc = raw.getCenterFrequency(frequency)

    # Now do the actual calculations.
    wvl = isce.core.speed_of_light / fc
    epoch, t, r = get_total_grid(rawfiles, azimuth_spacing, range_spacing)
    t = convert_epoch(t, epoch, orbit.reference_epoch)
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


def make_doppler(cfg: Struct, frequency='A'):
    log.info("Generating Doppler LUT from pointing")
    orbit = get_orbit(cfg)
    attitude = get_attitude(cfg)
    dem = get_dem(cfg)
    opt = cfg.processing.doppler
    az = np.radians(opt.azimuth_boresight_deg)
    rawfiles = cfg.InputFileGroup.InputFilePath

    fc, lut = make_doppler_lut(rawfiles,
                               az=az, orbit=orbit, attitude=attitude,
                               dem=dem, azimuth_spacing=opt.spacing.azimuth,
                               range_spacing=opt.spacing.range,
                               frequency=frequency,
                               interp_method=opt.interp_method,
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


def make_output_grid(cfg: Struct, igrid):
    t0 = t0in = igrid.sensing_start
    t1 = t1in = t0 + igrid.length / igrid.prf
    r0 = r0in = igrid.starting_range
    r1 = r1in = r0 + igrid.width * igrid.range_pixel_spacing
    # TODO crop chirp length, synthetic aperture, and reskew.
    # TODO snap start time to standard interval
    dr = igrid.range_pixel_spacing

    p = cfg.processing.output_grid
    if p.start_time:
        t0 = (DateTime(p.start_time) - igrid.ref_epoch).total_seconds()
    if p.end_time:
        t1 = (DateTime(p.end_time) - igrid.ref_epoch).total_seconds()
    r0 = p.start_range or r0
    r1 = p.end_range or r1
    prf = p.output_prf or igrid.prf

    if t1 < t0in:
        raise ValueError(f"Output grid t1={t1} < input grid t0={t0in}")
    if t0 > t1in:
        raise ValueError(f"Output grid t0={t0} > input grid t1={t1in}")
    if r1 < r0in:
        raise ValueError(f"Output grid r1={r1} < input grid r0={r0in}")
    if r0 > r1in:
        raise ValueError(f"Output grid r0={r0} > input grid r1={r1in}")

    nr = int(np.round((r1 - r0) / dr))
    nt = int(np.round((t1 - t0) * prf))
    assert (nr > 0) and (nt > 0)
    ogrid = isce.product.RadarGridParameters(t0, igrid.wavelength, prf, r0,
                                             dr, igrid.lookside, nt, nr,
                                             igrid.ref_epoch)
    return ogrid



def get_kernel(cfg: Struct):
    # TODO
    opt = cfg.processing.azcomp.kernel
    if opt.type.lower() != 'knab':
        raise NotImplementedError("Only Knab kernel implemented.")
    n = 1 + 2 * opt.halfwidth
    kernel = isce.core.KnabKernel(n, 1 / 1.2)
    assert opt.fit.lower() == "table"
    table = isce.core.TabulatedKernelF32(kernel, opt.fit_order)
    return table


def focus(runconfig):
    # Strip off two leading namespaces.
    cfg = runconfig.runconfig.groups
    rawfiles = cfg.InputFileGroup.InputFilePath
    if len(rawfiles) <= 0:
        raise IOError("need at least one raw data file")
    if len(rawfiles) > 1:
        raise NotImplementedError("mixed-mode processing not yet supported")

    input_raw_path = os.path.abspath(rawfiles[0])
    raw = Raw(hdf5file=input_raw_path)
    dem = get_dem(cfg)
    orbit = get_orbit(cfg)
    attitude = get_attitude(cfg)
    fc_ref, dop_ref = make_doppler(cfg)
    zerodop = isce.core.LUT2d()
    azres = cfg.processing.azcomp.azimuth_resolution
    atmos = cfg.processing.dry_troposphere_model or "nodelay"
    kernel = get_kernel(cfg)
    scale = cfg.processing.encoding_scale_factor

    use_gpu = check_gpu_opts(cfg)
    if use_gpu:
        # Set the current CUDA device.
        device = isce.cuda.core.Device(cfg.worker.gpu_id)
        isce.cuda.core.set_device(device)

        log.info(f"Processing using CUDA device {device.id} ({device.name})")

        backproject = isce.cuda.focus.backproject
    else:
        backproject = isce.focus.backproject

    # Generate reference output grid based on highest bandwidth, always A.
    log.info(f"Available polarizations: {raw.polarizations}")
    txref = raw.polarizations["A"][0][0]
    pulse_times, raw_grid = raw.getRadarGrid(frequency="A", tx=txref)

    log.info(f"len(pulses) = {len(pulse_times)}")
    log.info("Raw grid is %s", raw_grid)
    # Different grids for frequency A and B.
    ogrid = dict(A = make_output_grid(cfg, raw_grid))
    log.info("Output grid A is %s", ogrid["A"])
    if "B" in raw.frequencies:
        # Ensure aligned grids between A and B by just using an integer skip.
        # Sample rate of A is always an integer multiple of B.
        rskip = int(np.round(raw.getRanges("B").spacing
            / raw.getRanges("A").spacing))
        ogrid["B"] = ogrid["A"][:, ::rskip]
        log.info("Output grid B is %s", ogrid["B"])

    polygon = isce.geometry.get_geo_perimeter_wkt(ogrid["A"], orbit,
                                                  zerodop, dem)

    output_slc_path = os.path.abspath(cfg.ProductPathGroup.SASOutputFile)

    output_dir = os.path.dirname(output_slc_path)
    os.makedirs(output_dir, exist_ok=True)

    product = cfg.PrimaryExecutable.ProductType
    log.info(f"Creating output {product} product {output_slc_path}")
    slc = SLC(output_slc_path, mode="w", product=product)
    slc.set_orbit(orbit) # TODO acceleration, orbitType
    if attitude:
        slc.set_attitude(attitude, orbit.reference_epoch)
    slc.copy_identification(raw, polygon=polygon,
        track=cfg.Geometry.RelativeOrbitNumber,
        frame=cfg.Geometry.FrameNumber)

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

    # Scratch directory for intermediate outputs
    scratch_dir = os.path.abspath(cfg.ProductPathGroup.ScratchPath)
    os.makedirs(scratch_dir, exist_ok=True)

    # main processing loop
    channels = [(f, p) for f in raw.polarizations for p in raw.polarizations[f]]
    for frequency, pol in channels:
        log.info(f"Processing frequency{frequency} {pol}")
        rawdata = raw.getRawDataset(frequency, pol)
        log.info(f"Raw data shape = {rawdata.shape}")
        _, raw_grid = raw.getRadarGrid(frequency, tx=pol[0])
        fc = raw.getCenterFrequency(frequency)
        na = cfg.processing.rangecomp.block_size.azimuth
        nr = rawdata.shape[1]

        log.info("Generating chirp")
        chirp = get_chirp(cfg, raw, frequency)

        rcmode = parse_rangecomp_mode(cfg.processing.rangecomp.mode)
        log.info(f"Preparing range compressor with {rcmode}")
        rc = isce.focus.RangeComp(chirp, nr, maxbatch=na, mode=rcmode)

        # Rangecomp modifies range grid.  Also update wavelength.
        rc_grid = raw_grid.copy()
        rc_grid.starting_range -= (
            rc_grid.range_pixel_spacing * rc.first_valid_sample)
        rc_grid.width = rc.output_size
        rc_grid.wavelength = isce.core.speed_of_light / fc
        igeom = isce.container.RadarGeometry(rc_grid, orbit, dop[frequency])

        fd = tempfile.NamedTemporaryFile(dir=scratch_dir, suffix='.rc')
        log.info(f"Writing range compressed data to {fd.name}")
        rcfile = Raster(fd.name, rc.output_size, rawdata.shape[0], GDT_CFloat32)
        log.info(f"Range compressed data shape = {rcfile.data.shape}")

        for pulse in range(0, rawdata.shape[0], na):
            log.info(f"Range compressing block at pulse {pulse}")
            block = np.s_[pulse:pulse+na, :]
            # TODO fill invalid data during presum.
            ps = rawdata[block]
            ps[np.isnan(ps)] = 0.0
            rc.rangecompress(rcfile.data[block], ps)

        acdata = slc.create_image(frequency, pol, shape=ogrid[frequency].shape)

        nr = cfg.processing.azcomp.block_size.range
        na = cfg.processing.azcomp.block_size.azimuth

        if nr < 1:
            nr = ogrid[frequency].width

        if not cfg.processing.is_enabled.azcomp:
            continue

        for i in range(0, ogrid[frequency].length, na):
            # h5py doesn't follow usual slice rules, raises exception
            # if dest_sel slices extend past dataset shape.
            imax = min(i + na, ogrid[frequency].length)
            for j in range(0, ogrid[frequency].width, nr):
                jmax = min(j + nr, ogrid[frequency].width)
                block = np.s_[i:imax, j:jmax]
                log.info(f"Azcomp block at (i, j) = ({i}, {j})")
                bgrid = ogrid[frequency][block]
                ogeom = isce.container.RadarGeometry(bgrid, orbit, zerodop)
                z = np.zeros(bgrid.shape, 'c8')
                backproject(z, ogeom, rcfile.data, igeom, dem, fc, azres,
                            kernel, atmos, vars(cfg.processing.azcomp.rdr2geo),
                            vars(cfg.processing.azcomp.geo2rdr))
                log.debug(f"max(abs(z)) = {np.max(np.abs(z))}")
                zf = to_complex32(scale * z)
                acdata.write_direct(zf, dest_sel=block)

        # Raster/GDAL creates a .hdr file we have to clean up manually.
        hdr = fd.name.replace(".rc", ".hdr")
        # Careful to avoid race on file deletion.  Use pathlib in Python 3.8+
        try:
            os.unlink(hdr)
        except FileNotFoundError:
            pass


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
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = validate_config(load_config(args.config))
    echofile = cfg.runconfig.groups.ProductPathGroup.SASConfigFile
    if echofile:
        log.info(f"Logging configuration to file {echofile}.")
        dump_config(cfg, echofile)
    focus(cfg)


if __name__ == '__main__':
    main(sys.argv[1:])
