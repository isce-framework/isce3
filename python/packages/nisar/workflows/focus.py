#!/usr/bin/env python3
import argparse
import h5py
import isce3 as cisce # TODO need to port rdr2geo_cone
import logging
from nisar.products.readers import Base
from pathlib import Path
import numpy as np
import pybind_isce3 as isce
from pybind_isce3.io.gdal import Raster, GDT_CFloat32
import pyre
import re
from ruamel.yaml import YAML
import sys
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


def load_config(yaml):
    # TODO load defaults first.
    with open(yaml) as f:
        return Struct(YAML().load(f))


def validate_config(x):
    # TODO
    return x


complex32 = np.dtype([('r', np.float16), ('i', np.float16)])


class DataDecoder(object):
    """Handle the various data types floating around for raw data, currently
    complex32, complex64, and lookup table.  Indexing operatations always return
    data converted to complex64.
    """
    def __getitem__(self, key):
        return self.decoder(key)

    def _decode_lut(self, key):
        z = self.dataset[key]
        assert self.table is not None
        return self.table[z['r']] + 1j * self.table[z['i']]

    def _decode_complex32(self, key):
        with self.dataset.astype(np.complex64):
            z = self.dataset[key]
        return z

    def __init__(self, h5dataset):
        self.table = None
        self.decoder = lambda key: self.dataset[key]
        self.dataset = h5dataset
        self.shape = self.dataset.shape
        group = h5dataset.parent
        if "BFPQLUT" in group:
            assert group["BFPQLUT"].dtype == np.float32
            self.table = np.asarray(group["BFPQLUT"])
            self.decoder = self._decode_lut
            log.info("Decoding raw data with lookup table.")
        elif h5dataset.dtype == complex32:
            self.decoder = self._decode_complex32
            log.info("Decoding raw data from float16 encoding.")
        else:
            assert h5dataset.dtype == np.complex64
            log.info("Decoding raw data not required")


def to_complex32(z):
    zf = np.zeros(z.shape, dtype=complex32)
    zf['r'] = z.real
    zf['i'] = z.imag
    return zf

#PRODUCT = "RRSD"
PRODUCT = "RSD"

class Raw(Base, family='nisar.productreader.raw'):
    '''
    Class for parsing NISAR L0B products into isce structures.
    '''
    productValidationType = pyre.properties.str(default=PRODUCT)
    productValidationType.doc = 'Validation tag to ensure correct product type'
    _ProductType = pyre.properties.str(default=PRODUCT)
    _ProductType.doc = 'The type of the product.'

    def __init__(self, product="RRSD", **kwds):
        '''
        Constructor to initialize product with HDF5 file.
        '''
        log.info(f"Reading L0B file {kwds['hdf5file']}")
        super().__init__(**kwds)

    def parsePolarizations(self):
        '''
        Parse HDF5 and identify polarization channels available for each frequency.
        '''
        try:
            frequencyList = self.frequencies
        except:
            raise RuntimeError('Cannot determine list of available frequencies'
                + ' without parsing Product Identification')

        txpat = re.compile("^tx[HVLR]$")
        rxpat = re.compile("^rx[HVLR]$")
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            for freq in frequencyList:
                group = fid[f"{self.SwathPath}/frequency{freq}"]
                tx = [x[2] for x in group.keys() if txpat.match(x)]
                pols = []
                for t in tx:
                    rx = [x[2] for x in group[f"tx{t}"].keys() if rxpat.match(x)]
                    for r in rx:
                        pols.append(t + r)
                self.polarizations[freq] = pols

    def BandPath(self, frequency):
        return f"{self.SwathPath}/frequency{frequency}"

    def _rawGroup(self, frequency, polarization):
        tx, rx = polarization[0], polarization[1]
        return f"{self.BandPath(frequency)}/tx{tx}/rx{rx}"

    def rawPath(self, frequency, polarization):
        tx, rx = polarization[0], polarization[1]
        return f"{self._rawGroup(frequency, polarization)}/{tx}{rx}"

    def getRawDataset(self, frequency, polarization):
        '''
        Return raw dataset of given frequency and polarization from hdf5 file
        '''
        fid = h5py.File(self.filename, 'r', libver='latest', swmr=True)
        path = self.rawPath(frequency, polarization)
        return DataDecoder(fid[path])

    def getChirp(self, frequency: str = 'A'):
        """Return analytic chirp for a given frequency.
        """
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            band = f[self.BandPath(frequency)]
            T = band["chirpDuration"][()]
            K = band["chirpSlope"][()]
            fs = band["rangeSamplingFrequency"][()]
        log.info(f"Chirp({K}, {T}, {fs})")
        return np.asarray(isce.focus.form_linear_chirp(K, T, fs))

    @property
    def TelemetryPath(self):
        return f"{self.ProductPath}/telemetry"

    # XXX Base.getOrbit has @pyre.export decorator.  What's that do?
    # XXX Base uses SLC-specific path and returns Cython object.
    def getOrbit(self):
        path = f"{self.TelemetryPath}/orbit"
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            orbit = isce.core.Orbit.load_from_h5(f[path])
        return orbit

    def getAttitude(self):
        path = f"{self.TelemetryPath}/attitude"
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            q = isce.core.Quaternion.load_from_h5(f[path])
        return q

    def getRanges(self, frequency='A'):
        bandpath = self.BandPath(frequency)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            r = np.asarray(f[bandpath]["slantRange"])
            dr = f[bandpath]["slantRangeSpacing"][()]
        nr = len(r)
        out = isce.core.Linspace(r[0], dr, nr)
        assert np.isclose(out[-1], r[-1])
        return out


    def TransmitPath(self, frequency='A', tx='H'):
        return f"{self.BandPath(frequency)}/tx{tx}"


    def getPulseTimes(self, frequency='A', tx='H'):
        txpath = self.TransmitPath(frequency, tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            name = "UTCtime"
            t = np.asarray(f[txpath][name])
            epoch = isce.io.get_ref_epoch(f[txpath], name)
        return epoch, t


    def getCenterFrequency(self, frequency: str = 'A'):
        bandpath = self.BandPath(frequency)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            fc = f[bandpath]["centerFrequency"][()]
        return fc


    # XXX C++ and Base.py assume SLC.  Grid less well defined for Raw case
    # since PRF isn't necessarily constant.
    def getRadarGrid(self, frequency='A', tx='H', prf=None):
        fc = self.getCenterFrequency(frequency)
        wvl = isce.core.speed_of_light / fc
        r = self.getRanges(frequency)
        epoch, t = self.getPulseTimes(frequency, tx)
        nt = len(t)
        assert nt > 1
        if prf:
            nt = 1 + int(np.ceil((t[-1] - t[0]) * prf))
        else:
            prf = (nt - 1) / (t[-1] - t[0])
        side = self.identification.lookDirection
        grid = isce.product.RadarGridParameters(
            t[0], wvl, prf, r[0], r.spacing, side, nt, len(r), epoch)
        return t, grid


class LoggingH5File(h5py.File):
    def create_dataset(self, *args, **kw):
        log.debug(f"Creating dataset {args[0]}")
        return super().create_dataset(*args, **kw)


def cosine_window(n, pedestal):
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
    if cfg.inputs.waveform:
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
    if cfg.inputs.orbit:
        log.warning("Ignoring input orbit file.  Using L0B orbits.")
    if len(cfg.inputs.raw) > 1:
        raise NotImplementedError("Can't concatenate orbit data.")
    raw = Raw(hdf5file=cfg.inputs.raw[0])
    return raw.getOrbit()


def get_attitude(cfg: Struct):
    log.info("Loading attitude")
    if cfg.inputs.pointing:
        log.warning("Ignoring input pointing file.  Using L0B attitude.")
    if len(cfg.inputs.raw) > 1:
        raise NotImplementedError("Can't concatenate attitude data.")
    raw = Raw(hdf5file=cfg.inputs.raw[0])
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
    R = attitude.rotmat(t)
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
    fn = cfg.inputs.dem
    if fn:
        log.info(f"Loading DEM {fn}")
        dem.load_dem(fn)
    else:
        log.warning("No DEM given, using height=0.")
    return dem


def make_doppler(cfg: Struct, frequency='A'):
    log.info("Generating Doppler LUT from pointing")
    orbit = get_orbit(cfg)
    attitude = get_attitude(cfg)
    dem = get_dem(cfg)
    opt = cfg.processing.doppler
    az = np.radians(opt.azimuth_boresight_deg)
    raw = Raw(hdf5file=cfg.inputs.raw[0])
    side = raw.identification.lookDirection
    fc = raw.getCenterFrequency(frequency)
    wvl = isce.core.speed_of_light / fc

    epoch, t, r = get_total_grid(cfg.inputs.raw, opt.spacing.azimuth,
                                 opt.spacing.range)
    t = convert_epoch(t, epoch, orbit.reference_epoch)
    dop = np.zeros((len(t), len(r)))
    for i, ti in enumerate(t):
        _, v = orbit.interpolate(ti)
        vi = np.linalg.norm(v)
        for j, rj in enumerate(r):
            sq = squint(ti, rj, orbit, attitude, side, angle=az, dem=dem,
                        **vars(opt.rdr2geo))
            dop[i,j] = squint_to_doppler(sq, wvl, vi)
    lut = isce.core.LUT2d(np.asarray(r), t, dop, opt.interp_method, False)
    log.info(f"Constructed Doppler LUT for fc={fc} Hz.")
    return fc, lut


def focus(cfg):
    if len(cfg.inputs.raw) <= 0:
        raise IOError("need at least one raw data file")
    if len(cfg.inputs.raw) > 1:
        raise NotImplementedError("mixed-mode processing not yet supported")

    raw = Raw(hdf5file=cfg.inputs.raw[0])
    dem = get_dem(cfg)
    orbit = get_orbit(cfg)
    pulse_times, igrid = raw.getRadarGrid(frequency="A", tx="H")
    fc_ref, dop_ref = make_doppler(cfg)

    log.info(f"Creating output SLC product {cfg.outputs.slc}")
    slc = LoggingH5File(cfg.outputs.slc, mode="w")

    log.info(f"Available polarizations: {raw.polarizations}")
    channels = [(f, p) for f in raw.polarizations for p in raw.polarizations[f]]

    # TODO Find common center frequencies after mode intersection.
    # TODO Need a way to scale a LUT2d or at least get its data.
    g = slc.create_group(
        "/science/LSAR/RSLC/metadata/processingInformation/parameters")
    g.create_group("frequencyA")  # FIXME?
    dop_ref.save_to_h5(g, "frequencyA/dopplerCentroid", orbit.reference_epoch, "Hz")

    for frequency, pol in channels:
        log.info(f"Processing frequency{frequency} {pol}")
        rawdata = raw.getRawDataset(frequency, pol)
        log.info(f"Raw data shape = {rawdata.shape}")
        r = raw.getSlantRange(frequency)
        na = cfg.processing.rangecomp.block_size.azimuth
        nr = rawdata.shape[1]

        log.info("Generating chirp")
        chirp = get_chirp(cfg, raw, frequency)

        rcmode = parse_rangecomp_mode(cfg.processing.rangecomp.mode)
        log.info(f"Preparing range compressor with {rcmode}")
        rc = isce.focus.RangeComp(chirp, nr, maxbatch=na, mode=rcmode)

        name = str(Path(cfg.outputs.workdir) / "rangecomp")
        log.info(f"Writing range compressed data to {name}")
        rcfile = Raster(name, rc.output_size, rawdata.shape[0], GDT_CFloat32)
        log.info(f"Range compressed data shape = {rcfile.data.shape}")

        for pulse in range(0, rawdata.shape[0], na):
            log.info(f"Range compressing block at pulse {pulse}")
            block = np.s_[pulse:pulse+na, :]
            batch = rawdata[block].shape[0]
            rc.rangecompress(rcfile.data[block], rawdata[block], batch)

        name = f"/science/LSAR/RSLC/swaths/frequency{frequency}/{pol}"
        acdata = slc.create_dataset(name, dtype=complex32, shape=rcfile.data.shape)

        for pulse in range(0, rcfile.data.shape[0], na):
            block = np.s_[pulse:pulse+na, :]
            z = rcfile.data[block]
            # TODO azcomp
            log.info(f"Writing block at pulse {pulse}")
            zf = to_complex32(z)
            acdata.write_direct(zf, dest_sel=block)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args(argv)

    cfg = validate_config(load_config(args.config))
    focus(cfg)


if __name__ == '__main__':
    log_level = logging.DEBUG
    log.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    log.addHandler(sh)

    main(sys.argv[1:])
