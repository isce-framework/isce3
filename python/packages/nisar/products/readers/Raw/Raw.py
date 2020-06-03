from .DataDecoder import DataDecoder
import h5py
import logging
from nisar.products.readers import Base
import numpy as np
import pyre
import pybind_isce3 as isce
import re

# TODO some CSV logger
log = logging.getLogger("Raw")

PRODUCT = "RRSD"

def find_case_insensitive(group: h5py.Group, name: str) -> str:
    for key in group:
        if key.lower() == name.lower():
            return key
    raise ValueError(f"{name} not found in HDF5 group {group.name}")


class Raw(Base, family='nisar.productreader.raw'):
    '''
    Class for parsing NISAR L0B products into isce structures.
    '''
    productValidationType = pyre.properties.str(default=PRODUCT)
    productValidationType.doc = 'Validation tag to ensure correct product type'
    _ProductType = pyre.properties.str(default=PRODUCT)
    _ProductType.doc = 'The type of the product.'

    def __init__(self, product=PRODUCT, **kwds):
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
        rxpat = re.compile("^rx[HV]$")
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
            dr = band["slantRangeSpacing"][()]
            fs = isce.core.speed_of_light / 2 / dr
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
            # FIXME REE uses wrong case for UTCTime
            name = find_case_insensitive(f[txpath], "UTCTime")
            t = np.asarray(f[txpath][name])
            epoch = isce.io.get_ref_epoch(f[txpath], name)
        return epoch, t


    def getCenterFrequency(self, frequency: str = 'A'):
        bandpath = self.BandPath(frequency)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            fc = f[bandpath]["centerFrequency"][()]
        return fc


    # XXX C++ and Base.py assume SLC.  Grid less well defined for Raw case
    # since PRF isn't necessarily constant.  Return pulse times with grid?
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
