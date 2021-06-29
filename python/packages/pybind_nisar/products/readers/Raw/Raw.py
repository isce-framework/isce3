from .DataDecoder import DataDecoder
import h5py
import isce3
import logging
from pybind_nisar.products.readers import Base
import numpy as np
import pyre
import journal
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


class RawBase(Base, family='nisar.productreader.raw'):
    '''
    Base class for NISAR L0B products. Derived classes correspond to
    legacy (`LegacyRaw`) & current (`Raw`) versions of the product spec.
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

        # Set error channel
        self.error_channel = journal.error('Raw')

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

    # All methods assigned to _pulseMetaPath must present the same interface,
    # hence unused keyword arguments.
    def BandPath(self, frequency='A', **kw):
        return f"{self.SwathPath}/frequency{frequency}"

    def TransmitPath(self, frequency='A', tx='H'):
        return f"{self.BandPath(frequency)}/tx{tx}"

    # Some stuff got moved from BandPath to TransmitPath.  This method allows
    # a way to override which one to use in subclasses.  Intend to remove once
    # we're done transitioning raw data format.
    _pulseMetaPath = TransmitPath

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

    def getChirp(self, frequency: str = 'A', tx: str = 'H'):
        """Return analytic chirp for a given band/transmit.
        """
        _, fs, K, T = self.getChirpParameters(frequency, tx)
        log.info(f"Chirp({K}, {T}, {fs})")
        return np.asarray(isce3.focus.form_linear_chirp(K, T, fs))

    def getChirpParameters(self, frequency: str = 'A', tx: str = 'H'):
        """Get metadata describing chirp.

        Parameters
        ----------
        frequency : {'A', 'B'}, optional
            Sub-band
        tx : {'H', 'V', 'L', 'R'}, optional
            Transmit polarization

        Returns
        -------
        fc : float
            center frequency in Hz
        fs : float
            sample rate in Hz
        K : float
            chirp slope (signed) in Hz/s
        T : float
            chirp duration in s
        """
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            group = f[self._pulseMetaPath(frequency=frequency, tx=tx)]
            T = group["chirpDuration"][()]
            K = group["chirpSlope"][()]
            dr = group["slantRangeSpacing"][()]
        fs = isce.core.speed_of_light / (2 * dr)
        fc = self.getCenterFrequency(frequency, tx)
        return fc, fs, K, T

    @property
    def TelemetryPath(self):
        return f"{self.ProductPath}/lowRateTelemetry"

    # XXX Base.getOrbit has @pyre.export decorator.  What's that do?
    # XXX L0B doesn't put orbit in MetadataPath
    def getOrbit(self):
        path = f"{self.TelemetryPath}/orbit"
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            orbit = isce.core.Orbit.load_from_h5(f[path])
        return orbit

    def getAttitude(self):
        path = f"{self.TelemetryPath}/attitude"
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            q = isce.core.Attitude.load_from_h5(f[path])
        return q

    def getRanges(self, frequency='A', tx='H'):
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            group = f[path]
            r = np.asarray(group["slantRange"])
            dr = group["slantRangeSpacing"][()]
        nr = len(r)
        out = isce.core.Linspace(r[0], dr, nr)
        assert np.isclose(out[-1], r[-1])
        return out


    def getPulseTimes(self, frequency='A', tx='H'):
        """
        Read pulse time tags.

        Parameters
        ----------
        frequency : {'A', 'B'}
            Sub-band.  Typically main science band is 'A'.

        tx : {'H', 'V', 'L', 'R'}
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear), left circular, right circular

        Returns
        -------
        epoch : isce3.core.DateTime
            UTC time reference

        t : array_like
            Transmit time of each pulse, in seconds relative to epoch.
        """
        txpath = self.TransmitPath(frequency, tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            # FIXME product spec changed UTCTime -> UTCtime
            name = find_case_insensitive(f[txpath], "UTCtime")
            t = np.asarray(f[txpath][name])
            epoch = isce.io.get_ref_epoch(f[txpath], name)
        return epoch, t


    def getCenterFrequency(self, frequency: str = 'A', tx: str = None):
        if tx is None:
            tx = self.polarizations[frequency][0][0]
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[path]["centerFrequency"][()]


    # XXX C++ and Base.py assume SLC.  Grid less well defined for Raw case
    # since PRF isn't necessarily constant.  Return pulse times with grid?
    def getRadarGrid(self, frequency='A', tx='H', prf=None):
        fc = self.getCenterFrequency(frequency, tx)
        wvl = isce.core.speed_of_light / fc
        r = self.getRanges(frequency, tx)
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


    def getSubSwaths(self, frequency='A', tx='H'):
        """Get an array of indices denoting where raw data are valid (e.g., not
        within a transmit gap).  Shape is (ns, nt, 2) where ns is the number of
        sub-swaths and nt is the number of pulse times.  Each pair of numbers
        indicates the [start, end) valid samples.
        """
        txpath = self.TransmitPath(frequency, tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            ns = f[txpath]["numberOfSubSwaths"][()]
            ss1 = f[txpath]["validSamplesSubSwath1"][:]
            nt = ss1.shape[0]
            swaths = np.zeros((ns, nt, 2), dtype=int)
            swaths[0, ...] = ss1
            for i in range(1, ns):
                name = f"validSamplesSubSwath{i+1}"
                swaths[i, ...] = f[txpath][name][:]
        return swaths


# adapted from ReeUtilPy/REEout/AntPatAnalysis.py:getDCMant2sc
def get_rcs2body(el_deg=37.0, az_deg=0.0, side='left') -> isce3.core.Quaternion:
    """
    Get quaternion for conversion from antenna to spacecraft ijk, a forward-
    right-down body-fixed system.  For details see section 8.1.2 of REE User's
    Guide (JPL D-95653).

    Parameters
    ----------
    el_deg : float
        angle (deg) between mounting X-Z plane and Antenna X-Z plane

    az_deg : float
        angle (deg) between mounting Y-Z plane and Antenna Y-Z plane

    side : {'right', 'left'}
        Radar look direction.

    Returns
    -------
    q : isce3.core.Quaternion
        rcs-to-body quaternion
    """
    d = -1.0 if side.lower() == 'left' else 1.0
    az, el = np.deg2rad([az_deg, el_deg])
    saz, caz = np.sin(az), np.cos(az)
    sel, cel = np.sin(el), np.cos(el)

    R = np.array([
        [0, -d, 0],
        [d,  0, 0],
        [0,  0, 1]
    ])
    Ry = np.array([
        [ cel, 0, sel],
        [   0, 1,   0],
        [-sel, 0, cel]
    ])
    Rx = np.array([
        [1,   0,    0],
        [0, caz, -saz],
        [0, saz,  caz]
    ])
    return isce3.core.Quaternion(R @ Ry @ Rx)


class LegacyRaw(RawBase, family='nisar.productreader.raw'):
    """
    Reader for legacy L0B format.  Specicifally this corresponds to
    git commit ab2fcca of the PIX repository at
        https://github-fn.jpl.nasa.gov/NISAR-ADT/NISAR_PIX
    which occurred on 2019-09-09.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        log.warning("Using deprecated L0B format.")
        # XXX Default configuration used in NISAR sims.
        self.rcs2body = get_rcs2body(side=self.identification.lookDirection)

    @property
    def TelemetryPath(self):
        return f"{self.ProductPath}/telemetry"

    _pulseMetaPath = RawBase.BandPath

    def getAttitude(self):
        old = super().getAttitude()
        # XXX Big kludge: convert body2ecef to rcs2ecef.
        # Depends on self.rcs2body being set correctly.
        qs = [body2ecef * self.rcs2body for body2ecef in old.quaternions]
        return isce3.core.Attitude(old.time, qs, old.reference_epoch)



class Raw(RawBase, family='nisar.productreader.raw'):
    # TODO methods for new telemetry fields.
    pass


def open_rrsd(filename) -> RawBase:
    """Open a NISAR L0B file (RRSD product), returning a product reader of
    the appropriate type.  Useful for supporting multiple variants of the
    evolving L0B product spec.
    """
    # Peek at internal paths to try to determine flavor of L0B data.
    # A good check is the telemetry, which is split into high- and low-rate
    # groups in the 2020 updates.
    with h5py.File(filename, 'r', libver='latest', swmr=True) as f:
        if "/science/LSAR/RRSD/telemetry" in f:
            return LegacyRaw(hdf5file=filename)
        return Raw(hdf5file=filename)