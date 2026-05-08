from __future__ import annotations
from .DataDecoder import DataDecoder
import h5py
import isce3
from isce3.focus import RadarPoint, RadarBoundingBox
import logging
from nisar.products.readers import Base
import numpy as np
import pyre
import journal
import re
from warnings import warn
from typing import Tuple
from enum import IntEnum, unique
from nisar.antenna import CalPath, get_calib_range_line_idx
from isce3.core import speed_of_light

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
        fs = isce3.core.speed_of_light / (2 * dr)
        fc = self.getCenterFrequency(frequency, tx)
        return fc, fs, K, T

    def is_tx_off(self, frequency, pol):
        """
        Check whether the transmit (TX) is off for a particular frequency band
        and polarization in the raw (L0B) product.

        Parameters
        ----------
        frequency : str
            Frequency band character such as "A" or "B".
        pol : str
            Transmit-Receive polarization such as "HH", "HV", etc.

        Returns
        -------
        bool
            True if no-transmit for the specified frequency band
            and polarization; otherwise False.

        Notes
        -----
        If both RF center frequency and bandwidth of the TX chirp is zero
        then it is assumed no transmit signal!

        """
        # check if the TX chirp bandwidth or its RF center frequency
        # is zero or not.
        fc, _, rate, pw = self.getChirpParameters(frequency, pol[0])
        return np.isclose(abs(rate * pw), 0) and np.isclose(fc, 0)

    def getRangeBandwidth(self, frequency: str = 'A', tx: str = 'H'):
        """Get RF bandwidth of a desired TX frequency band and pol.

        Parameters
        ----------
        frequency : {'A', 'B'}, optional
            Sub-band
        tx : {'H', 'V', 'L', 'R'}, optional
            Transmit polarization

        Returns
        -------
        float
            Bandwidth in Hz.

        """
        tx_path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[tx_path]["rangeBandwidth"][()]

    @property
    def TelemetryPath(self):
        return f"{self.ProductPath}/lowRateTelemetry"

    # XXX Base.getOrbit has @pyre.export decorator.  What's that do?
    # XXX L0B doesn't put orbit in MetadataPath
    def getOrbit(self):
        path = f"{self.TelemetryPath}/orbit"
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            orbit = isce3.core.load_orbit_from_h5_group(f[path])
        return orbit

    def getAttitude(self):
        path = f"{self.TelemetryPath}/attitude"
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            q = isce3.core.Attitude.load_from_h5(f[path])
        return q

    def getRanges(self, frequency='A', tx='H'):
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            group = f[path]
            r = np.asarray(group["slantRange"])
            dr = group["slantRangeSpacing"][()]
        nr = len(r)
        out = isce3.core.Linspace(r[0], dr, nr)
        assert np.isclose(out[-1], r[-1])
        return out


    def getPulseTimes(self, frequency=None, tx=None, epoch=None):
        """
        Read pulse time tags.

        Parameters
        ----------
        frequency : {'A', 'B'}, optional
            Sub-band.  Typically main science band is 'A'.
            Default is the first frequency in self.frequencies.

        tx : {'H', 'V', 'L', 'R'}, optional
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear), left circular, right circular
            Default is the first pol under `frequency`.

        epoch : isce3.core.DateTime, optional
            Desired time reference.  If not provided the one from the file
            metadata will be used.  The absolute time stamps (epoch + t) are
            identical in either case.

        Returns
        -------
        epoch : isce3.core.DateTime
            UTC time reference

        t : array_like
            Transmit time of each pulse, in seconds relative to epoch.
        """
        if frequency is None:
            frequency = sorted(self.frequencies)[0]
        if tx is None:
            tx = self.polarizations[frequency][0][0]
        txpath = self.TransmitPath(frequency, tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            # FIXME product spec changed UTCTime -> UTCtime
            name = find_case_insensitive(f[txpath], "UTCtime")
            t = np.asarray(f[txpath][name])
            file_epoch = isce3.io.get_ref_epoch(f[txpath], name)
        if epoch is None:
            return file_epoch, t
        t += (file_epoch - epoch).total_seconds()
        return epoch, t

    def getNominalPRF(self, frequency='A', tx='H'):
        """Nominal PRF defined as mean PRF for dithered case.

        Parameters
        ----------
        frequency : {'A', 'B'}, optional
            Sub-band.  Typically main science band is 'A'.

        tx : {'H', 'V', 'L', 'R'}
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear), left circular, right circular

        Returns
        -------
        float
            PRF in Hz.

        """
        _, az_time = self.getPulseTimes(frequency, tx)
        return (az_time.size - 1) / (az_time[-1] - az_time[0])

    def isDithered(self, frequency='A', tx=None, tol=1e-8, num_ignore=0):
        """Whether or not PRF is dithering.

        That is more than one PRF value within entire azimuth duration.

       Parameters
        ----------
        frequency : {'A', 'B'}, optional
            Sub-band.  Typically main science band is 'A'.

        tx : {'H', 'V', 'L', 'R'}, optional
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear), left circular, right circular
            Default is the first pol under `frequency`.

        tol : float, optional
            Tolerance for PRI comparisons in seconds.  Default is less than
            NISAR's 100 ns clock tics.

        num_ignore : int, optional
            The dataset is only considered dithered when more than num_ignore
            consecutive PRIs are not equal.

        Returns
        -------
        bool
            True if multiple PRF values and False if PRF is fixed.    

        """
        if tx is None:
            tx = self.polarizations[frequency][0][0]
        _, az_time = self.getPulseTimes(frequency, tx)
        pri = np.diff(az_time)
        is_unequal = np.abs(np.diff(pri)) > tol
        return np.sum(is_unequal) > num_ignore


    def getCenterFrequency(self, frequency: str = 'A', tx: str = None):
        if tx is None:
            tx = self.polarizations[frequency][0][0]
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[path]["centerFrequency"][()]

    def getListOfTxTRMs(self, frequency: str = 'A', tx: str = None):
        """
        Get list of TR modules used for Transmit.

        Parameters
        ----------
        frequency : {'A', 'B'}
           Sub-band.  Typically main science band is 'A'.
        tx : {'H', 'V', 'L', 'R'}
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear), left circular, right circular. 

        Returns
        -------
        listOfTxTRMs : array_like int
            List of Tx channel indices.
        """

        if tx is None:
            tx = self.polarizations[frequency][0][0]
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[path]["listOfTxTRMs"][()]

    def getListOfRxTRMs(self, frequency: str, polarization: str):
        """
        Get list of TR modules used for Receive.

        Parameters
        ----------
        frequency : {'A', 'B'}
           Sub-band.  Typically main science band is 'A'.
        tx : {'H', 'V', 'L', 'R'}
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear), left circular, right circular. 

        Returns
        -------
        listOfRxTRMs: array_like int
            List of Rx channel indices.
        """

        path = self._rawGroup(frequency, polarization)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[path]["listOfRxTRMs"][()]

    def getRangeLineIndex(self, frequency: str = 'A', tx: str = None):
        """
        Get range line indices.

        Returns range line indices derived from the hardware rangeline counter,
        which starts at 1 at the beginning of a datatake and increases sequentially.
        Except for the first observation within a datatake, the first index will be
        some value other than 1.

        If a rangeline was missed due to corrupted data, for example, that would be
        reflected as a skipped value in the index sequence.

        Parameters
        ----------
        frequency : {'A', 'B'}
           Sub-band.  Typically main science band is 'A'.
        tx : {'H', 'V', 'L', 'R'}
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear), left circular, right circular. 

        Returns
        -------
        rangeLineIndex: array_like int
            List of range line indices.
        """

        if tx is None:
            tx = self.polarizations[frequency][0][0]
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[path]["rangeLineIndex"][()]


    def _parse_chirpcorrelator_from_hrt_qfsp(
            self,
            txrx_pol: str) -> np.ndarray:
        """
        Parse three-tap chirp correlator array with shape (lines, 12, 3)
        as well as cal type with shape (lines,) from high-rate-telemetry
        (HRT) quadrature first-stage processor (QFSP).

        Parameters
        ----------
        txrx_pol : str
            TxRx polarization such as HH, VH, etc

        Returns
        -------
        np.ndarray(complex)
            3-D complex array of chirp correlator with shape (Lines, channels, 3)

        Raises
        ------
        KeyError
            Missing respective dataset in L0B

        See Also
        --------
        getChirpCorrelator

        Notes
        -----
        This function simply parse 3-tap chirp correlator from low-level
        HRT field of L0B as it is, e.g. w/o proper separation of co-pol
        and cross-pol for Quad pol.
        The main reason for `None` as a return value is to support old and
        simulated NISAR L-band L0B products and ISCE3 test data where the
        respective field is missing.

        """
        # get HRT path
        hrt_path = self.TelemetryPath.replace('low', 'high')
        qfsp_path = f'{hrt_path}/tx{txrx_pol[0]}/rx{txrx_pol[1]}/QFSP'
        with h5py.File(self.filename, mode='r', swmr=True) as f5:
            # loop over three qfsp
            for i_qfsp in range(3):
                p_qfsp = f'{qfsp_path}{i_qfsp}'
                # loop over 4 channels per qfsp:
                for nn in range(4):
                    i_chn = nn + i_qfsp * 4
                    n_rx = i_chn + 1
                    # loop over 3 taps per channel
                    for i_tap in range(3):
                        n_tap = i_tap + 1
                        # form the path to the dataset per I and Q
                        # use RX pol!
                        p_ds_i = (f'{p_qfsp}/CHIRP_CORRELATOR_I{n_tap}_'
                                  f'{txrx_pol[1]}{n_rx:02d}')
                        p_ds_q = (f'{p_qfsp}/CHIRP_CORRELATOR_Q{n_tap}_'
                                  f'{txrx_pol[1]}{n_rx:02d}')
                        try:
                            ds_i = f5[p_ds_i]
                        except KeyError as err:
                            warn(
                                f'Missing dataset {p_ds_i} in {self.filename}.'
                                f' Detailed error -> {err}'
                            )
                            raise
                        else:
                            # initialize the 3-D array, lines by 12 by 3
                            if i_qfsp == nn == i_tap == 0:
                                # initialize the 3-D array for chirp correlator
                                num_lines = ds_i.size
                                chp_cor = np.ones((num_lines, 12, 3), dtype='c8')
                            chp_cor[:, i_chn, i_tap].real = ds_i[()]
                            chp_cor[:, i_chn, i_tap].imag = f5[p_ds_q][()]
            return chp_cor


    def _parse_caltype_from_hrt_qfsp(
            self,
            txrx_pol: str) -> np.ndarray:
        """
        Parse cal-path types with shape (lines,) from high-rate telemetry
        (HRT) quadrature first-stage processor (QFSP).

        Parameters
        ----------
        txrx_pol : str
            TxRx polarization such as HH, VH, etc

        Returns
        -------
        np.ndarray(uint8) or None
            1-D array of cal type w/ values HPA=0, LNA=1, BYPASS=2, and
            INVALID=255.

        Raises
        ------
        KeyError
            Missing respective dataset in L0B

        See Also
        --------
        getCalType

        Notes
        -----
        This function simply parse cal path types from low-level
        HRT field of L0B as it is, e.g. w/o proper separation of co-pol
        and cross-pol for Quad pol.

        """
        # get HRT path
        hrt_path = self.TelemetryPath.replace('low', 'high')
        qfsp_path = f'{hrt_path}/tx{txrx_pol[0]}/rx{txrx_pol[1]}/QFSP'
        with h5py.File(self.filename, mode='r', swmr=True) as f5:
            # XXX get caltype from the very first qFSP assuming
            # it is qFSP independent!
            i_qfsp = 0
            p_qfsp = f'{qfsp_path}{i_qfsp}'
            p_type = f'{p_qfsp}/CP_CAL_TYPE_{txrx_pol[1]}{i_qfsp}'
            # XXX Following Try/exception block is added to
            # support old sim L0B products lacking HRT!
            try:
                ds_cal_type = f5[p_type]
            except KeyError as err:
                warn(f'Missing dataset "{p_type}" in '
                    f'"{self.filename}". Detailed error -> {err}')
                raise
            else:
                return ds_cal_type[()].astype(CalPath)


    def _parse_rangeline_index_from_hrt(
            self,
            txrx_pol: str = None) -> np.ndarray:
        """
        Get range line index over all range lines from
        HRT.

        Parameters
        ----------
        txrx_pol : str
            TxRx polarization such as HH, VH, etc

        Returns
        -------
        np.ndarray(uint) or None
            If not available in L0b, None will be returned.

        Raises
        ------
        KeyError
            Missing respective dataset in L0B

        """
        hrt_path = self.TelemetryPath.replace('low', 'high')
        freq_band = sorted(self.frequencies)[0]
        pols = self.polarizations[freq_band]
        if txrx_pol is None:
            txrx_pol = pols[0]
        elif txrx_pol not in pols:
            raise ValueError(f'Available pols {pols} but got {txrx_pol}!')
        rgl_idx_path = (f'{hrt_path}/tx{txrx_pol[0]}/rx{txrx_pol[1]}/'
                        'RangeLine/RH_RANGELINE_INDEX')
        with h5py.File(self.filename, mode='r', swmr=True) as f5:
            try:
                ds_rgl_idx = f5[rgl_idx_path]
            except KeyError as err:
                warn(f'Can not parse range line index from HRT. Error -> {err}')
                raise
            else:
                return ds_rgl_idx[()]


    def getCalType(self, frequency: str = 'A', tx: str = None):
        """
        Extract Tx Calibration mask for each range line.
        HPA = 0, LNA = 1, BYPASS = 2

        Parameters
        ----------
        frequency : {'A', 'B'}
           Sub-band.  Typically main science band is 'A'.
        tx : {'H', 'V', 'L', 'R'}
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear), left circular, right circular. 

        Returns
        -------
        LCAL_INTERVAL: int
            Tx LNA path range line interval, e.g. 1024.
        """

        if tx is None:
            tx = self.polarizations[frequency][0][0]
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[path]["calType"][()]

    def getChirpCorrelator(self, frequency: str = 'A', tx: str = None):
        """
        Extract all 3 taps of 3-tap calibration correlator values for Transmit.

        Parameters
        ----------
        frequency : {'A', 'B'}
           Sub-band.  Typically main science band is 'A'.
        tx : {'H', 'V'}
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear). 

        Returns
        -------
        chirpCorrelator: 3D array of complex
            3-tap correlator values for Transmit.
            size = [num range lines x num chan x 3].
        """

        if tx is None:
            tx = self.polarizations[frequency][0][0]
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[path]["chirpCorrelator"][()]

    def getTxPhase(self, frequency: str = 'A', tx: str = None):
        """
        Extract transmit-path phase values for all channels and
        range lines in degrees.

        Parameters
        ----------
        frequency : {'A', 'B'}
           Sub-band.  Typically main science band is 'A'.
        tx : {'H', 'V'}
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear). 

        Returns
        -------
        txPhase: 2D array of float
            TX-path phases in degrees.
            size = [num range lines x num chan].
        """

        if tx is None:
            tx = self.polarizations[frequency][0][0]
        path = self._pulseMetaPath(frequency=frequency, tx=tx)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as f:
            return f[path]["txPhase"][()]

    def getCaltone(self, frequency='A', polarization=None):
        """Get complex caltone coefficients for all channels and range lines.

        Caltone coefficients are complex values obtained from pulsed CW
        (continuous wave) signal of each RX channel.

        Parameters
        ----------
        frequency : {'A', 'B'}
            Sub-band.  Typically main science band is 'A'.
        polarization : {'HH', 'HV', 'VH', 'VV', 'RH','RV', 'LH', 'LV'}, optional
            Transmit-Receive polarization. If not specified, the first
            polarization in the `frequency` band will be used.

        Returns
        -------
        np.ndarray(complex)
            2-D complex float of caltone (CW) coefficients,
            size = [rangelines x channels].

        """
        if polarization is None:
            polarization = self.polarizations[frequency][0]
        path_txrx = self._rawGroup(frequency, polarization)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            return fid[path_txrx]["caltone"][()]

    def getRD(self, frequency='A', polarization=None):
        """
        Get round trip RX sample index (RD) of digital beam formed (DBF)
        range lines. These integer indexes are provided at ADC clock rate
        (240MHz for NISAR) for all RX channels and range lines.

        Parameters
        ----------
        frequency : {'A', 'B'}
            Sub-band.  Typically main science band is 'A'.
        polarization : {'HH', 'HV', 'VH', 'VV', 'RH','RV', 'LH', 'LV'}, optional
            Transmit-Receive polarization. If not specified, the first
            polarization in the `frequency` band will be used.

        Returns
        -------
        np.ndarray(uint32)
            2-D array of integer values with shape = (rangelines, channels).

        """
        if polarization is None:
            polarization = self.polarizations[frequency][0]
        path_txrx = self._rawGroup(frequency, polarization)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            return fid[path_txrx]["RD"][()]

    def getWD(self, frequency='A', polarization=None):
        """
        Get start sample index of RX data window (WD) w.r.t. RD of digital beam
        formed (DBF) range lines. These integer indexes are provided at ADC
        clock rate (240MHz for NISAR) for all RX channels and range lines.

        Parameters
        ----------
        frequency : {'A', 'B'}
            Sub-band.  Typically main science band is 'A'.
        polarization : {'HH', 'HV', 'VH', 'VV', 'RH','RV', 'LH', 'LV'}, optional
            Transmit-Receive polarization. If not specified, the first
            polarization in the `frequency` band will be used.

        Returns
        -------
        np.ndarray(uint32)
            2-D array of integer values with shape = (rangelines, channels).

        """
        if polarization is None:
            polarization = self.polarizations[frequency][0]
        path_txrx = self._rawGroup(frequency, polarization)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            return fid[path_txrx]["WD"][()]

    def getWL(self, frequency='A', polarization=None):
        """
        Get RX data window length (WL) of digital beam formed (DBF) range
        lines. These integer values are provided at ADC clock rate
        (240MHz for NISAR) for all RX channels and range lines.

        Parameters
        ----------
        frequency : {'A', 'B'}
            Sub-band.  Typically main science band is 'A'.
        polarization : {'HH', 'HV', 'VH', 'VV', 'RH','RV', 'LH', 'LV'}, optional
            Transmit-Receive polarization. If not specified, the first
            polarization in the `frequency` band will be used.

        Returns
        -------
        np.ndarray(uint32)
            2-D array of integer values with shape = (rangelines, channels).

        """
        if polarization is None:
            polarization = self.polarizations[frequency][0]
        path_txrx = self._rawGroup(frequency, polarization)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            return fid[path_txrx]["WL"][()]

    def getSampleRateDBF(self, frequency="A", polarization=None):
        """
        Get sample rate corresponding to RD, WD, WL timing metadata.  For
        NISAR this is always 240 MHz, but it may be different for simulated
        datasets.

        Parameters
        ----------
        frequency : {'A', 'B'}
            Sub-band.  Typically main science band is 'A'.
        polarization : {'HH', 'HV', 'VH', 'VV', 'RH','RV', 'LH', 'LV'}, optional
            Transmit-Receive polarization. If not specified, the first
            polarization in the `frequency` band will be used.

        Returns
        -------
        fs : float
            Sample rate in Hz
        """
        if polarization is None:
            polarization = self.polarizations[frequency][0]
        path_txrx = self._rawGroup(frequency, polarization)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            group = fid[path_txrx]
            key = "sampleRateDBF"
            if key in group:
                return group[key][()]
            else:
                log.info("sampleRateDBF not found in L0B, assuming 240 MHz")
                return 240e6

    def getRdWdWl(self, frequency='A', polarization=None):
        """
        Get all three DBF-related parameters "RD", "WD", and "WL" in one
        place for convenience.

        Parameters
        ----------
        frequency : {'A', 'B'}
            Sub-band.  Typically main science band is 'A'.
        polarization : {'HH', 'HV', 'VH', 'VV', 'RH','RV', 'LH', 'LV'}, optional
            Transmit-Receive polarization. If not specified, the first
            polarization in the `frequency` band will be used.

        Returns
        -------
        RD : np.ndarray(uint32)
            2-D array of integer values with shape = (rangelines, channels)
        WD : np.ndarray(uint32)
            2-D array of integer values with shape = (rangelines, channels)
        WL : np.ndarray(uint32)
            2-D array of integer values with shape = (rangelines, channels)

        See Also
        --------
        getRD
        getWD
        getWL

        """
        if polarization is None:
            polarization = self.polarizations[frequency][0]
        path_txrx = self._rawGroup(frequency, polarization)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            return (fid[path_txrx]["RD"][()], fid[path_txrx]["WD"][()],
                    fid[path_txrx]["WL"][()])

    def getRadarGrid(self, frequency='A', tx='H', prf=None, epoch=None):
        """
        Return the timestamps and radar grid for the raw data.  Since the actual
        azimuth grid may be irregular due to PRF dithering, the azimuth grid
        metadata will be filled with nominal values according to the optional
        `prf` parameter.

        Parameters
        ----------
        frequency : {'A', 'B'}, optional
            Sub-band.  Typically main science band is 'A'.
        tx : {'H', 'V'}, optional
            Transmit polarization.  Abbreviations correspond to horizontal
            (linear), vertical (linear). Defaults to 'H'.
        prf : float, optional
            Pulse repetition frequency in Hz.  If provided, use as grid.prf and
            set grid.length to match the total time span.  If not provided,
            then grid.length is equal to number of pulses and grid.prf is the
            inverse of the mean pulse interval.
        epoch : isce3.core.DateTime, optional
            Desired time reference.  If not provided the one from the file
            metadata will be used.  The absolute time stamps (epoch + t) are
            identical in either case.

        Returns
        -------
        t : np.ndarray[float]
            Time of each pulse in seconds relative to grid.ref_epoch.
        grid : isce3.product.RadarGridParameters
            Grid parameters describing posting of raw data.
        """
        fc = self.getCenterFrequency(frequency, tx)
        wvl = isce3.core.speed_of_light / fc
        r = self.getRanges(frequency, tx)
        epoch, t = self.getPulseTimes(frequency, tx, epoch=epoch)
        nt = len(t)
        assert nt > 1
        if prf:
            nt = 1 + int(np.ceil((t[-1] - t[0]) * prf))
        else:
            prf = (nt - 1) / (t[-1] - t[0])
        side = self.identification.lookDirection
        grid = isce3.product.RadarGridParameters(
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


    def getSubSwathBboxes(self, frequency, polarization=None, epoch=None, num_ignore=0):
        """
        Return the bounding box for each sub-swath.

        Parameters
        ----------
        frequency : {"A", "B"}
            Sub-band identifier.
        polarization : {'HH', 'HV', 'VH', 'VV', 'RH','RV', 'LH', 'LV'}, optional
            Transmit-Receive polarization. If not specified, the first
            polarization in the `frequency` band will be used.
        epoch : isce3.core.DateTime
            Reference epoch for azimuth time tags.
        num_ignore : int, optional
            Number of pulses to ignore at the end of an observation.  This
            option is useful when a fixed-PRF observation is followed by a
            dithered-PRF one, in which case the gaps in the last few receive
            windows will have irregular spacing due to the dithered pulses in
            the air.

        Returns
        -------
        bboxes : list[list[RadarBoundingBox]]
            Bounding box in radar coordinates for each sub-swath for each
            segment of constant data window position/length.
        """
        if polarization is None:
            polarization = self.polarizations[frequency][0]
        tx = polarization[0]
        times, grid = self.getRadarGrid(frequency, tx, epoch=epoch)
        nt, nr = grid.shape
        subswaths = self.getSubSwaths(frequency, tx=tx)
        is_dithered = self.isDithered(frequency, tx=tx, num_ignore=num_ignore)
        rd, wd, wl = self.getRdWdWl(frequency, polarization)

        # Replace enormous fill values with number of samples.
        subswaths = np.where(subswaths > nr, nr, subswaths)

        # Replace last num_ignore pulses with previous value.
        if (not is_dithered) and (num_ignore > 0):
            # Avoid problems with trivially short observations, though this
            # shouldn't ever happen.
            if num_ignore > nt:
                log.warning(f"Asked to ignore {num_ignore} pulses but there "
                    f"are only {nt} total.")
                num_ignore = nt
            # NOTE Need singleton middle dimension for proper broadcasting.
            i = subswaths.shape[1] - num_ignore
            subswaths[:, i:, :] = subswaths[:, i:(i + 1), :]

        # For dithered replace subswaths (gap mask) with a single subswath
        # that merely tracks min/max valid sample.  Note that gaps may still
        # interfere with min/max, though.
        if is_dithered:
            # Determine non-empty ranges.
            starts, ends = subswaths[..., 0], subswaths[..., 1]
            valid = ends > starts
            # Construct masked arrays to simplify stats.
            starts = np.ma.array(subswaths[..., 0], mask=~valid)
            ends = np.ma.array(subswaths[..., 1], mask=~valid)

            # Get masked min and max valid sample for each pulse.
            min_starts = np.min(starts, axis=0)
            max_ends = np.max(ends, axis=0)

            # Now replace subswaths with a single subswath with start/end
            # so we can use same logic as fixed PRF below.
            subswaths = np.vstack((min_starts, max_ends)).transpose().reshape(
                (1, -1, 2))

        changes = get_dwp_change_indices(rd, wd, wl)

        # Append first and last pulses to generate pairs of constant DWP.
        breaks = np.hstack(([0], changes, [grid.shape[0] - 1]))
        bbox_lists = []
        for ibreak in range(len(breaks) - 1):
            ipulse0, ipulse1 = breaks[ibreak], breaks[ibreak + 1]
            t0, t1 = times[ipulse0], times[ipulse1]  # one past end point
            bboxes = []
            for iswath, (j0, j1) in enumerate(subswaths[:, ipulse0, :]):
                # Exclude empty subswaths.
                if j1 <= j0:
                    continue
                # If dithered peek ahead in case gap overlaps start or end of
                # valid swath.  Only need to check one pulse ahead assuming
                # dither sequence is correctly designed to avoid consecutive
                # gaps.
                if is_dithered:
                    assert iswath == 0  # due to restructuring above
                    assert ipulse0 < (nt - 1)  # from construction of breaks
                    j0next = subswaths[iswath, ipulse0 + 1, 0]
                    j1next = subswaths[iswath, ipulse0 + 1, 1]
                    if j1next > j0next:
                        j0 = min(j0, j0next)
                        j1 = max(j1, j1next)
                    else:
                        log.warning(f"Pulse {ipulse0 + 1} immediately after "
                            "DWP change has no valid data.  Mask may be wrong.")
                r0 = grid.slant_ranges[j0]
                r1 = grid.slant_ranges[j1 - 1] + grid.slant_ranges.spacing
                bboxes.append(RadarBoundingBox(
                    RadarPoint(t0, r0),
                    RadarPoint(t1, r1)))
            if len(bboxes) == 0:
                log.warning(f"no valid subswath for time interval [{t0}, {t1})")
                continue
            bbox_lists.append(bboxes)

        return bbox_lists


    def getProductLevel(self):
        '''
        Returns the product level
        '''
        return "L0B"


    def getBasebandPhaseCorrection(self, frequency='A', polarization=None):
        """
        Get the phasor needed to rotate the raw data to have a constant phase
        with respect to the transmit.  This is required for sensors like NISAR
        where the raw data phase is constant with respect to the opening of the
        receive window and the receive window timing may change during an
        observation.

        Parameters
        ----------
        frequency : {'A', 'B'}
            Sub-band.  Typically the main science band is 'A', which is the
            default value.
        polarization : {'HH', 'HV', 'VH', 'VV', 'RH','RV', 'LH', 'LV'}, optional
            Transmit-Receive polarization. If not specified, the first
            polarization in the `frequency` band will be used.

        Returns
        -------
        np.ndarray[np.complex64]
            Array of complex values with shape = (rangelines,) whose elements at
            each index should be multiplied into all samples at the
            corresponding rangeline.

        Notes
        -----
        This method does not actually require that the raw data file contains a
        dataset that stores this correction.  If it doesn't, an array of ones
        will be returned instead (which imparts no phase rotation).
        """
        if polarization is None:
            polarization = self.polarizations[frequency][0]
        naz = self.getRawDataset(frequency, polarization).shape[-2] # beware DM2
        path_txrx = self._rawGroup(frequency, polarization)
        with h5py.File(self.filename, 'r', libver='latest', swmr=True) as fid:
            group = fid[path_txrx]
            key = "basebandPhaseCorrection"
            return group.get(key, np.ones(naz, np.complex64))[()]


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


def get_dwp_change_indices(rd, wd, wl):
    """
    Determine the pulses where the data window position changes.

    Parameters
    ----------
    rd, wd, wl : np.ndarray
        Arrays with shape (num_channels, num_pulses) containing the range delay,
        window delay, and window length DBF parameters.  They must all be
        provided in the same units.

    Returns
    -------
    indices : np.ndarray
        The pulses i where either the min(RD+WD) or the max(RD+WD+WL) changes.
        That is the pulse at i will have a different data window position than
        pulse (i - 1).
    """
    if not (rd.shape == wd.shape == wl.shape):
        raise ValueError("shape mismatch among inputs")
    if not rd.ndim == 2:
        raise ValueError("expected 2D input data")
    start = np.min(rd + wd, axis=1).astype(np.int64)
    end = np.max(rd + wd + wl, axis=1).astype(np.int64)
    location = np.vstack((start, end))
    return np.where(np.any(np.diff(location, axis=1) != 0, axis=0))[0] + 1


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


@unique
class PolarizationTypeId(IntEnum):
    """Enumeration for polarization types of L-band NISAR
    """
    single_h = 0
    """Single Pol HH"""
    single_v = 1
    """Single Pol VV"""
    dual_h = 2
    """Dua Pol HH/HV"""
    dual_v = 3
    """Dual Pol VV/VH"""
    quad = 4
    """Linear Quad Pol HH/HV/VH/VV"""
    compact = 5
    """Left Compact Pol LH/LV"""
    none = 6
    """Unknown"""
    quasi_quad = 7
    """Quasi Linear Quad Pol HH/HV(A) + VV/VH(B)"""
    quasi_dual = 8
    """Quasi Dual Pol HH(A) + VV(B)"""


# helper functions that uses Raw as input

def polarization_type_from_raw(raw: Raw) -> PolarizationTypeId:
    """
    Get polarization ID and type from L0B DRT.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
        L0B raw parser object

    Returns
    --------
    nisar.products.readers.Raw.PolarizationTypeId
        An enumeration for various polarimetric modes of L-band NISAR.

    Raises
    ------
    KeyError
        Missing respective polarization type dataset in L0B

    """
    pol_path = f'{raw.TelemetryPath}/DRT/MISC/CP_IFSW_POLARIZATION'
    with h5py.File(raw.filename, mode='r', swmr=True) as f5:
        try:
            ds_pol = f5[pol_path]
        except KeyError as err:
            warn(f'Missing dataset "{pol_path}" in "{raw.filename}". '
                 'Detailed err -> {err}.')
            raise
        else:
            i_pol = ds_pol[()]
            id_pol = np.nanmedian(i_pol)
            return PolarizationTypeId(id_pol)


def is_raw_quad_pol(raw: Raw) -> bool:
    """
    Determine whether NISAR raw L0B product is
    linear Quad or not.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
        L0B raw parser object

    Returns
    --------
    bool
        True if the L0B product is quad pol otherwise False.

    Notes
    -----
    If the polarization type is missing in the raw (KeyError),
    It will issue a warning and check the polarizations under
    the main frequency band.
    This behaviour is needed for now to avoid failure in some ISCE3 test
    cases due to simulated or old L0B products that do not contain
    polarization type/id field! However, this is subject to change
    in the future.

    """
    try:
        pol_type = polarization_type_from_raw(raw)
    except KeyError as err:
        # XXX If the polarization type is missing in the raw (KeyError),
        # It will issue a warning and check the polarizations under
        # the main frequency band.
        # This behaviour is needed for now to avoid failure in some ISCE3 test
        # cases due to simulated or old L0B products that do not contain
        # polarization type/id field! However, this is subject to change
        # in the future once some ISCE3 test files are updated!
        warn(f'Polarization type is missing in L0B due to error "{err}". '
             'Check for polarizations under the main frequency band. '
             'Outcome might be wrong!')
        freq = np.sort(raw.frequencies)[0]
        pols = raw.polarizations[freq]
        tx_pol = pols[0][0]
        return (len(pols) == 4 and tx_pol in ('H', 'V'))
    else:
        return pol_type == PolarizationTypeId.quad


def first_tx_pol_for_quad(raw: Raw) -> str:
    """
    Determine first TX polarization, H or V, from only linear Quad pol product

    Parameters
    ----------
    raw : nisar.products.readers.Raw
        L0B raw parser object

    Returns
    --------
    str
        TX polarization that transmitted first in the linear quad pol mode.

    Raises
    ------
    ValueError
        If L0B product is not linear quad pol.
    KeyError
        If the polarization type dataset or range line index
        is missing in L0B.

    Notes
    -----
    For NISAR L-band linear quad pol, V is transmitted on
    odd range line index while H is on even one.
    Preferably by checking the first range line index from HRT,
    the first TX polarization can be reliably determined.
    If such information does not exist in L0B product,
    the first individual range line index per TX polarization
    is compared for the smallest index to represent first
    TX pol.

    """
    if not is_raw_quad_pol(raw):
        raise ValueError('Not a quad pol!')
    try:
        idx_rgl = raw._parse_rangeline_index_from_hrt()
    except KeyError:
        # if not in HRT parse single-pol version from swath path
        idx_rgl_h = raw.getRangeLineIndex('A', 'H')[0]
        idx_rgl_v = raw.getRangeLineIndex('A', 'V')[0]
        if idx_rgl_v < idx_rgl_h:
            return 'V'
        return 'H'
    else:  # odd range line is V pol first and even is H pol first!
        return {0: 'H', 1: 'V'}.get(idx_rgl[0] % 2)


def opposite_linear_pol(pol: str) -> str:
    """Get the oppsoite linear pol
    Parameters
    ----------
    pol : str
        Linear pol, H or V

    Returns
    -------
    str
        H if `pol=V` and V if `pol=H`

    Raises
    ------
    ValueError
        If input pol is circular `L` or `R`.

    """
    if pol == 'H':
        return 'V'
    elif pol == 'V':
        return 'H'
    else:
        raise ValueError(
            f'Expected linear pol "H" or "V" but got "{pol}"!')


def chirpcorrelator_caltype_from_raw(
        raw: Raw,
        txrx_pol: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse three-tap chirp correlator array with shape (lines, 12, 3)
    as well as cal type with shape (lines,) from Raw L0B for a certain
    TxRX pol

    Parameters
    ----------
    raw : nisar.products.readers.Raw
    txrx_pol : str
        TxRx polarization such as HH, VH, etc

    Returns
    -------
    np.ndarray(complex)
        3-D complex array of chirp correlator with shape (Lines, channels, 3)
    np.ndarray(uint8)
        1-D array of cal type w/ values HPA=0, LNA=1, BYPASS=2, and INVALID=255

    Notes
    -----
    Assumptions for NISAR L-band products in regards
    to sniffer (BYPASS/LNA) pulses:
    - The first TX pulse in any datatake is sniffer pulse (BYPASS).
    - The period of sniffer pulse is always an even number!
    - The tx=V on odd TX pulses while tx=H on even ones in Quad pol.

    """
    try:
        chp_cor = raw._parse_chirpcorrelator_from_hrt_qfsp(txrx_pol=txrx_pol)
        cal_type = raw._parse_caltype_from_hrt_qfsp(txrx_pol=txrx_pol)
    except KeyError:
        # XXX if the respective field does not exist then use co-pol under
        # swath in L0B for the sake of backward compatibility
        freq_band = [f for f in raw.frequencies if
                     txrx_pol in raw.polarizations[f]][0]
        chp_cor = raw.getChirpCorrelator(freq_band, txrx_pol[0])
        cal_type = raw.getCalType(freq_band, txrx_pol[0])
        return chp_cor, cal_type
    # Linear Quad Pol (QP) case:
    # TX Cal path type under HRT/QFSP is the same for all
    # polarizations "txrx_pol". That is, no difference between
    # H and V!
    # Both cal type and chirp correlators under HRT/QFSP are
    # provided over all TX pulses at fastest PRF clock!
    if is_raw_quad_pol(raw):
        tx_pol_first = first_tx_pol_for_quad(raw)
        # check input TX pol against first TX pol in
        # linear QP to get proper single-pol indexing (slow PRF)
        # from fast indexing (fast PRF) of HRT.
        if tx_pol_first == txrx_pol[0]:
            first_slice = np.s_[::2]
            second_slice = np.s_[1::2]
        else:  # the opposite pol
            first_slice = np.s_[1::2]
            second_slice = np.s_[::2]
        # Now check if the input TX pol is H or V.
        if txrx_pol[0] == 'V':
            # TX pol = V. No special treatment.
            chp_cor = chp_cor[first_slice]
            cal_type = cal_type[first_slice]
        else:
            # TX = H pol that requires special
            # treatment for sniffer pulses (LNA/BYPASS)!
            # get sniffer pulses from the opposite TX
            # (other pulse indices) but same receive
            # to update chirp correlator and cal type.
            cal_type_x = cal_type[second_slice]
            chp_cor_x = chp_cor[second_slice]
            _, idx_byp, idx_lna, _ = get_calib_range_line_idx(cal_type_x)
            # Parse and update the chirp correlator and cal
            # type for BYPASS/LNA (sniffer pulses)
            chp_cor = chp_cor[first_slice]
            cal_type = cal_type[first_slice]
            chp_cor[idx_byp] = chp_cor_x[idx_byp]
            chp_cor[idx_lna] = chp_cor_x[idx_lna]
            cal_type[idx_byp] = CalPath.BYPASS
            cal_type[idx_lna] = CalPath.LNA
    # set x-pol HPA to INVALID given they are the mix of
    # LNA from co-pol and HPA from x-pol!
    if txrx_pol in ('HV', 'VH'):
        idx_hpa, _, _, _ = get_calib_range_line_idx(cal_type)
        if idx_hpa.size > 0:
            log.info(f'Set HPA cal type for x-pol {txrx_pol} to INVALID!')
            cal_type[idx_hpa] = CalPath.INVALID
    return chp_cor, cal_type


def caltone_frequency_from_raw(
        raw: Raw,
        txrx_pol: str
) -> float:
    """get caltone frequency in Hz from low rate telemetry in L0B

    Parameters
    ----------
    raw : nisar.products.readers.Raw
    txrx_pol : str
        TxRx polarization such as HH, VH, etc

    Returns
    -------
    float
        Caltone frequency in Hz.

    Notes
    -----
    If the resepctive DRT field is not found in L0B, caltone frequency
    will be set to 1214.883 MHz.

    """
    # default caltone if dataset is not available (Hz)
    default = 1214.883e6
    # frequency of local oscillator (Hz)
    lo = 1200e6
    # ADC clock (Hz)
    clock = 240e6
    c_p = (f'{raw.TelemetryPath}/DRT/MISC/CP_IFSW_CALTONE_PHASE_STEP_'
           f'{txrx_pol[1]}')
    with h5py.File(raw.filename, mode='r', swmr=True) as f5:
        try:
            ds_caltone_phase = f5[c_p]
        except KeyError:
            warn(f'Missing path "{c_p}" in L0B! Caltone frequency will '
                 f'be set to {default} (Hz)')
            return default
        else:
            i_cal = np.median(ds_caltone_phase[()]).astype(int)
            if i_cal < 2**16:
                warn('CALTONE_PHASE_STEP seems too small, '
                     'caltone frequency may be invalid')
            caltone_freq = (i_cal / 2**32) * clock + lo
            return caltone_freq


def range_delay_sequential_tx_from_raw(
        raw: Raw,
        freq_band: str,
        txrx_pol: str
) -> float:
    """
    Get range delay (seconds) of the second pulse wrt the pulsewidth
    of the first TX pulse in sequential split-spectrum transmit
    for L0B for specific frequency band and polarization if exists.

    Parameters
    ----------
    raw : nisar.products.readers.Raw
    freq_band: str
        Frequency band such A or B.
    txrx_pol : str
        TxRx polarization such as HH, VH, etc

    Returns
    -------
    float
        Delay in seconds

    Notes
    -----
    It is assumed that slant ranges of A and B properly represent
    the start of a first valid samples and all instrument relative
    delays between A and B products due to onboard digital filetrs
    are already corrected for.
    Alternatively, one can use pulsewidth of band = A as range delay
    for band B. But due to different bandwidths and onboard digital
    filters, the respective group delays are diffrent and needs to
    be taken into account on top of pulsewidth.

    """
    # check if band is B and it is split spectrum
    if freq_band == 'B' and len(raw.frequencies) == 2:
        pols = raw.polarizations
        # check if this is sequential transmit
        if txrx_pol in pols['A']:
            sr_b = raw.getRanges('B', txrx_pol[0])
            sr_a = raw.getRanges('A', txrx_pol[0])
            delay = 2 * (sr_b.first - sr_a.first) / speed_of_light
            return delay
    return 0.0
