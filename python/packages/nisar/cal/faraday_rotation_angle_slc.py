"""
It contains functions/classes for Faraday rotation angle estimation
from SLC data.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import os
from tempfile import TemporaryDirectory, NamedTemporaryFile
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from scipy import fft
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from isce3.core import (
    DateTime, TimeDelta, speed_of_light, Ellipsoid, LUT2d, LLH
)
from isce3.geometry import geo2rdr
from nisar.log import set_logger
from nisar.cal import est_peak_loc_cr_from_slc
from nisar.cal.pol_channel_imbalance_slc import (
    BlockInfo, _check_quadpol_band_slc,
    _get_block_size_slice, _get_bin_limits
)


@dataclass
class FaradayAngleProductSlc:
    """
    Faraday Rotation (FR) Angle Product estimated from quad-pol RSLC
    product in RSLC grid.

    Attributes
    ----------
    faraday_ang : 2-D array of float
        One-way Faraday rotation angle in (rad) at center frequency with
         shape (azimuth blocks, range blocks).
    magnitude : 2-D array of float
        Absolute magnitude (linear) of the FR estimator.
        The shape of 2-D array is (azimuth blocks, range blocks).
    slant_range : 1-D array of float
        Slant ranges in (m) at the center of each block.
    az_datetime : 1-D sequence of isce3.core.DateTime
        Azimuth (AZ) datetime in (UTC) at the center of each block.
    slope_freq : 2-D array or None, optional
        Slope (radians/Hz) in fast-time frequency domain defined as
        FR(deg)/1(MHz) with shape (azimuth blocks, range blocks).
        This is only available when using frequency domain approach.

    """
    faraday_ang: np.ndarray
    magnitude: np.ndarray
    slant_range: np.ndarray
    az_datetime: List[DateTime]
    slope_freq: Optional[np.ndarray] = None


@dataclass
class FaradayAngleProductCR:
    """
    Faraday Rotation (FR) Angle Product estimated from quad-pol RSLC
    in RSLC grid containing Corner reflectors (CR).

    Attributes
    ----------
    faraday_ang : float
        One-way Faraday rotation angle in (rad) at center frequency.
    magnitude : float
        Absolute magnitude of FR estimator.
    slant_range : float
        Slant ranges in (m).
    az_datetime : isce3.core.DateTime
        Azimuth (AZ) datetimes in (UTC) of the platform when CR is observed at
        the beam center of the antenna (electrical boresight) for native
        doppler RSLC geometry or at the closest distance for zero-doppler RSLC
        geometry (NISAR default).
    llh : isce3.core.LLH
        The geodetic coordinates of the corner reflector: the geodetic
        longitude and latitude in radians and the height wrt WGS 84 ellipsoid
        in meters.
    el_ant : float
        Elevation (EL) angle (rad) in antenna frame.
    az_ant : float
        Azimuth (AZ) angle (rad) in antenna frame
    slope_freq : float or simply None, optional
        Slope (radians/Hz) in fast-time frequency domain defined as
        FR(deg)/1(MHz). This is only available when applying frequency
        domain approach.

    """
    faraday_ang: float
    magnitude: float
    slant_range: float
    az_datetime: DateTime
    llh: LLH
    el_ant: float
    az_ant: float
    slope_freq: Optional[float] = None


def faraday_rot_angle_from_cr(slc, cr_llh, *, freq_band='A', num_pixels=16,
                              doppler=LUT2d()):
    """
    Estimate Faraday rotation angle from radiometrically and polarimetrically
    calibrated linear quad-pol RSLC product over one or more trihedral corner
    reflectors.

    The estimator is based on scattering matrix, the first order approach
    proposed by [Freeman2004]_. Although, this approach is not suitable for
    random distributed targets, it is perfect for point-like targets like
    trihedral corner reflectors (CR).

    Note that the desired [peak] region of CR for each received polarization
    is defined by its respective co-pol impulse response given there is ideally
    no cross-pol return from an ideal CR with high signal-to-clutter ratio!

    Parameters
    ----------
    slc : nisar.products.readers.SLC
    cr_llh : 1-D/2-D sequence or array of three floats
        Approximate Geodetic Longitude, latitude, height in (rad, rad, m) of
        CR(s). For more than one CR, the 2-D array shall have shape `Nx3`
        where  `N` is number of CRs.
    freq_band : {'A', 'B'}, default='A'
        Frequency band char of the RSLC product
    num_pixels : int, default=16
        Number of pixels used in interpolation around each corner reflector
        in both range and azimuth of the respective RSLC.
    doppler : isce3.core.LUT2d, default=zero Doppler
        Doppler in (Hz) as a function azimuth time and slant range in
        the form 2-D LUT used for RSLC. The default assumes zero-Doppler
        geometry for RSLC radar grid where CRs are located and resampled.

    Returns
    -------
    List of FaradayAngleProductCR
        Faraday rotation angle products for all CR reflectors within the
        same RSLC product.

    Raises
    ------
    ValueError
        For bad or out of range frequency.
        RSLC does not contain all four products {HH, HV, VH, VV}. That is,
        RSLC is not a linear quad-pol product!
    RuntimeError
        No corner reflector is found within the RSLC boundary minus the
        margins defined by `num_pixels`.

    Warnings
    --------
    OutOfSlcBoundWarning
        For "cr_llh" out of RSLC data margins within +/- half
        of `num_pixels`.
        If any CR is outside the bounds of the RSLC radar grid, they
        will be skipped and this warning will be emitted.

    References
    ----------
    .. [Freeman2004] A. Freeman, 'Calibration of linearly polarized
        polarimetric SAR data subject to Faraday rotation,' IEEE Trans.
        Geosci. Remote Sens., Vol 42, pp.1617-1624, August, 2004.

    """
    # check freq_band and quad-pol condition
    _check_quadpol_band_slc(slc, freq_band)

    # get a list of CR info for all CRs
    crs_info = est_peak_loc_cr_from_slc(
        slc, cr_llh, freq_band=freq_band, num_pixels=num_pixels,
        rel_pow_th_db=3.0)

    num_crs = len(crs_info)
    if num_crs == 0:
        raise RuntimeError('No CR is found within the RSLC boundary!')

    # get orbit and radar grid
    orbit = slc.getOrbit()
    radgrid = slc.getRadarGrid(freq_band)

    # loop over CR and generate Faraday rotation angle product
    fr_prod_crs = []
    for cr in crs_info:
        # FR angle = 0.5 arctan(((VH-HV) / (HH+VV)).real) in (radians)
        # take the real part given the expected imaginary part
        # shall be theoretically zero but instead it may be filled with
        # noise + clutter + residual cx-pol of a non-ideal CR!
        fra = .5 * np.arctan(
            ((cr.amp_pol['VH'] - cr.amp_pol['HV']) /
             (cr.amp_pol['HH'] + cr.amp_pol['VV'])).real
        )
        # magnitude of the FR
        mag_fr = np.sqrt(abs(cr.amp_pol['VH'] - cr.amp_pol['HV']) ** 2 +
                         abs(cr.amp_pol['HH'] + cr.amp_pol['VV']) ** 2)
        # get platform az time where CR is observed
        azt, sr = geo2rdr(
            cr.llh, Ellipsoid(), orbit, doppler, radgrid.wavelength,
            radgrid.lookside, maxiter=60, delta_range=0.1
        )
        # convert relative azimuth time to DateTime
        az_dt = orbit.reference_epoch + TimeDelta(azt)

        # form the final Faraday Angle product
        fr_prod_crs.append(
            FaradayAngleProductCR(
                fra, mag_fr, sr, az_dt, LLH(*cr.llh), cr.el_ant, cr.az_ant)
        )

    return fr_prod_crs


class FaradayRotAngleSlc(ABC):
    """
    Abstract base class for a general Faraday rotation (FR) angle estimator
    class which relies on quad-pol RSLC product.

    Parameters
    ----------
    slc : nisar.products.readers.SLC
        Radiometrically calibrated linear quad-pol RSLC product.
    freq_band : {'A', 'B'}, default='A'
        Frequency band used for RSLC product.
    dir_tmp : str, default='.'
        A path for temporary directory containing large intermediate
        memmap binary files as well as PNG plots if any.
    logger : logging.Logger, optional
        If not provided a logger with StreamHandler will be set.
    min_sr_spacing : float, default=400.0
        Min slant range spacing between range blocks in (m).
        The default corresponds to around 50 mdeg in EL for spaceborne.
    min_azt_spacing : float, default=0.25
        Min azimuth time spacing between azimuth blocks in (sec).
        The default corresponds to around 500 pulses for most spaceborne
        cases w/ PRF around 2KHz.
    plot : bool, default=False
        Generates PNG plot stored under `dir_tmp` for final Faraday rotation
        angles and the estimator magnitudes in radar grid.
        Note that if package `matplotlib` does not exist,
        no plots will be generated and a warning will be issued!

    """

    def __init__(self, slc, *, freq_band='A',
                 dir_tmp='.', logger=None, min_sr_spacing=400,
                 min_azt_spacing=0.25, plot=False):

        # Parameters
        self._slc = slc
        self._freq_band = freq_band
        # check freq band and quad pol condition
        _check_quadpol_band_slc(slc, freq_band)

        # check logger
        if logger is None:
            self._logger = set_logger("FaradayRotAngleSlc")
        else:
            self._logger = logger

        # open tmp dir for large binary files
        self._tmpdir = TemporaryDirectory(suffix='_faraday_rot_ang_slc',
                                          dir=dir_tmp)
        # output dir for plots if any
        self._plotdir = dir_tmp

        # get radar grid and swath from SLC
        self._rdr_grid = slc.getRadarGrid(freq_band)

        self._min_sr_spacing = min_sr_spacing
        self._min_azt_spacing = min_azt_spacing

        # set the flag for plotting
        self._plot = plot
        if self._plot:
            if plt is None:
                self._logger.warning(
                    'Missing package "matplotlib"! No plots will be generated!'
                )
                self._plot = False

    def __enter__(self):
        return self

    def __exit__(self, val_, type_, tb_):
        self._tmpdir.cleanup()

    def __repr__(self):
        slc_name = os.path.basename(self._slc.filename)
        return (f'{self.__class__.__name__}('
                f'slc={slc_name}'
                f'freq_band={self.freq_band}'
                f'dir_name_tmp={self.tmp_dir_name}'
                f'logger_name={self.logger.name}'
                f'plot={self._plot})'
                )

    @property
    def freq_band(self) -> str:
        return self._freq_band

    @property
    def tmp_dir_name(self) -> str:
        return os.path.abspath(self._tmpdir.name)

    @property
    def logger(self):
        return self._logger

    def _get_block_info(self, azt_lim, sr_lim, azt_blk_size, sr_blk_size):
        """Get az-sr block size and limits info for extended-scene RSLC.

        Parameters
        ----------
        azt_lim : tuple[float/isce3.core.DateTime, float/isce3.core.DateTime]
            Azimuth time/datetime limit of RSLC extended scene data used for
            estimation. The limit is defined as [first, last] in either
            relative times in (sec) or in DateTimes in (UTC).
        sr_lim : tuple[float, float]
            Slant range limit of the RSLC extended scene data used for
            estimation. The limit is defined as [first, last] of slant range
            in (m).
        azt_blk_size : float
            Max azimuth block size in (sec).
        sr_blk_size : float
            Max slant range block size in (m).

        Returns
        -------
        BlockInfo
            Azimuth bins
        BlockInfo
            Range bins

        Raises
        ------
        ValueError
            For bad block size in range and/or azimuth
        OutOfSlcBoundError
            For AZ and range limits out of SLC boundary

        """
        # first convert any AZ datetime to seconds
        azt_lim_sec = list(azt_lim)
        for nn in range(2):
            if isinstance(azt_lim[nn], DateTime):
                azt_lim_sec[nn] = (azt_lim[nn] -
                                   self._rdr_grid.ref_epoch).total_seconds()
        # check and get bin limits to be within SLC boundary for both azimuth
        # and range
        azb_limit = _get_bin_limits(
            azt_lim_sec, self._rdr_grid.sensing_start,
            self._rdr_grid.az_time_interval,
            self._rdr_grid.length, err_msg='azimuth time limit'
        )

        rgb_limit = _get_bin_limits(
            sr_lim, self._rdr_grid.starting_range,
            self._rdr_grid.range_pixel_spacing, self._rdr_grid.width,
            err_msg='slant range limit'
        )
        # check the block size in slant range
        if sr_blk_size < max(self._min_sr_spacing,
                             self._rdr_grid.range_pixel_spacing):
            raise ValueError('Slant range block size is too small!')
        tot_sr_dist = (self._rdr_grid.slant_range(rgb_limit[1]) -
                       self._rdr_grid.slant_range(rgb_limit[0]))
        if sr_blk_size > tot_sr_dist:
            self.logger.warning('Slant range block size is larger than '
                                f'total range distance {tot_sr_dist:.3f} (m)!'
                                ' It will be set to the max!')
            sr_blk_size = tot_sr_dist

        # check the block size in azimuth
        tot_azt_dur = (self._rdr_grid.sensing_datetime(azb_limit[1]) -
                       self._rdr_grid.sensing_datetime(azb_limit[0])
                       ).total_seconds()
        if azt_blk_size > tot_azt_dur:
            self.logger.warning('Azimuth time block size is larger than total '
                                f'duration {tot_azt_dur:.5f} (sec)!'
                                ' It will be set to the max.')
            azt_blk_size = tot_azt_dur

        if azt_blk_size < self._min_azt_spacing:
            raise ValueError('Azimuth time block size is too small!')

        # get range bins per block and the number of blocks of sr
        n_rgb_blk = round(sr_blk_size / self._rdr_grid.range_pixel_spacing)
        n_blks_sr = int(np.ceil(np.diff(rgb_limit) / n_rgb_blk))

        self.logger.info(f'Full block size in slant range -> {n_rgb_blk}')
        self.logger.info(f'Total number of range blocks -> {n_blks_sr}')

        # get azimuth bins per block and the number of blocks of azt
        n_azb_blk = round(azt_blk_size * self._rdr_grid.prf)
        n_blks_azt = int(np.ceil(np.diff(azb_limit) / n_azb_blk))

        self.logger.info(f'Full block size in AZ time -> {n_azb_blk}')
        self.logger.info(f'Total number of AZ blocks -> {n_blks_azt}')

        # build block slice generator and block size callable
        az_blksz_fun, azb_slice = _get_block_size_slice(
            azb_limit, n_blks_azt, n_azb_blk, self._rdr_grid.length)

        rg_blksz_fun, rgb_slice = _get_block_size_slice(
            rgb_limit, n_blks_sr, n_rgb_blk, self._rdr_grid.width)

        # form block info
        return (BlockInfo(azb_slice, az_blksz_fun, n_blks_azt),
                BlockInfo(rgb_slice, rg_blksz_fun, n_blks_sr))

    @abstractmethod
    def estimate(self, azt_blk_size=5.0, sr_blk_size=3000.0,
                 azt_lim=(None, None), sr_lim=(None, None)):
        """
        Estimates Faraday rotation angle and generate a FR product from
        linear quad-pol RSLC.

        Parameters
        ----------
        azt_blk_size : float, default=5.0
            Max azimuth block size in (sec).
        sr_blk_size : float, default=3000.0
            Max slant range block size in (m).
        azt_lim : tuple[float/isce3.core.DateTime, float/isce3.core.DateTime],
                  optional
            Azimuth time/datetime limit of RSLC extended scene data used for
            estimation. The limit is defined as [first, last] in either
            relative times (sec) or in DateTimes (UTC).
        sr_lim : tuple[float, float], optional
            Slant range limit of the RSLC extended scene data used for
            estimation. Span is defined as [first, last] of slant range (m).

        Returns
        -------
        FaradayAngleProductSlc
            Faraday rotation angles along with its estimator magnitudes
            which can be used as so-called quality measure of the estimates
            over all blocks.

        Raises
        ------
        OutOfSlcBoundError
            If azimuth time and/or slant range limits are out of SLC boundary.
        ValueError
            Unordered values for range and/or azimuth limits.
            Too small block sizes in either range or azimuth.

        """
        pass


class FaradayRotEstBickelBates(FaradayRotAngleSlc):
    """
    Faraday Rotation Angle estimator based on [BICKEL1965]_ estimator
    (circular scattering matrix based estimator) from radiometrically
    and polarimetrically calibrated quad-pol RSLC.

    This method is time-spatial domain approach. However, the frequency
    domain version of this method is also jointly implemented.
    The frequency doamin is based on the slope of the linear
    regression of FR angle as a function of RF frequency to
    extract the mean FR angle at the center of the RF band as described
    in [PAPATHAN2017]_.

    slc : nisar.products.readers.SLC
        Radiometrically calibrated linear quad-pol RSLC product.
    freq_band : {'A', 'B'}, default='A'
        Frequency band used for RSLC product.
    dir_tmp : str, default='.'
        A path for temporary directory containing large intermediate
        memmap binary files as well as PNG plots if any.
    logger : logging.Logger, optional
        If not provided a logger with StreamHandler will be set.
    min_sr_spacing : float, default=400.0
        Min slant range spacing between range blocks in (m).
        The default corresponds to around 50 mdeg in EL for spaceborne.
    min_azt_spacing : float, default=0.25
        Min azimuth time spacing between azimuth blocks in (sec).
        The default corresponds to around 500 pulses for most spaceborne
        cases w/ PRF around 2KHz.
    use_slope_freq : bool, default=False
        Whether or not use slope in frequency-domain to estimate mean FR at
        the center of the fast-time bandwidth of `freq_band`.
    num_cpu_fft : int, optional
        Number of CPUs (workers) used in fast-time FFT if `use_slope_freq`
        is True. If negative, the value wraps around total number of CPUs.
        Thus, `-1` implies that all CPUs will be used.
    ovsf : float, default=1.2
        Oversampling factor as a rato of fast-time sampling rate to TX chirp
        bandwidth. This limits the fast-time frequency coverage over
        which Faraday rotation angle is estimated if `use_slope_freq` is True.
    plot : bool, default=False
        Generates PNG plot stored under `dir_tmp` for final Faraday rotation
        angles and the estimator magnitudes in radar grid. It also
        generates plots of both measured and poly-fitted FR v.s. RF frequencies
        if `use_slope_freq`. Note that if package `matplotlib` does not exist,
        no plots will be generated and a warning will be issued!

    References
    ----------
    .. [BICKEL1965] S. B. Bickel and R. H. T. Bates, 'Effects of
        magneto-ionic propagation on the polarization scattering matrix,'
        Proc. IEEE, vol 53, pp. 1089-1091, August 1965.
    .. [PAPATHAN2017] K. P. Papathanassiou and J. S. Kim, 'Polarimetric
        system calibration in the presence of Faraday rotation,' Proc.
        IGARSS IEEE, pp. 2430-2433, 2017.

    """

    def __init__(self, slc, *, freq_band='A',
                 dir_tmp='.', logger=None, min_sr_spacing=400,
                 min_azt_spacing=0.25, use_slope_freq=False,
                 num_cpu_fft=None, ovsf=1.2,  plot=False):

        super().__init__(slc, freq_band=freq_band, dir_tmp=dir_tmp,
                         logger=logger, min_sr_spacing=min_sr_spacing,
                         min_azt_spacing=min_azt_spacing, plot=plot)

        # get RF center frequency in (MHz)
        swath = slc.getSwathMetadata(freq_band)
        self._fc_mhz = swath.acquired_center_frequency * 1e-6

        self._use_slope_freq = use_slope_freq

        # check number of CPUs
        if num_cpu_fft is not None:
            if num_cpu_fft > 0:
                self.num_cpu_fft = min(num_cpu_fft, os.cpu_count())
            else:
                self.num_cpu_fft = max(1, num_cpu_fft % (os.cpu_count() + 1))
            self.logger.info(f'Number of CPUs for FFT -> {self.num_cpu_fft}')
        else:
            self.num_cpu_fft = num_cpu_fft

        # check oversampling factor
        if ovsf < 1.0:
            raise ValueError('"ovsf" shall be equal or larger than 1.0!')
        self._ovsf = ovsf
        self.logger.info(f'Oversampling factor -> {self._ovsf}')

        # Constant
        # slope (deg/MHz) to angle (deg) scalar @ center frequency
        # used only for frequency approach
        self._slope2ang = -0.5 * self._fc_mhz

        # convert from (deg/MHz) to (rad/Hz) for final slope product
        self._degmhz2radhz = np.deg2rad(1) * 1e-6

    @property
    def ovsf(self) -> float:
        return self._ovsf

    @property
    def use_slope_freq(self) -> bool:
        return self._use_slope_freq

    def _get_freq_slice(self, nfft: int) -> tuple[np.ndarray, slice]:
        """Get frequency vector (MHz) and fast-time freq-bin slice"""
        # compute fast-time sampling frequency (MHz)
        fs_mhz = 1e-6 * speed_of_light / (
            2 * self._rdr_grid.range_pixel_spacing)
        # frequency bins to be excluded out of nfft
        fb_exc = int(0.5 * nfft * (self.ovsf - 1) / self.ovsf)
        # get freq bin slice excluding fb_exc on both ends.
        if fb_exc == 0:
            slice_freq_bin = slice(None)
        else:
            slice_freq_bin = slice(fb_exc, -fb_exc)
        self.logger.info(
            f'Frequency bin slice used for slope -> {slice_freq_bin}')
        # RF frequency vector (MHz)
        freq_mhz = (self._fc_mhz +
                    fs_mhz * fft.fftshift(fft.fftfreq(nfft))[slice_freq_bin])
        self.logger.info(
            '[Min, Max] RF frequencies in FR estimator (MHz, MHz) ->'
            f' [{freq_mhz[0]:.2f}, {freq_mhz[-1]:.2f}]'
        )
        return freq_mhz, slice_freq_bin

    def estimate(self, azt_blk_size=5.0, sr_blk_size=3000.0,
                 azt_lim=(None, None), sr_lim=(None, None)):
        """
        Estimate Faraday Rotation Angle based on [BICKEL1965]_ estimator
        (circular scattering matrix based estimator) from radiometrically
        and polarimetrically calibrated quad-pol RSLC.

        The frequency-domain version of this where the slope of the linear
        regression of FR angle as a function of RF frequency is employed to
        extract the mean FR angle at the center of the RF band is also
        implemenetd [PAPATHAN2017]_.

        Parameters
        ----------
        azt_blk_size : float, default=5.0
            Max azimuth block size in (sec).
        sr_blk_size : float, default=3000.0
            Max slant range block size in (m).
        azt_lim : tuple[float/isce3.core.DateTime, float/isce3.core.DateTime],
                  optional
            Azimuth time/datetime limit of RSLC extended scene data used for
            estimation. The limit is defined as [first, last] in either
            relative times (sec) or in DateTimes (UTC).
        sr_lim : tuple[float, float], optional
            Slant range limit of the RSLC extended scene data used for
            estimation. Span is defined as [first, last] of slant range (m).

        Returns
        -------
        FaradayAngleProductSlc
            Faraday rotation angles along with its magnitudes of estimator
            which can be used as so-called quality measure of the estimates
            over all blocks.

        Raises
        ------
        OutOfSlcBoundError
            If azimuth time and/or slant range limits are out of SLC boundary.
        ValueError
            Unordered values for range and/or azimuth limits.
            Too small block sizes in either range or azimuth.

        Notes
        -----
        If any of values in "azt_lim" and "rg_lim" is None, it will be set
        to its respective start/stop limit of RSLC radar grid.
        The estimator has unambiguous period of pi/2, that is the FR angles
        can be unambiguously estimated within [-pi/4, pi/4].

        References
        ----------
        .. [BICKEL1965] S. B. Bickel and R. H. T. Bates, 'Effects of
            magneto-ionic propagation on the polarization scattering matrix,'
            Proc. IEEE, vol 53, pp. 1089-1091, August 1965.
        .. [PAPATHAN2017] K. P. Papathanassiou and J. S. Kim, 'Polarimetric
            system calibration in the presence of Faraday rotation,' Proc.
            IGARSS IEEE, pp. 2430-2433, 2017.

        """
        # get block info in az/range
        blk_az, blk_rg = self._get_block_info(
            azt_lim, sr_lim, azt_blk_size, sr_blk_size)

        # get decoded RSLC dataset for all pols
        dset_hv = self._slc.getSlcDatasetAsNativeComplex(self.freq_band, 'HV')
        dset_vh = self._slc.getSlcDatasetAsNativeComplex(self.freq_band, 'VH')
        dset_hh = self._slc.getSlcDatasetAsNativeComplex(self.freq_band, 'HH')
        dset_vv = self._slc.getSlcDatasetAsNativeComplex(self.freq_band, 'VV')

        # create tmp files for memory mapping of a block under tmp dir
        fid_co = NamedTemporaryFile(suffix='_rslc_copol.c8',
                                    dir=self.tmp_dir_name,
                                    delete=False)
        fid_cx = NamedTemporaryFile(suffix='_rslc_cxpol.c8',
                                    dir=self.tmp_dir_name,
                                    delete=False)
        fid_rl = NamedTemporaryFile(suffix='_rslc_rl.c8',
                                    dir=self.tmp_dir_name,
                                    delete=False)
        fid_lr = NamedTemporaryFile(suffix='_rslc_lr.c8',
                                    dir=self.tmp_dir_name,
                                    delete=False)

        # memory maps for block analysis using full block size
        block_shape = (blk_az.fun_size(0), blk_rg.fun_size(0))

        mmap_blk_co = np.memmap(fid_co, mode='w+', dtype=np.complex64,
                                shape=block_shape)
        mmap_blk_cx = np.memmap(fid_cx, mode='w+', dtype=np.complex64,
                                shape=block_shape)
        mmap_blk_rl = np.memmap(fid_rl, mode='w+', dtype=np.complex64,
                                shape=block_shape)
        mmap_blk_lr = np.memmap(fid_lr, mode='w+', dtype=np.complex64,
                                shape=block_shape)

        # shape of all output containers
        out_shape = (blk_az.num_blks, blk_rg.num_blks)

        # if frequency-domain slope is requested, define new mmap and fft size
        slope = None
        if self.use_slope_freq:
            self.logger.info('Using slope in fast-time frequency domain!')
            self.logger.info('FR angle is estimated at center '
                             f'frequency (MHz) -> {self._fc_mhz:.3f}')
            # a constant for FR angle estimator in frequency domain
            # It accounts for `OMEGA(f)=2*pi*f` in FFT
            cst_fra_fft = 0.25 / (2 * np.pi)

            # FFT in range direction only! This can be parallalized in AZ
            nfft = fft.next_fast_len(block_shape[1])
            self.logger.info(f'Number FFT points in range -> {nfft}')

            # get frequency bin slice within around [-BW/2, BW/2] and its
            # corresponding RF frequencies (MHz).
            freq_mhz, slice_freq_bin = self._get_freq_slice(nfft)

            # 2-D shape of a block in frequency domain
            block_fft_shape = (block_shape[0], nfft)
            # create memmap for FFT arrays
            fid_rl_fft = NamedTemporaryFile(suffix='_rslc_rl_fft.c8',
                                            dir=self.tmp_dir_name,
                                            delete=False)
            fid_lr_fft = NamedTemporaryFile(suffix='_rslc_lr_fft.c8',
                                            dir=self.tmp_dir_name,
                                            delete=False)
            mmap_blk_rl_fft = np.memmap(
                fid_rl_fft, mode='w+', dtype=np.complex64,
                shape=block_fft_shape
            )
            mmap_blk_lr_fft = np.memmap(
                fid_lr_fft, mode='w+', dtype=np.complex64,
                shape=block_fft_shape
            )
            # initialize output container for slope in (rad/MHz)
            slope = np.ones(out_shape, dtype='f4')

        # container for all range bin slices
        rgb_slice_all = tuple(blk_rg.gen_slice)

        # initialize output containers
        fr_ang = np.ones(out_shape, dtype='f4')
        mag_fr = np.ones(out_shape, dtype='f4')

        # mid azimuth/range bins (float values) for all blocks
        azb_mid_all = np.zeros(blk_az.num_blks, dtype='f4')
        rgb_mid_all = np.zeros(blk_rg.num_blks, dtype='f4')

        # loop over all azimuth (AZ) blocks
        for n_az, azb_slice in enumerate(blk_az.gen_slice):
            # get az block size
            az_blksz = blk_az.fun_size(n_az)
            # get mid azimuth bin for the block
            azb_mid_all[n_az] = 0.5 * (azb_slice.stop + azb_slice.start - 1)

            # loop over all range (RG) blocks
            for n_rg, rgb_slice in enumerate(rgb_slice_all):
                self.logger.info('Processing block (AZ, RG) -> '
                                 f'({azb_slice}, {rgb_slice})')
                # get range block size
                rg_blksz = blk_rg.fun_size(n_rg)

                # get mid range bin for the block
                if n_az == 0:
                    rgb_mid_all[n_rg] = 0.5 * (rgb_slice.stop +
                                               rgb_slice.start - 1)

                # block processing steps
                # sum of co-pols
                mmap_blk_co[:az_blksz, :rg_blksz] = (
                    dset_hh[azb_slice, rgb_slice] +
                    dset_vv[azb_slice, rgb_slice]
                )
                mmap_blk_co[:az_blksz, :rg_blksz] *= 1j

                # diff of X-pols
                mmap_blk_cx[:az_blksz, :rg_blksz] = (
                    dset_vh[azb_slice, rgb_slice] -
                    dset_hv[azb_slice, rgb_slice]
                )

                # form diagonal terms of left/right circular scattering matrix
                # RL = 1j*CO + CX
                mmap_blk_rl[:az_blksz, :rg_blksz] = (
                    mmap_blk_co[:az_blksz, :rg_blksz] +
                    mmap_blk_cx[:az_blksz, :rg_blksz]
                )
                # LR = 1j*CO - CX
                mmap_blk_lr[:az_blksz, :rg_blksz] = (
                    mmap_blk_co[:az_blksz, :rg_blksz] -
                    mmap_blk_cx[:az_blksz, :rg_blksz]
                )

                # Cross-correlator in time domain CX = LR * conj(RL)
                mmap_blk_cx[:az_blksz, :rg_blksz] = (
                    mmap_blk_lr[:az_blksz, :rg_blksz] *
                    mmap_blk_rl[:az_blksz, :rg_blksz].conj()
                )

                # average xcor over all bins within a block to
                # get complex-value estimate
                cmp_est = np.nanmean(mmap_blk_cx[:az_blksz, :rg_blksz])

                # get magnitude of estimator from time-domain correlator
                mag_fr[n_az, n_rg] = abs(cmp_est)
                self.logger.info('Absolute magnitude of FR estimator (linear)'
                                 f' -> {mag_fr[n_az, n_rg]:.3f}')

                # estimate FR angle either from time or frequency domain
                if self.use_slope_freq:  # freq domain
                    # FFT of LR and RL in range
                    mmap_blk_rl_fft[:az_blksz] = fft.fft(
                        mmap_blk_rl[:az_blksz, :rg_blksz],
                        n=nfft, axis=1, workers=self.num_cpu_fft
                    )
                    mmap_blk_lr_fft[:az_blksz] = fft.fft(
                        mmap_blk_lr[:az_blksz, :rg_blksz],
                        n=nfft, axis=1, workers=self.num_cpu_fft
                    )
                    # get cross correlation in freq domain
                    # LR *= conj(RL)
                    mmap_blk_lr_fft[:az_blksz] *= (
                        mmap_blk_rl_fft[:az_blksz].conj())

                    # get FR angle varation in fast-time frequency by
                    # averaging over azimuth, take fftshift in range, and
                    # limit fast-time frequency within around [-BW/2, BW/2]
                    z_fft = fft.fftshift(
                        np.nanmean(mmap_blk_lr_fft[:az_blksz], axis=0))
                    fr_ang_fft = cst_fra_fft * np.rad2deg(np.unwrap(
                        np.angle(z_fft[slice_freq_bin])))
                    # perform linear regression and get the slope (deg/MHz)
                    # Use degrees in place of radians and MHz (or GHz) in place
                    # of Hz to avoid possible singularity in polyfit for very
                    # small FR angles!
                    pf_coefs = np.polyfit(freq_mhz, fr_ang_fft, deg=1)
                    # convert from (deg/MHz) to (rad/Hz) for final product
                    slope[n_az, n_rg] = self._degmhz2radhz * pf_coefs[0]
                    self.logger.info(
                        'Estimated slope of FR in frequency domain (deg/MHz)'
                        f' -> {pf_coefs[0]:.7f}')
                    # get FR at the center frequency in radians
                    fra_mean = self._slope2ang * pf_coefs[0]
                    fr_ang[n_az, n_rg] = np.deg2rad(fra_mean)

                    # plot estimated and polyfitted FR angles as a function
                    # of RF frequency per block if requested
                    if self._plot:
                        name_plot = os.path.join(
                            self._plotdir,
                            'faraday_angle_fftslope_bickel_bates_'
                            f'az{n_az+1}_rg{n_rg+1}.png'
                        )
                        _plot_fra_fft(
                            fra_mean, fr_ang_fft, freq_mhz, pf_coefs,
                            name_plot, 'Bickel-Bates'
                        )

                else:  # time domain
                    # use phase of the time-domain correlator
                    fr_ang[n_az, n_rg] = 0.25 * np.angle(cmp_est)

        # free memory
        del mmap_blk_co, mmap_blk_cx, mmap_blk_lr, mmap_blk_rl
        if self.use_slope_freq:
            del mmap_blk_rl_fft, mmap_blk_lr_fft

        # form slant range vector over all range blocks in (m)
        sr_all = (self._rdr_grid.starting_range +
                  rgb_mid_all * self._rdr_grid.range_pixel_spacing)

        # form azimuth date-time in UTC for all blocks
        az_dt_all = [self._rdr_grid.sensing_datetime(azb)
                     for azb in azb_mid_all]

        # build FR angle product
        fra_prod = FaradayAngleProductSlc(
            fr_ang, mag_fr, sr_all, az_dt_all, slope_freq=slope
        )
        # 2-D plots of faraday rotation angle and its magnitude of estimator
        if self._plot:
            plot_name = os.path.join(
                self._plotdir, 'faraday_angle_magnitude_bickel_bates.png'
            )
            _plot2d(fra_prod, 90.0, plot_name, 'Bickel-Bates')

        return fra_prod


class FaradayRotEstFreemanSecond(FaradayRotAngleSlc):
    """
    Faraday Rotation Angle estimator based on second-order estimator
    proposed by [Freeman2004]_ fom radiometrically and polarimetrically
    calibrated quad-pol RSLC product.

    Parameters
    ----------
    slc : nisar.products.readers.SLC
        Radiometrically calibrated linear quad-pol RSLC product.
    freq_band : {'A', 'B'}, default='A'
        Frequency band used for RSLC product.
    dir_tmp : str, default='.'
        A path for temporary directory containing large intermediate
        memmap binary files as well as PNG plots if any.
    logger : logging.Logger, optional
        If not provided a logger with StreamHandler will be set.
    min_sr_spacing : float, default=400.0
        Min slant range spacing between range blocks in (m).
        The default corresponds to around 50 mdeg in EL for spaceborne.
    min_azt_spacing : float, default=0.25
        Min azimuth time spacing between azimuth blocks in (sec).
        The default corresponds to around 500 pulses for most spaceborne
        cases w/ PRF around 2KHz.
    plot : bool, default=False
        Generates PNG plot stored under `dir_tmp` for final Faraday rotation
        angles and the estimator magnitudes in radar grid.
        Note that if package `matplotlib` does not exist,
        no plots will be generated and a warning will be issued!

    References
    ----------
    .. [Freeman2004] A. Freeman, 'Calibration of linearly polarized
        polarimetric SAR data subject to Faraday rotation,' IEEE Trans.
        Geosci. Remote Sens., Vol 42, pp.1617-1624, August, 2004.

    """

    def estimate(self, azt_blk_size=5.0, sr_blk_size=3000.0,
                 azt_lim=(None, None), sr_lim=(None, None)):
        """
        Estimate Faraday Rotation Angle based on second-order estimator
        proposed by [Freeman2004]_ fom radiometrically and polarimetrically
        calibrated quad-pol RSLC. This method is called "FS".

        Parameters
        ----------
        azt_blk_size : float, default=5.0
            Max azimuth block size in (sec).
        sr_blk_size : float, default=3000.0
            Max slant range block size in (m).
        azt_lim : tuple[float/isce3.core.DateTime, float/isce3.core.DateTime],
                  optional
            Azimuth time/datetime limit of RSLC extended scene data used for
            estimation. The limit is defined as [first, last] in either
            relative times (sec) or in DateTimes (UTC).
        sr_lim : tuple[float, float], optional
            Slant range limit of the RSLC extended scene data used for
            estimation. Span is defined as [first, last] of slant range (m).

        Returns
        -------
        FaradayAngleProductSlc
            Faraday rotation angles along with magnitudes of FR estimator
            which can be used as so-called quality measure of the estimates
            over all blocks.

        Raises
        ------
        OutOfSlcBoundError
            If azimuth time and/or slant range limits are out of SLC boundary.
        ValueError
            Unordered values for range and/or azimuth limits.
            Too small block sizes in either range or azimuth.

        Notes
        -----
        If any of values in "azt_lim" and "rg_lim" is None, it will be set
        to its respective start/stop limit of RSLC dataset.

        The original estimator has unambiguous period of pi/2, that is the FR
        angles can be unambiguously estimated within [-pi/4, pi/4]. However,
        this second-order version can simply estimate magnitude of FR by
        itself.

        Note that using `arctan2` in place of `arctan` will not improve
        unambiguous extent given both terms in ratio `y/x` are positive!

        To determine the sign for its `arctan`, the real-part of the
        multi-looked first-order equation in [Freeman2004]_ is taken into
        account.

        Theoretically, the imaginary part shall be zero given the
        polarimetric products are well balanced, cross-talk level is
        either removed or relatively low (below -40 dB), and
        signal-to-noise ratio (SNR) is good or equivalently low
        noise equivalent sigma zero (NESZ < -30 dB).

        References
        ----------
        .. [Freeman2004] A. Freeman, 'Calibration of linearly polarized
            polarimetric SAR data subject to Faraday rotation,' IEEE Trans.
            Geosci. Remote Sens., Vol 42, pp.1617-1624, August, 2004.

        """
        # get block info in az/range
        blk_az, blk_rg = self._get_block_info(
            azt_lim, sr_lim, azt_blk_size, sr_blk_size)

        # get decoded RSLC dataset for all pols
        dset_hv = self._slc.getSlcDatasetAsNativeComplex(self.freq_band, 'HV')
        dset_vh = self._slc.getSlcDatasetAsNativeComplex(self.freq_band, 'VH')
        dset_hh = self._slc.getSlcDatasetAsNativeComplex(self.freq_band, 'HH')
        dset_vv = self._slc.getSlcDatasetAsNativeComplex(self.freq_band, 'VV')

        # create tmp files for memory mapping of a block under tmp dir
        fid_co = NamedTemporaryFile(suffix='_rslc_co.c8',
                                    dir=self.tmp_dir_name,
                                    delete=False)
        fid_cx = NamedTemporaryFile(suffix='_rslc_cx.c8',
                                    dir=self.tmp_dir_name,
                                    delete=False)
        fid_mag = NamedTemporaryFile(suffix='_rslc_mag.f4',
                                     dir=self.tmp_dir_name,
                                     delete=False)

        # memory maps for block analysis using full block size
        block_shape = (blk_az.fun_size(0), blk_rg.fun_size(0))

        mmap_blk_co = np.memmap(fid_co, mode='w+', dtype=np.complex64,
                                shape=block_shape)
        mmap_blk_cx = np.memmap(fid_cx, mode='w+', dtype=np.complex64,
                                shape=block_shape)
        mmap_blk_mag = np.memmap(fid_mag, mode='w+', dtype=np.float32,
                                 shape=block_shape)

        # shape of all output containers
        out_shape = (blk_az.num_blks, blk_rg.num_blks)

        # container for all range bin slices
        rgb_slice_all = tuple(blk_rg.gen_slice)

        # initialize output containers
        fr_ang = np.ones(out_shape, dtype='f4')
        mag_fr = np.ones(out_shape, dtype='f4')

        # mid azimuth/range bins (float values) for all blocks
        azb_mid_all = np.zeros(blk_az.num_blks, dtype='f4')
        rgb_mid_all = np.zeros(blk_rg.num_blks, dtype='f4')

        # loop over all azimuth (AZ) blocks
        for n_az, azb_slice in enumerate(blk_az.gen_slice):
            # get az block size
            az_blksz = blk_az.fun_size(n_az)
            # get mid azimuth bin for the block
            azb_mid_all[n_az] = 0.5 * (azb_slice.stop + azb_slice.start - 1)

            # loop over all range (RG) blocks
            for n_rg, rgb_slice in enumerate(rgb_slice_all):
                self.logger.info('Processing block (AZ, RG) -> '
                                 f'({azb_slice}, {rgb_slice})')
                # get range block size
                rg_blksz = blk_rg.fun_size(n_rg)

                # get mid range bin for the block
                if n_az == 0:
                    rgb_mid_all[n_rg] = 0.5 * (rgb_slice.stop +
                                               rgb_slice.start - 1)

                # block processing steps
                # diff of X-pols <VH - HV>
                mmap_blk_cx[:az_blksz, :rg_blksz] = (
                    dset_vh[azb_slice, rgb_slice] -
                    dset_hv[azb_slice, rgb_slice]
                )
                # get power of X pol diff
                mmap_blk_mag[:az_blksz, :rg_blksz] = abs(
                    mmap_blk_cx[:az_blksz, :rg_blksz]
                ) ** 2
                # averaged power of x-pol diff
                pow_avg_cx = np.nanmean(
                    mmap_blk_mag[:az_blksz, :rg_blksz]
                )
                # sum of co-pols <HH + VV>
                mmap_blk_co[:az_blksz, :rg_blksz] = (
                    dset_hh[azb_slice, rgb_slice] +
                    dset_vv[azb_slice, rgb_slice]
                )
                # get power of co-pol sum
                mmap_blk_mag[:az_blksz, :rg_blksz] = abs(
                    mmap_blk_co[:az_blksz, :rg_blksz]
                ) ** 2
                # averaged power of co-pol sum
                pow_avg_co = np.nanmean(
                    mmap_blk_mag[:az_blksz, :rg_blksz]
                )
                # calculate magnitude of estimator based on
                # sqrt(|HH + VV|^2 + |VH -HV|^2)
                mag_fr[n_az, n_rg] = np.sqrt(pow_avg_cx + pow_avg_co)
                self.logger.info('Absolute magnitude of FR estimator (linear)'
                                 f' -> {mag_fr[n_az, n_rg]:.3f}')

                # determine the sign of FR
                # use only the real part given imag part shall be
                # too small or theoretically zero!
                fra_sign = np.sign(np.nanmean(
                    mmap_blk_cx[:az_blksz, :rg_blksz] /
                    mmap_blk_co[:az_blksz, :rg_blksz]
                ).real)
                self.logger.info(f'Sign of FR angle -> {fra_sign}')

                # get FR angle (rad) based on
                # 0.5 * arctan(sqrt(|VH - HV|^2 / |HH + VV|^2))
                # and apply the correct sign
                fr_ang[n_az, n_rg] = 0.5 * fra_sign * np.arctan(
                    np.sqrt(pow_avg_cx / pow_avg_co)
                )

        # free memory
        del mmap_blk_co, mmap_blk_cx, mmap_blk_mag

        # form slant range vector over all range blocks in (m)
        sr_all = (self._rdr_grid.starting_range +
                  rgb_mid_all * self._rdr_grid.range_pixel_spacing)

        # form azimuth date time in UTC for all blocks
        az_dt_all = [self._rdr_grid.sensing_datetime(azb)
                     for azb in azb_mid_all]

        # build FR angle product from RSLC
        fra_prod = FaradayAngleProductSlc(fr_ang, mag_fr, sr_all, az_dt_all)

        # 2-D plots of faraday rotation angle and its magnitude of estimator
        if self._plot:
            plot_name = os.path.join(
                self._plotdir, 'faraday_angle_magnitude_freeman_second.png'
            )
            _plot2d(fra_prod, 90.0, plot_name, 'Freeman-Second')

        return fra_prod


# helper functions


def _plot_fra_fft(fra_mean: float, fr_ang_fft: np.ndarray,
                  freq_mhz: np.ndarray, pf_coefs: list, name_plot: str,
                  method: str):
    """
    Plot FR angles as a function of RF frequency (MHz) v.s. its polyfitted
    values.
    """
    pf_vals = np.polyval(pf_coefs, freq_mhz) + fra_mean
    plt.figure(figsize=(7, 6))
    plt.plot(freq_mhz, fr_ang_fft + fra_mean, 'b--',
             freq_mhz, pf_vals, 'r-.', linewidth=2)
    plt.legend(['Estimated', 'Polyfit'], loc='best')
    plt.title('Estimated FR angle v.s. its Linear Fit in Frequency'
              f' via {method}')
    plt.xlabel('RF Frequency (MHz)')
    plt.ylabel('Faraday Angle (deg)')
    plt.grid(True)
    plt.savefig(name_plot)
    plt.close()


def _plot2d(fr_prod: FaradayAngleProductSlc, period_deg: float,
            plot_name: str, tag: str):
    """2-D scatter plots of Faraday rotation angle and its magnitude"""
    # use the first az date-time as reference epoch and get relative az time
    # in seconds
    ref_epoch = fr_prod.az_datetime[0].isoformat()
    azt_rel = [(dt - fr_prod.az_datetime[0]).total_seconds()
               for dt in fr_prod.az_datetime]

    # form 2-D mesh grid az relative time (sec) by slant range (km)
    sr_mesh, azt_mesh = np.meshgrid(fr_prod.slant_range * 1e-3, azt_rel)
    fra_deg = np.rad2deg(fr_prod.faraday_ang)
    max_abs_deg = 0.5 * period_deg

    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.scatter(sr_mesh, azt_mesh, c=fra_deg, marker='o', cmap='hsv',
                vmin=-max_abs_deg, vmax=max_abs_deg)
    plt.colorbar(label='Faraday Rotation Angle (deg)')
    plt.xlabel('Slant Ranges (km)')
    plt.ylabel('Azimuth Relative Time (sec)')
    plt.grid(True)
    plt.title(f'"{tag}" Faraday Rotation Angle & its Estimator Magnitude\n'
              f'w.r.t. Ref Epoch "{ref_epoch}"')
    plt.subplot(212)
    plt.scatter(sr_mesh, azt_mesh, c=fr_prod.magnitude,
                marker='*', cmap='jet')
    cbar = plt.colorbar(label='Magnitude of Estimator (linear)')
    cbar.ax.tick_params(rotation=45)
    plt.xlabel('Slant Ranges (km)')
    plt.ylabel('Azimuth Relative Time (sec)')
    plt.grid(True)
    plt.savefig(plot_name)
