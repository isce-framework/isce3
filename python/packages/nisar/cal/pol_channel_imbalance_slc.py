"""
It contains functions/classes for Polarimetric Channel Imbalance Estimation
"""
from __future__ import annotations
import os
from tempfile import TemporaryDirectory, NamedTemporaryFile
import numpy as np
from dataclasses import dataclass
import typing
from scipy.interpolate import interp1d

from isce3.antenna import CrossTalk, PolImbalanceRatioAnt, rdr2ant
from nisar.cal import CRInfoSlc, est_peak_loc_cr_from_slc
from isce3.core import Ellipsoid, DateTime
from isce3.geometry import DEMInterpolator, rdr2geo
from nisar.log import set_logger


class OutOfSlcBoundError(Exception):
    """Azimuth time or slant range values are out of SLC data boundary"""
    pass


@dataclass(frozen=True)
class PolImbalanceProductSlc:
    """
    Polarimetric Channel Imbalance Products extracted from RSLC in RSLC grid.
    Pol imbalance is defined as a complex ratio of "V co-pol" to "H co-pol".

    Attributes
    ----------
    tx_pol_ratio : 2-D array of complex
        with shape (azimuth blocks, range blocks)
    rx_pol_ratio : 2-D array of complex
        with shape (azimuth blocks, range blocks)
    slant_range : 1-D array of float
        Slant ranges in (m)
    az_datetime : 1-D array of isce3.core.DateTime
        Azimuth (AZ) datetime in (UTC)

    """
    tx_pol_ratio: np.ndarray
    rx_pol_ratio: np.ndarray
    slant_range:  np.ndarray
    az_datetime: np.ndarray


@dataclass(frozen=True)
class BlockInfo:
    """Define info of blocks of bins in AZ or range direction

    Attributes
    ----------
    gen_slice : typing.Iterator[slice]
        Generator slice object for each block.
    fun_size : typing.Callable[[int], int]
        Function to return block size per block number starting from 0.
        The valid integer values are within range(num_blks).
    num_blks : int
        Total number of blocks, full and partial.

    """
    gen_slice: typing.Iterator[slice]
    fun_size: typing.Callable[[int], int]
    num_blks: int


class PolChannelImbalanceSlc:
    """
    Polarimetric Channel Imabalance Class based on RSLC products.
    The algorithm to extract polarimetric channel imbalance from RSLC
    products is mainly based on the algorithm described in [FREEMAN2004]_.

    Parameters
    ----------
    slc_ext : nisar.products.readers.SLC
        Radiometrically calibrated quad-pol [linear] RSLC product over
        homogenous extended scene like Amazon rain forest.
    slc_cr : nisar.products.readers.SLC
        Radiometrically calibrated quad-pol [linear] RSLC product over corner
        reflector (CR).
    cr_llh : 1-D/2-D sequence or array of three floats
        Corner reflectors' approximate geodetic longitude, latitude, and height
        in (rad, rad, m) exist in RSLC product "slc_cr". 2-D shape is (N by 3).
    cross_talk : isce3.antenna.CrossTalk, optional
        Antenna cross talk values, 1-D LUT as a function of Elevation (EL)
        angles (rad). If not provided, cross talk will be ignored!
    freq_band : {'A', 'B'}, default='A'
        Frequency band used for both RSLC products.
    dem : isce3.core.DEMInterpolator, optional
        DEM heights, either a fixed height wrt reference ellipsoid or DEM
        raster w/ topography. If not provided, it will be assumed to be zero
        height w.r.t. WGS84 ellipsoid.
    ignore_faraday : bool, default=False
        If true, it is assumed Faraday rotation is too small to have
        any effect.
    pixel_slc_margin : int, default=8
        Number of pixels used as a margin in both range and azimuth to decide
        whether a CR is wihtin its SLC bound or not. This value will be used as
        a half of chip size around corner reflector in its RSLC product.
        It must be equal or larger than 2!
    interp_method : {'linear', 'nearest', 'quadratic', 'cubic'},
                    default='linear'.
        The kind of 1-D SciPy interpolation function. This is used to
        interpolate (and extrapolate) azimuth-averaged channel imbalances
        as a function of elevation angles in cross track.
    dir_tmp : str, default='.'
        A path for temporary directory containing large intermediate
        memmap binary files.
    logger : logging.Logger, optional
        If not provided a logger with StreamHandler will be set.
    min_sr_spacing : float, default=200.0
        Min slant range spacing between range blocks in (m).
        The default corresponds to around 25 mdeg in EL for spaceborne.
    min_frac_sar : float, default=0.1
        Min azimuth duration per azimuth block expressed approximately
        as fraction of a SAR aperture duration.
    sr_threshold : float, default=1.0
        Slant range threshold/step used in `geo2rdr`/`rdr2geo` in (m)
    max_iter : int, default=50
        Max number of iterations used in `geo2rdr` and `rdr2geo`
    rel_pow_th_db : float, default=3.0
        Relative power threshold used for CR extracted peak values in
        `est_peak_loc_cr_from_slc`.

    Attributes
    ----------
    cr_info_slc : list of isce3.cal.CRInfoSlc
        Corner reflector info from SLC
    vv2hh_cr : a complex scalar
        VV/HH ratio computed from all CRs polarimetric data
    tmp_dir_name : str
        Name of the temporary directory
    skip_xtalk_removal : bool
        Whether or not skip cross-talk removal

    Raises
    ------
    ValueError
        If pixel_slc_margin < 2.
        Wrong interpolation method.
        Either of RSLC products is not linear quad-pol
        Wrong/unavailable frequency band
    RuntimeError
        No CR data extracted from its RSLC

    Warnings
    --------
    OutOfSlcBoundWarning
        For CR out of RSLC bound within margin "pixel_slc_margin".

    References
    ----------
    .. [FREEMAN2004] A. Freeman, 'Calibration of Linearly Polarized
        Polarimetric SAR Data Subject to Faraday Rotation',
        IEEE Transaction on GeoSci and Remote Sensing, Vol. 42,
        August 2004.

    """

    def __init__(self, slc_ext, slc_cr, cr_llh, *, cross_talk=None,
                 freq_band='A', dem=None, ignore_faraday=False,
                 pixel_slc_margin=8, interp_method='linear', dir_tmp='.',
                 logger=None, min_sr_spacing=200, min_frac_sar=0.1,
                 sr_threshold=1.0, max_iter=50, rel_pow_th_db=3.0):
        # Parameters
        self._min_sr_spacing = min_sr_spacing
        self._min_frac_sar = min_frac_sar
        self._sr_threshold = sr_threshold
        self._max_iter = max_iter
        self._rel_pow_th_db = rel_pow_th_db
        # check freq_band and quad-pol condition
        _check_quadpol_band_slc(slc_ext, freq_band)
        self._slc_ext = slc_ext
        _check_quadpol_band_slc(slc_cr, freq_band)
        self._slc_cr = slc_cr
        self._cr_llh = cr_llh
        self._freq_band = freq_band
        self._ignore_faraday = ignore_faraday
        # check DEM
        if dem is None:
            self._dem = DEMInterpolator()
        else:
            self._dem = dem
        # check interp method
        if interp_method not in ('linear', 'nearest', 'quadratic', 'cubic'):
            raise ValueError('Wrong interpolation method!')
        self.interp_method = interp_method

        # check logger
        if logger is None:
            self._logger = set_logger("PolChannelImbalance")
        else:
            self._logger = logger

        # Public Read-only Attributes
        if pixel_slc_margin < 2:
            raise ValueError('"pixel_slc_margin" is less than 2!')
        self._chip_size = 2 * pixel_slc_margin

        # get/set cross talk values
        self._cross_talk = cross_talk
        if cross_talk is None:
            self._skip_xtalk_removal = True
            self._logger.warning('No cross-talk data is provided! '
                                 'Cross-talk removal will be skipped!')
        else:
            self._logger.info('H-pol TX cross-talk @ EL=0 -> '
                              f'{self._cross_talk.tx_xpol_h(0):.4f}')
            self._logger.info('V-pol TX cross-talk @ EL=0 -> '
                              f'{self._cross_talk.tx_xpol_v(0):.4f}')
            self._logger.info('H-pol RX cross-talk @ EL=0 -> '
                              f'{self._cross_talk.rx_xpol_h(0):.4f}')
            self._logger.info('V-pol RX cross-talk @ EL=0 -> '
                              f'{self._cross_talk.rx_xpol_v(0):.4f}')
            # check the cross talk values to see if they are all zeros.
            self._skip_xtalk_removal = (
                np.allclose(np.abs(self._cross_talk.tx_xpol_h.y), 0.0) and
                np.allclose(np.abs(self._cross_talk.tx_xpol_v.y), 0.0) and
                np.allclose(np.abs(self._cross_talk.rx_xpol_h.y), 0.0) and
                np.allclose(np.abs(self._cross_talk.rx_xpol_v.y), 0.0)
            )
            if self._skip_xtalk_removal:
                self._logger.info(
                    'All cross-talk values are zero so skipping application.')
        # get list of CR info from its SLC
        self._cr_info_slc = est_peak_loc_cr_from_slc(
            slc_cr, cr_llh, num_pixels=self._chip_size,
            rel_pow_th_db=self._rel_pow_th_db,
            rg_tol=self._sr_threshold, max_iter=self._max_iter)
        if len(self._cr_info_slc) == 0:
            raise RuntimeError('No polarimetric CR data available for getting'
                               ' individual channel imbalance.')
        # get mean of ratios VV/HH over all CRs or use weighted mean of ratio
        # based on total HH+VV power.
        cr_vv2hh = []
        for cr in self._cr_info_slc:
            vv2hh = cr.amp_pol['VV'] / cr.amp_pol['HH']
            el_deg = np.rad2deg(cr.el_ant)
            az_deg = np.rad2deg(cr.az_ant)
            self._logger.info(f'CR at (el, az) = ({el_deg:.2f}, {az_deg:.2f})'
                              f' (deg, deg) has VV/HH -> {vv2hh:.4f}')
            cr_vv2hh.append(vv2hh)
        self._cr_vv2hh = np.mean(cr_vv2hh)
        self._logger.info('Mean VV/HH values among all CRs -> '
                          f'{self._cr_vv2hh:.4f}')
        # open tmp dir
        self._tmpdir = TemporaryDirectory(suffix='_pol_channel_imb',
                                          dir=dir_tmp)

        # Private Attributes
        self._rdr_grid = slc_ext.getRadarGrid(freq_band)
        self._pbw = slc_ext.getSwathMetadata(
            freq_band).processed_azimuth_bandwidth
        self._orbit = slc_ext.getOrbit()
        self._attitude = slc_ext.getAttitude()

    def __enter__(self):
        return self

    def __exit__(self, val_, type_, tb_):
        self._tmpdir.cleanup()

    def __repr__(self):
        slc_ext_name = os.path.basename(self._slc_ext.filename)
        slc_cr_name = os.path.basename(self._slc_cr.filename)
        return (f'{self.__class__.__name__}('
                f'slc_ext={slc_ext_name}, '
                f'slc_cr={slc_cr_name}, '
                f'freq_band={self.freq_band}, '
                f'dem_ref_height={self.dem.ref_height}, '
                f'num_cr={len(self._cr_llh)}, '
                f'skip_xtalk_removal={self.skip_xtalk_removal}, '
                f'ignore_faraday={self.ignore_faraday}, '
                f'chip_size_cr={self.chip_size_cr}, '
                f'interp_method={self.interp_method}, '
                f'dir_name_tmp={self.tmp_dir_name}, '
                f'logger_name={self.logger.name})'
                )

    @property
    def freq_band(self) -> str:
        return self._freq_band

    @property
    def dem(self) -> DEMInterpolator:
        return self._dem

    @property
    def logger(self):
        return self._logger

    @property
    def tmp_dir_name(self) -> str:
        return os.path.abspath(self._tmpdir.name)

    @property
    def chip_size_cr(self) -> int:
        """Chip size used in CR analysis"""
        return self._chip_size

    @property
    def cross_talk(self) -> CrossTalk:
        return self._cross_talk

    @property
    def skip_xtalk_removal(self) -> bool:
        return self._skip_xtalk_removal

    @property
    def ignore_faraday(self) -> bool:
        return self._ignore_faraday

    @property
    def cr_info_slc(self) -> list[CRInfoSlc]:
        """List of Corner reflector info extracted from its RSLC product"""
        return self._cr_info_slc

    @property
    def vv2hh_cr(self) -> complex:
        """
        A complex VV/HH ratio over all CRs prior to channel
        imbalance correction.
        """
        return self._cr_vv2hh

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
        # get half of min SAR length as a lower limit for azimuth block size
        # at around mid azimuth time and mid slant range
        azt_mid = (self._rdr_grid.sensing_start + np.mean(azb_limit) *
                   self._rdr_grid.az_time_interval)
        sr_mid = self._rdr_grid.slant_range(rgb_limit[0]) + 0.5 * tot_sr_dist
        sar_dur = calculate_sar_dur(
            azt_mid, sr_mid, self._rdr_grid.wavelength, self._pbw,
            self._orbit
        )
        self.logger.info(f'Estimated SAR duration -> {sar_dur:.3f} (sec)')
        frac_sar_dur = self._min_frac_sar * sar_dur
        if azt_blk_size < frac_sar_dur:
            raise ValueError('Azimuth time block size is less than fraction '
                             'of SAR duration {frac_sar_dur} (sec)!')

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

    def _build_avg_dem_no_topo(self, azb_slice, rgb_slice):
        """
        Build approximately locally averaged DEM per az/range block.

        The new topography-free DEM object can be safely used for homomorphic
        transformation between time/range (radar grid) and angle
        (antenna grid).

        Parameters
        ----------
        slice
            azimuth bin slice [start, stop)
        slice
            range bin slice [start, stop)

        Returns
        -------
        isce3.core.DEMInterpolator

        Notes
        -----
        Get mean height among Max 9 LLH values calculated from radar grid
        at (start, mid, end) azimuth and range block.
        If input/original DEM has no raster (no topography), its copy will
        be returned as output w/o any computation.

        """
        if not self.dem.have_raster:
            return self.dem
        # There is a DEM raster data
        # check if there is more than one bin in the AZ slice
        if (azb_slice.stop - azb_slice.start) > 1:
            azt_vec = (self._rdr_grid.sensing_start +
                       self._rdr_grid.az_time_interval *
                       np.asarray([azb_slice.start,
                                   0.5 * (azb_slice.start +
                                          azb_slice.stop),
                                   azb_slice.stop - 1])
                       )
        else:  # single-bin slice
            azt_vec = [self._rdr_grid.sensing_start +
                       self._rdr_grid.az_time_interval * azb_slice.start]

        # check if there is more than one bin in the RG slice
        if (rgb_slice.stop - rgb_slice.start) > 1:
            sr_vec = (self._rdr_grid.starting_range +
                      self._rdr_grid.range_pixel_spacing *
                      np.asarray([rgb_slice.start,
                                  0.5 * (rgb_slice.start + rgb_slice.stop),
                                  rgb_slice.stop - 1])
                      )
        else:  # single-bin slice
            sr_vec = [self._rdr_grid.starting_range +
                      self._rdr_grid.range_pixel_spacing * rgb_slice.start]

        # get averaged height at (start, mid, end) azimuth and range block
        hgt = []
        for azt in azt_vec:
            for sr in sr_vec:
                llh = rdr2geo(
                    azt, sr, self._orbit, self._rdr_grid.lookside, 0.0,
                    self._rdr_grid.wavelength, self.dem,
                    threshold=self._sr_threshold, maxiter=self._max_iter
                )
                hgt.append(llh[2])

        return DEMInterpolator(np.mean(hgt))

    def estimate(self, azt_blk_size=5.0, sr_blk_size=3000.0,
                 azt_lim=(None, None), sr_lim=(None, None)):
        """
        Estimate polarimetric channel imbalance complex ratios.

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
            relative times (sec) or in DateTimes (UTC ISO8601).
        sr_lim : tuple[float, float], optional
            Slant range limit of the RSLC extended scene data used for
            estimation. Span is defined as [first, last] of slant range (m).

        Returns
        -------
        PolImbalanceRatioAnt
            TX/RX channel imbalance ratio as a function of EL angles in (rad).
            These values are averaged one over several azimuth blocks if any.
            This object can be used to extrapolate TX/RX channel imabalance for
            out-of-bound EL values.
        PolImbalanceProductSlc
            Channel imbalance of TX and RX over all blocks.

        Raises
        ------
        OutOfSlcBoundError
            If azimuth time and/or slant range limits are out of SLC boundary
            for extended homogenous scene.
        ValueError
            Unordered values for range and/or azimuth limits
            AZ block size is smaller than `min_frac_sar` of SAR aperture.

        Notes
        -----
        If any of values in "azt_lim" and "rg_lim" is None, it will be set
        to its respective start/stop limit of RSLC dataset.

        """
        # get block info in az/range
        blk_az, blk_rg = self._get_block_info(
            azt_lim, sr_lim, azt_blk_size, sr_blk_size)

        # check total number of range blocks against the 1-D interpolation
        # order and reset the method to max lower order if necessary
        if self.interp_method != 'nearest':
            interp_kind = _check_reset_interp_kind(blk_rg.num_blks,
                                                   self.interp_method)
            if self.interp_method != interp_kind:
                self.interp_method = interp_kind
                self.logger.warning(
                    f'Interpolation method is reset to {interp_kind} due to '
                    'number of range blocks'
                )

        # create tmp files for memory mapping of a block under tmp dir
        fid_hv = NamedTemporaryFile(suffix='_rslc_polhv.c8',
                                    dir=self.tmp_dir_name)
        fid_vh = NamedTemporaryFile(suffix='_rslc_polvh.c8',
                                    dir=self.tmp_dir_name)

        # memory maps for block analysis using full block size
        block_shape = (blk_az.fun_size(0), blk_rg.fun_size(0))
        mmap_blk_hv = np.memmap(fid_hv, mode='w+', dtype=np.complex64,
                                shape=block_shape)
        mmap_blk_vh = np.memmap(fid_vh, mode='w+', dtype=np.complex64,
                                shape=block_shape)

        # need an extra mmap for cross correlation used in resolving
        # the pi ambiguity of the phase of channel imbalance ratio RX/TX
        if not self.ignore_faraday:
            fid_xcor = NamedTemporaryFile(suffix='_rslc_xcor.c8',
                                          dir=self.tmp_dir_name)
            mmap_blk_xcor = np.memmap(fid_xcor, mode='w+', dtype=np.complex64,
                                      shape=block_shape)

        # get decoded RSLC dataset for all pols
        dset_hv = self._slc_ext.getSlcDatasetAsNativeComplex(
            self.freq_band, 'HV'
        )
        dset_vh = self._slc_ext.getSlcDatasetAsNativeComplex(
            self.freq_band, 'VH'
        )
        dset_hh = self._slc_ext.getSlcDatasetAsNativeComplex(
            self.freq_band, 'HH'
        )
        dset_vv = self._slc_ext.getSlcDatasetAsNativeComplex(
            self.freq_band, 'VV'
        )

        # container for all range bin slices
        rgb_slice_all = tuple(blk_rg.gen_slice)

        # initialize output containers
        out_shape = (blk_az.num_blks, blk_rg.num_blks)
        imb_rx2tx = np.ones(out_shape, dtype='c8')

        # mid azimuth/range bins (float values) for all blocks
        azb_mid_all = np.zeros(blk_az.num_blks, dtype='f4')
        rgb_mid_all = np.zeros(blk_rg.num_blks, dtype='f4')
        sr_all = np.zeros(blk_rg.num_blks, dtype='f8')

        # loop over all Azimuth (AZ) blocks
        for n_az, azb_slice in enumerate(blk_az.gen_slice):
            # get az block size
            az_blksz = blk_az.fun_size(n_az)
            # get mid azimuth bin for the block
            azb_mid_all[n_az] = 0.5 * (azb_slice.stop + azb_slice.start - 1)

            # loop over all Range (RG) blocks
            for n_rg, rgb_slice in enumerate(rgb_slice_all):
                self.logger.info('Processing block (AZ, RG) -> '
                                 f'({azb_slice}, {rgb_slice})')
                # get range block size
                rg_blksz = blk_rg.fun_size(n_rg)

                # get mid range bin and slant range for the block
                if n_az == 0:
                    rgb_mid_all[n_rg] = 0.5 * (rgb_slice.stop +
                                               rgb_slice.start - 1)

                    # store mid slant range vector over all range blocks in (m)
                    sr_all[n_rg] = self._rdr_grid.slant_range(
                        rgb_mid_all[n_rg])

                # block processing steps
                mmap_blk_hv[:az_blksz, :rg_blksz] = dset_hv[
                    azb_slice, rgb_slice]
                mmap_blk_vh[:az_blksz, :rg_blksz] = dset_vh[
                    azb_slice, rgb_slice]
                # cross talk removal of x-pol products, ignoring squared terms
                if not self.skip_xtalk_removal:

                    # get an approxmiate mean DEM within a block to be used for
                    # slant range to EL conversion w/o topography effect
                    dem_avg_blk = self._build_avg_dem_no_topo(
                        azb_slice, rgb_slice)

                    # get mid azimuth time in (sec) for the block
                    if n_rg == 0:
                        azt_mid_blk = (self._rdr_grid.sensing_start +
                                       self._rdr_grid.az_time_interval *
                                       azb_mid_all[n_az])

                    # get EL angle corresponding to mid range/azimuth of
                    # a block with an approximate-average DEM height
                    el_blk, _ = rdr2ant(
                        azt_mid_blk, sr_all[n_rg], self._orbit,
                        self._attitude, self._rdr_grid.lookside,
                        self._rdr_grid.wavelength, dem=dem_avg_blk,
                        threshold=self._sr_threshold, maxiter=self._max_iter
                    )

                    # remove x-talk for a certain EL angle per mid
                    # range/azimuth of the block
                    mmap_blk_vh[:az_blksz, :rg_blksz] -= (
                        self.cross_talk.tx_xpol_v(el_blk) *
                        dset_hh[azb_slice, rgb_slice] +
                        self.cross_talk.rx_xpol_v(el_blk) *
                        dset_vv[azb_slice, rgb_slice]
                    )
                    mmap_blk_hv[:az_blksz, :rg_blksz] -= (
                        self.cross_talk.rx_xpol_h(el_blk) *
                        dset_hh[azb_slice, rgb_slice] +
                        self.cross_talk.tx_xpol_h(el_blk) *
                        dset_vv[azb_slice, rgb_slice]
                    )
                # check if Faraday rotation is negligible if so
                # directly get the complex imbalance ratio RX/TX
                # from ensemble averaged ratio of only x-pol products.
                # The only assumption is scene backscatter reciprocity.
                if self.ignore_faraday:
                    imb_rx2tx[n_az, n_rg] = np.nanmean(
                        mmap_blk_hv[:az_blksz, :rg_blksz] /
                        mmap_blk_vh[:az_blksz, :rg_blksz]
                    )
                else:  # Faraday rotation can not be ignored!
                    # phase and amp of the ratio will be calculated
                    # seperately w/ +/- pi phase ambiguity by using
                    # x-pol products provided scene reflection/AZ symmetry!
                    phs_imb_rx2tx = np.angle(np.nanmean(
                        mmap_blk_hv[:az_blksz, :rg_blksz] *
                        mmap_blk_vh[:az_blksz, :rg_blksz].conj()
                    ))
                    amp_imb_rx2tx = np.sqrt(abs(
                        np.nanmean(mmap_blk_hv[:az_blksz, :rg_blksz] *
                                   mmap_blk_hv[:az_blksz, :rg_blksz].conj()) /
                        np.nanmean(mmap_blk_vh[:az_blksz, :rg_blksz] *
                                   mmap_blk_vh[:az_blksz, :rg_blksz].conj())
                    ))
                    # form a complex ratio holding amp and phase
                    # w/ ambiguous sign to be resolved
                    imb_rx2tx[n_az, n_rg] = amp_imb_rx2tx * np.exp(
                        1j * phs_imb_rx2tx)
                    # form a x-correlation HH-HV whose magnitude will be
                    # used to resolve the sign of complex ratio.
                    # try with zero and pi phase adjustments of RX/TX.
                    xcor_hh_hv_zero = _xcor_hh_hv(
                        mmap_blk_xcor, mmap_blk_hv, mmap_blk_vh, dset_hh,
                        az_blksz, rg_blksz, azb_slice, rgb_slice,
                        imb_rx2tx[n_az, n_rg]
                    )
                    self.logger.info(
                        'Cross-correlation HH-HV w/o phase correction of '
                        f'imbalance RX/TX -> {xcor_hh_hv_zero:.4f} (linear)'
                    )
                    xcor_hh_hv_pi = _xcor_hh_hv(
                        mmap_blk_xcor, mmap_blk_hv, mmap_blk_vh, dset_hh,
                        az_blksz, rg_blksz, azb_slice, rgb_slice,
                        -imb_rx2tx[n_az, n_rg]
                    )
                    self.logger.info(
                        'Cross-correlation HH-HV w/ phase correction of '
                        f'imbalance RX/TX -> {xcor_hh_hv_pi:.4f} (linear)'
                    )
                    # adding pi to the phase of RX/TX ratio if needed!
                    # Note that whichever phase results in lower x-correlation
                    # is the right phase given approximate azimuth symmetry of
                    # scene! The difference in (dB) between two x-correlations
                    # is pretty large.
                    if xcor_hh_hv_pi < xcor_hh_hv_zero:
                        imb_rx2tx[n_az, n_rg] *= -1

        # free memory
        del mmap_blk_vh, mmap_blk_hv

        # closing the tmp files (fid.close()) , not necessary given it can be
        # done via context manager or garbage collected controlled by users.

        # get individual pol channel imabalance by using CR(s) co-pol info
        # extracted from its respective RSLC
        # RX imbalance = sqrt(cr_vv2hh * imb_rx2tx)
        imb_rx = np.sqrt(self._cr_vv2hh * imb_rx2tx)
        imb_tx = imb_rx / imb_rx2tx

        # form azimuth date time in UTC for all blocks
        az_dt_all = [self._rdr_grid.sensing_datetime(azb)
                     for azb in azb_mid_all]

        # store final 2-D values for all blocks of SLC expressed in radar grid
        imb_prod_slc = PolImbalanceProductSlc(
            imb_tx, imb_rx, sr_all, az_dt_all)

        # mean AZ time (sec) among all azimuth blocks
        azt_mean = (self._rdr_grid.sensing_datetime(azb_mid_all.mean()) -
                    self._rdr_grid.ref_epoch).total_seconds()
        self.logger.info(
            f'Mean AZ time used for LUT1d products -> {azt_mean:.6f} (sec)'
        )
        # convert slant ranges to EL angles in antenna frame by using
        # mean DEM (no topography)
        az_s = slice(int(azb_mid_all[0]), int(azb_mid_all[-1]))
        rg_s = slice(int(rgb_mid_all[0]), int(rgb_mid_all[-1]))
        dem_mean = self._build_avg_dem_no_topo(az_s, rg_s)
        self.logger.info('Local averaged DEM height used for LUT1d products ->'
                         f' {dem_mean.ref_height:.3f} (m)')

        el_ant = np.zeros_like(sr_all)
        for nn, sr in enumerate(sr_all):
            el, az = rdr2ant(
                azt_mean, sr, self._orbit, self._attitude,
                self._rdr_grid.lookside, self._rdr_grid.wavelength,
                dem=dem_mean, threshold=self._sr_threshold,
                maxiter=self._max_iter
            )
            el_ant[nn] = el

        # get mean value along azimuth over all AZ blocks to simply
        # have values as a function of range.
        imb_rx_med = np.mean(imb_rx, axis=0)
        imb_tx_med = np.mean(imb_tx, axis=0)

        # make sure the size of the vectors along range is at least 2
        # if not repeat the value to avoid TypeError in interp1d.
        if blk_rg.num_blks < 2:
            el_ant = el_ant.repeat(2)
            imb_rx_med = imb_rx_med.repeat(2)
            imb_tx_med = imb_tx_med.repeat(2)

        # form 1-D LUT of complex pol imabalance as a function of EL
        # angles (rad) w/ interpolation/extrapolation via interp1d
        rx_imb_lut1d = interp1d(el_ant, imb_rx_med, kind=self.interp_method,
                                fill_value='extrapolate', assume_sorted=True)
        tx_imb_lut1d = interp1d(el_ant, imb_tx_med, kind=self.interp_method,
                                fill_value='extrapolate', assume_sorted=True)
        imb_prod_lut1d = PolImbalanceRatioAnt(tx_imb_lut1d, rx_imb_lut1d)

        return imb_prod_lut1d, imb_prod_slc

# some helper functions


def _check_quadpol_band_slc(slc, freq_band) -> None:
    """
    Check to see if SLC is quad-pol product and the frequency band exists.
    If not raise exception.

    Parameters
    ----------
    slc : nisar.products.readers.SLC
    freq_band : str

    Raises
    ------
    ValueError

    """
    if freq_band not in slc.frequencies:
        raise ValueError(f'Wrong freq band for SLC "{slc.filename}" with'
                         f' frequencies {slc.frequencies}!')
    list_pols = slc.polarizations[freq_band]
    if (len(list_pols) != 4) and ('HH' not in list_pols):
        raise ValueError(f'The SLC "{slc.filename}" is not linear quad-pol!')


def _check_reset_interp_kind(size: int, kind: str) -> str:
    """Check and reset interpolation kind based on size of the data.
    """
    max_order = min(size - 1, 3)
    kind2order = {'nearest': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3}
    order2kind = {val: key for key, val in kind2order.items()}
    return order2kind[min(kind2order[kind], max_order)]


def _get_bin_limits(lim: tuple[float, float], first: float, spacing: float,
                    num_bins: int, err_msg='') -> tuple[int, int]:
    """
    Get [start, stop) bin limits per desired slant range/azimuth time limits
    """
    limit = list(lim)
    # start
    if limit[0] is None:
        b_start = 0
    else:
        b_start = int(np.floor((limit[0] - first) / spacing))
        if b_start < 0 or b_start > (num_bins - 1):
            raise OutOfSlcBoundError(f'Start {err_msg}!')
    # stop
    if limit[1] is None:
        b_stop = num_bins
    else:
        b_stop = int(np.ceil((limit[1] - first) / spacing))
        if b_stop > num_bins:
            raise OutOfSlcBoundError(f'Stop {err_msg}!')
        if b_start >= b_stop:
            raise ValueError(f'Bad unordered {err_msg}!')
    return b_start, b_stop


def _get_block_size_slice(bin_lim: tuple[int, int], n_blks: int,
                          n_bin_blk: int, max_size):
    """
    Get block-size function and block-slice generator per az/range bins limits

    Returns
    -------
    Callable[[int], int]
        Function to return block size per block number
    Iterator[slice]
        Slice generator per block

    """
    b_stop_last = bin_lim[0] + n_blks * n_bin_blk
    blksz_last = min(b_stop_last, max_size) - (b_stop_last - n_bin_blk)
    bn_last = n_blks - 1

    def blksz_fun(n):
        if n < bn_last:
            return n_bin_blk
        return blksz_last

    b_slice = (slice(b_start, b_start + n_bin_blk)
               for b_start in range(*bin_lim, n_bin_blk))

    return blksz_fun, b_slice


def _xcor_hh_hv(mmap_blk_xcor, mmap_blk_hv, mmap_blk_vh, dset_hh,
                az_blksz, rg_blksz, azb_slice, rgb_slice, imb_rx2tx) -> float:
    """
    Calculate normalized cross-correlation between HH and symmetrized version
    of HV in order to resolve for pi ambiguity in phase of channel imbalance
    ratio RX/TX.

    """
    # form a symmetrized cross pol product. The factor 0.5
    # will be ignored.
    mmap_blk_xcor[:az_blksz, :rg_blksz] = (
        mmap_blk_hv[:az_blksz, :rg_blksz] + (
            imb_rx2tx * mmap_blk_vh[:az_blksz, :rg_blksz])
    )
    # power of symmetrized x-pol scaled by 4 due to
    # missing scalar 0.5
    pow_hv_sym = abs(np.nanmean(
        mmap_blk_xcor[:az_blksz, :rg_blksz] *
        mmap_blk_xcor[:az_blksz, :rg_blksz].conj()
    ))
    # no need to calculate power of HH for x-correlation
    # normalization

    # cross correlate symmetrized x-pol by co-pol
    mmap_blk_xcor[:az_blksz, :rg_blksz] *= (
        dset_hh[azb_slice, rgb_slice].conj()
    )
    # store power-normalized magnitude of ensemble average of
    # cross correlation between co-pol and symmetrized x-pol
    # product
    xcor_hh_hv = abs(np.nanmean(
        mmap_blk_xcor[:az_blksz, :rg_blksz])) / pow_hv_sym

    return xcor_hh_hv

# some public functions


def calculate_sar_dur(azt, sr, wl, pbw, orbit):
    """
    Calculate SAR duration approximately in time (sec) at a desired azimuth
    relative time and at a desired slant range given orbit, wavelength and
    azimuth processing bandwidth (PBW).

    Parameters
    ----------
    azt : float
        Azimuth time in (sec) w.r.t ref epoch of orbit
    sr : float
        slant range in (m)
    wl : float
        Wavelength in (m)
    pbw : float
        Azimuth processing bandwidth in (Hz)
    orbit : isce3.core.Orbit

    Returns
    -------
    float
        SAR duration in (sec)

    Notes
    -----
    Ground velocity is simply calculated for the along-track radius of
    curvature of an approximate sphere defined at spacecraft geolocation.

    """
    pos, vel = orbit.interpolate(azt)
    wgs = Ellipsoid()
    llh = wgs.xyz_to_lon_lat(pos)
    re = wgs.r_dir(*llh[:2])
    vel_m = np.linalg.norm(vel)
    pos_m = np.linalg.norm(pos)
    return (sr * wl * pbw * pos_m) / (2 * re * vel_m**2)
