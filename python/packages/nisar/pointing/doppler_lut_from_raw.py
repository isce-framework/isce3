"""
Function to generate Doppler LUT2d from Raw L0B data.
"""
import os
import numpy as np
from scipy import fft
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from isce3.signal import (cheby_equi_ripple_filter, corr_doppler_est,
                          sign_doppler_est, unwrap_doppler)
from isce3.core import LUT2d, speed_of_light
from isce3.antenna import Frame
from isce3.geometry import DEMInterpolator
from isce3.signal import form_single_tap_dbf_echo
from nisar.log import set_logger


def doppler_lut_from_raw(raw, *, freq_band='A', txrx_pol=None,
                         orbit=None, attitude=None, ant=None,
                         dem=None, num_rgb_avg=8, az_block_dur=4.0,
                         time_interval=2.0, dop_method='CDE', subband=False,
                         polyfit_deg=3, polyfit=False, exclude_beams=None,
                         out_path='.', plot=False, logger=None):
    """Generates 2-D Doppler LUT as a function of slant range and azimuth time.

    It generates Doppler map in isce3.core.LUT2d format.
    It optionally generates  Doppler plots as a function of
    slant ranges at various azimuth times stored in PNG files.
    For algorithms
    See references [GHAEMI2018]_, [MADSEN1989]_, [BAMLER1991]_.

    The subbanding is a joint time-frequency approach where three frequency
    bands lower, mid, and upper part of the echo are individually used in
    time-domain doppler correlator estimator and then a linear regression is
    applied to the three doppler values as a function of frequency. Finally,
    the doppler is evaluated at the center frequency of the band from the
    first-degree polyfit coefficients.

    To generate three sub-bands, a FIR Chebyshev Equi-rippler low-pass filter
    is designed. This filter is up/down converted to perform band-pass
    filtering of lower/upper part of the band.

    In case of polyfit, the Doppler at invalid range bins will be replaced by
    poly evaluated ones and thus the final respective valid mask will all set
    to True. That is no invalid range bins will be reported!

    Parameters
    ----------
    raw : nisar.products.readers.Raw.RawBase
        Raw L0B product parser base object
    freq_band : {'A', 'B'}
        Frequency band in multi-band TX chirp.
    txrx_pol : str, optional
        TxRx polarization such as {'HH', 'HV',...}. If not provided the first
        product under `freq_band` will be used.
    orbit : isce3.core.Orbit, optional
        If specified, this will be used in place of the orbit data
        stored in L0B.
    attitude : isce3.core.Attitude, optional
        If specified, this will be used in place of attitude data
        stored in L0B.
    ant : nisar.products.readers.antenna.AntennaParser, optional
        Antenna HDF5 product parser object. If not provided, the elevation
        and azimith angles of unit look vector used in absolute doppler
        and ambiguity computation will be set to zero degrees.
        Note that this object is required for multi-channel (NISAR DM2)
        raw product.
    dem : isce3.geometry.DEMInrerpolator, optional
        If None, the WGS84 ellipsoid will be used.
        It is simply used for multi-channel (NISAR DM2) case where elevation
        angles are converted into slant ranges at beams transition locations.
    num_rgb_avg : int, default=8
        Number of range bins to be averaged in final Doppler values.
    az_block_dur : float, default=4.0
        Azimuth block duration in seconds defining time-domain correlator
        length used in Doppler estimator.
    time_interval : float, default=2.0
        Time stamp interval between azimuth blocks in seconds.
        It should not be larger than "az_block_dur".
    dop_method : {'CDE', 'SDE'}
        Correlator-based time-domain Doppler estimator method, either of
        Correlation Doppler Estimator ('CDE') or Sign-Doppler estimator ('SDE')
        See [MADSEN1989]_. These methods used as a base method
        in subbanding time-frequency approach if requested via `subband`.
        See [BAMLER1991]_ and [GHAEMI2018]_.
    subband : bool, default=False
        Whether or not use sub-banding frequency approach on top of correlator
        one in Doppler estimation.
    polyfit_deg : int, default=3
        Polyfit degree used in Doppler plots for polyfitting of doppler as a
        function of slant ranges and its statistical mean/std variation over
        the swath. If "polyfit" flag set to True, the polyfitted version of
        the estimated dopplers in slant range will be used as the final 2-D
        LUT product!
        The polyfitting will be done over valid range bins per azimuth block!
    polyfit : bool, default=False
        If is True, then polyfitted Doppler product with degree "polyfit_deg"
        will be used in place of estimated one as a function of slant range per
        azimuth block.
    exclude_beams : list of int, optional
        List of excluded beam numbers from the polyfit simply in DM2 case.
        Ignored if the input product is not DM2.
        The beam number is in the range [1, N], where N is the number
        of beams or RX channels. By default, all beams are included
        in the polyfitting of Doppler over entire swath of DM2 products.
    out_path : str, default='.'
        Ouput directory for dumping PNG files, if `plot` is True.
    plot : bool, default=False
        If True, it will generate bunch of .png plots of both True and
        poly-fitted Doppler centroid as a function of slant ranges per azimuth
        block. The polyfit degree used in plotting is 3 if not set by the
        `polyfit_deg`!
    logger : logging.Logger, optional
        If not provided a longger with StreamHandler will be set.

    Notes
    -----
    PRF must be constant. Dithered PRF is not supported.
    The LUT2d product requires at least two blocks in each directions.
    In case of polyfit, the number of valid range bins must be larger than
    (polyfit_deg * num_rgb_avg).
    NISAR Non-science multi-channel aka diagnostic mode # 2 (DM2),
    single-channel SAR (non-NISAR or DM1 NISAR) and composite DBFed
    SAR data (NISAR science mode) L0 products are all supported.

    Returns
    -------
    isce3.core.LUT2d
        Doppler values (Hz) as a function of `x=`slant range (m) and
        `y=`azimuth/pulse time (sec)
    isce3.core.DateTime
        Reference epoch UTC time for azimuth/pulse times
    np.ndarray(bool)
        Mask array for valid averaged range bins
    np.ndarray(float32)
        Correlation coefficients within [0,1]
    str
        TxRx polarization of the product
    float
        Center frequency of the `freq_band` in (Hz)
    np.ndarray or None
        Prototype filter coeffs centered at chirp center frequency (LPF)
        if `subband=True`, otherwise None.

    Raises
    ------
    ValueError
        For bad input parameters or non-existent polarization and/or
        frequency band.
        Missing `ant` object (None) in case of multi-channel (DM2) raw data.
    RuntimeError
        For dithered PRF.
        Less than 2 azimuth blocks.
        Too many invalid range bins w.r.t polyfit degree in case of polyfit.

    References
    ----------
    .. [GHAEMI2018]  H. Ghaemi and S. Durden, 'Pointing Estimation Algorithms
        and Simulation Results', JPL Report, February 2018.
    .. [MADSEN1989]  S. Madsen, 'Estimating The Doppler Centroid of SAR Data',
       IEEE Transaction On Aerospace and Elect Sys, March 1989.
    .. [BAMLER1991]  R. Bamler and H. Runge, 'PRF-Ambiguity Resolving by
        Wavelength Diversity', IEEE Transaction on GeoSci and Remote Sensing,
        November 1991.

    """
    # List of Constants
    num_subband = 3
    ripple_flt = 0.2  # passband ripple of subband filter (dB)
    rolloff_flt = 1.25  # roll-off of subband filter
    stopatt_flt = 27.0  # stop-band attenuation of subband filter (dB)

    # check inputs
    if polyfit_deg < 1:
        raise ValueError('polyfit_deg must be greater than 0')
    if az_block_dur <= 0.0:
        raise ValueError('az_block_dur must be a positive value')
    if (time_interval <= 0.0 or time_interval > az_block_dur):
        raise ValueError(
            'time_interval must be a positive value less than az_block_dur')
    if num_rgb_avg < 1:
        raise ValueError('Number of range bins must be a positive value')
    # set logger
    if logger is None:
        logger = set_logger("DopplerLUT")

    # check if there is matplotlib package needed for plotting if requested
    if plot:
        if plt is None:
            logger.warning('No plots due to missing package "matplotlib"!')
            plot = False

    # Check frequency band
    if freq_band not in raw.polarizations:
        raise ValueError(
            'Wrong frequency band! The available bands -> '
            f'{list(raw.polarizations)}'
        )
    logger.info(f"Frequency band -> '{freq_band}'")
    #  check for txrx_pol
    list_txrx_pols = raw.polarizations[freq_band]
    if txrx_pol is None:
        txrx_pol = list_txrx_pols[0]
    elif txrx_pol not in list_txrx_pols:
        raise ValueError(
            f'Wrong TxRx polarization! The available ones -> {list_txrx_pols}')
    logger.info(f"TxRx Pol -> '{txrx_pol}'")

    # Get chirp parameters and wavelength
    centerfreq, samprate, _, pulsewidth = \
        raw.getChirpParameters(freq_band, txrx_pol[0])

    wavelength = speed_of_light / centerfreq

    bandwidth = raw.getRangeBandwidth(freq_band, txrx_pol[0])

    # Get Pulse/azimuth time and ref epoch
    epoch_utc, az_time = raw.getPulseTimes(freq_band,
                                           txrx_pol[0])
    epoch_utc_str = epoch_utc.isoformat()
    # get PRF and check for dithering
    prf = raw.getNominalPRF(freq_band, txrx_pol[0])
    dithered = raw.isDithered(freq_band, txrx_pol[0])
    pri = 1. / prf

    if dithered:
        raise RuntimeError("Dithered PRF is not supported!")
    logger.info(f'Fast-time sampling rate -> {samprate * 1e-6:.2f} (MHz)')
    logger.info(f'Chirp bandwidth -> {bandwidth * 1e-6:.2f} (MHz)')
    logger.info(f'Chirp pulsewidth -> {pulsewidth * 1e6:.2f} (us)')
    logger.info(f'Chirp center frequency -> {centerfreq * 1e-6:.2f} (MHz)')
    logger.info(f'PRF -> {prf:.3f} (Hz)')

    # Get raw dataset
    raw_dset = raw.getRawDataset(freq_band, txrx_pol)
    # multi channel (DM2) versus single channel
    if raw_dset.ndim == 3:
        logger.info('Multi-channel SAR (DM2) is assumed!')
        is_multi_chanl = True
        num_chnl, tot_pulses, tot_rgbs = raw_dset.shape
        logger.info('Shape of the echo data (channel, pulses, ranges) -> '
                    f'{num_chnl, tot_pulses, tot_rgbs}')
        # existance of antenna object is a must!
        if ant is None:
            raise ValueError('Multi-channel (DM2) raw requires'
                             ' antenna object!')
        # get overlap boundaries between beams for only active RX channels
        # these are beam "transition" points suitable for forming 1-tap DBFed
        # composite echo range line later on!
        el_trans, az_trans = ant.locate_beams_overlap(txrx_pol[1])
        # pick those EL angles at beams transition that are within the list of
        # active RX channels
        list_rx_active = raw.getListOfRxTRMs(freq_band, txrx_pol)
        logger.info(f'List of active RX channels -> {list_rx_active}')
        el_trans = el_trans[list_rx_active[:-1] - 1]
        logger.info(
            f'EL angles @ beams transitions -> {np.rad2deg(el_trans)} (deg)')
        logger.info('AZ angle for all beams transitions -> '
                    f'{np.rad2deg(az_trans)} (deg)')
        # check list of beams to be excluded from polyfit if any for DM2
        if exclude_beams is not None:
            exclude_beams = set(exclude_beams)
            logger.info('Beams to be excluded in DM2 Doppler polyftting '
                        f'-> {exclude_beams}')
            if not exclude_beams.issubset(list_rx_active):
                logger.warning(
                    f'Excluded beams {exclude_beams} is out of range of '
                    f'active RX channels {list_rx_active}!'
                    )
                exclude_beams.intersection_update(list_rx_active)
                logger.warning(
                        f'Updated list of excluded beams -> {exclude_beams}')
        # initialize DEM object if None
        if dem is None:
            dem = DEMInterpolator()
    else:  # single channel
        logger.info('Single (composite) channel SAR is assumed!')
        is_multi_chanl = False
        tot_pulses, tot_rgbs = raw_dset.shape
        logger.info('Shape of the echo data (pulses, ranges) -> '
                    f'{tot_pulses, tot_rgbs}')

    # blocksize in range
    if num_rgb_avg > (tot_rgbs // 2):
        raise ValueError(
            'Number of range bins to be averaged must be equal or less than '
            f'{tot_rgbs // 2} to result in at least 2 range blocks!'
        )
    logger.info(f'Number of range bins per range block -> {num_rgb_avg}')
    num_blk_rg = tot_rgbs // num_rgb_avg
    logger.info(f'Number of range blocks -> {num_blk_rg}')

    # Get prototype LPF coeffs if subbanding is requested
    coeff_lpf = None
    if subband:
        logger.info("Perform sub-banding on echo data!")
        logger.info(f'Number of subbands -> {num_subband}')
        bw_flt = bandwidth / num_subband
        coeff_lpf = cheby_equi_ripple_filter(samprate, bw_flt, rolloff_flt,
                                             ripple_flt, stopatt_flt,
                                             force_odd_len=True)
        len_flt = len(coeff_lpf)
        logger.info(
            'Subbanding filter passband bandiwdth -> '
            f'{bw_flt * 1e-6:.2f} (MHz)'
        )
        logger.info(
            f'Subbanding filter passband ripple -> {ripple_flt:.2f} (dB)')
        logger.info(
            f'Subbanding filter stopband attenuaton -> {stopatt_flt:.2f} (dB)')
        logger.info(f'Subbanding filter rolloff factor -> {rolloff_flt}')
        logger.info(f'Length of subband filter -> {len_flt}')

        # convolution length and group delay
        len_conv = tot_rgbs + len_flt - 1
        grp_del = len_flt // 2

        # Get number of FFT and total group delay caused by rgcomp + subband
        nfft = fft.next_fast_len(len_conv)
        logger.info(
            f'Number of FFT points in rangecomp and/or subbanding -> {nfft}')

        # Get FFT of the prototype LPF
        coeff_lpf_fft = fft.fft(coeff_lpf, nfft)
        slice_grp_del = slice(grp_del, grp_del + tot_rgbs)

        # Calculate center frequencies for only three bands:first, mid,and last
        fcnt_first = (1 - num_subband) / (2. * num_subband) * bandwidth
        fcnt_last = -fcnt_first
        fcnt_subbands = [fcnt_first, 0.0, fcnt_last]
        fcnt_rf_subbands = np.asarray(fcnt_subbands) + centerfreq
        logger.info(
            'The RF center freq of subbands -> '
            '({:.2f}, {:.2f}, {:.2f}) (MHz)'.format(*(fcnt_rf_subbands * 1e-6))
        )

        # Mixer func for up/down conversion of LPF -> BPF
        def mixer_fun(fc):
            return np.exp(
                1j * 2.0 * np.pi * fc / samprate * np.arange(len_flt))

        # Get freq-domain BPFs Coeffs for two edge bands from LPF prototype
        coef_bpf_fft_first = fft.fft(coeff_lpf * mixer_fun(fcnt_first), nfft)
        coef_bpf_fft_last = fft.fft(coeff_lpf * mixer_fun(fcnt_last), nfft)

        # plot three suband BPF in frequency domain
        if plot:
            plt_name = f'Subband_Filter_Plot_Freq{freq_band}_Pol{txrx_pol}.png'
            name_plot = os.path.join(out_path, plt_name)
            _plot_subband_filters(samprate, centerfreq, coef_bpf_fft_first,
                                  coeff_lpf_fft, coef_bpf_fft_last, name_plot)

    # get complex echo for all range bins but limited pulses
    dop_method = dop_method.upper()
    logger.info(
        f'Doppler estimator method per block and per band -> {dop_method}')
    # form a generic doppler estimator function covering both methods
    if dop_method == 'SDE':
        def time_dop_est(echo, prf, lag=1, axis=None):
            return sign_doppler_est(echo, prf, lag, axis), 1
    elif dop_method == 'CDE':
        time_dop_est = corr_doppler_est
    else:
        raise ValueError(
            f'Unexpected time-domain Doppler method "{dop_method}"')

    # get orbit object and update its ref epoch if necessary to match
    # that of L0B echo for both internal and external cases.
    # This is needed for absolute Doppler and ambiguity computation.
    if orbit is None:
        logger.info('Orbit data stored in L0B will be used.')
        orbit = raw.getOrbit()
    else:  # use external orbit data
        logger.info('External orbit data will be used.')
    if orbit.reference_epoch != epoch_utc:
        logger.warning(
            'Reference epoch of orbit, '
            f'{orbit.reference_epoch.isoformat()}, and that'
            f' of L0B pulse time, {epoch_utc_str}, '
            'does not match!'
        )
        logger.warning('Reference epoch of L0B pulsetime will be used!')
        orbit = orbit.copy()
        orbit.update_reference_epoch(epoch_utc)

    # get attitude object and update its ref epoch if necessary to match
    # that of L0B echo for both internal and external cases.
    # This is needed for absolute Doppler and ambiguity computation.
    if attitude is None:
        logger.info('Attitude data stored in L0B will be used.')
        attitude = raw.getAttitude()
    else:  # use external attitude data
        logger.info('External attitude data will be used.')
    if attitude.reference_epoch != epoch_utc:
        logger.warning(
            'Reference epoch of attitude, '
            f'{attitude.reference_epoch.isoformat()}, and that'
            f' of L0B pulse time, {epoch_utc_str}, '
            'does not match!'
        )
        logger.warning('Reference epoch of L0B pulsetime will be used!')
        attitude = attitude.copy()
        attitude.update_reference_epoch(epoch_utc)

    # get ~mid-swath look vector in antenna frame to be used for doppler
    # ambiguity calculations from orbit+attitude data for all azimuth blocks
    look_vec_ant = _compute_look_vector_ant(raw, ant, freq_band, txrx_pol)
    logger.info(f'Look vector in antenna XYZ -> {look_vec_ant}')

    # Get slant ranges per range  block centered at each block
    sr_lsp = raw.getRanges(freq_band, txrx_pol[0])
    sr_spacing = sr_lsp.spacing * num_rgb_avg
    sr_start = sr_lsp.first + 0.5 * sr_spacing
    sr_stop = sr_start + (num_blk_rg - 1) * sr_spacing
    slrg_per_blk = np.linspace(sr_start, sr_stop, num=num_blk_rg)

    # form the blocks of range lines / azimuth bins
    len_az_blk_dur, len_tm_int, num_blk_az = _get_az_block_interval_len(
        tot_pulses, az_block_dur, prf, time_interval)

    logger.info(
        f'Final full azimuth block duration -> {len_az_blk_dur/prf:.3f} (sec)')
    logger.info(
        f'Number of range lines of a full azimuth block -> {len_az_blk_dur}')
    logger.info(
        f'Time interval between azimuth blocks -> {len_tm_int/prf:.3f} (sec)')
    logger.info(
        'Number of range line seperation between azimuth blocks -> '
        f'{len_tm_int}'
    )
    logger.info(f'Total number of azimuth blocks -> {num_blk_az}')

    slice_lines = _azblk_slice_gen(
        tot_pulses, len_az_blk_dur, len_tm_int, num_blk_az)

    # parse valid subswath index for all range lines used later
    valid_sbsw_all = raw.getSubSwaths(freq_band, txrx_pol[0])
    # initialized output mask array for averaged range bins for
    # all azimuth blocks
    mask_rgb_avg_all = np.zeros((num_blk_az, num_blk_rg), dtype='bool')

    # initialize correlator coeff for all range bins and azimuth blocks
    corr_coef = np.ones((num_blk_az, num_blk_rg), dtype='float32')

    # initialize the azimuth time block and set an intermediate var
    half_az_blk_dur = (len_az_blk_dur - 1) / 2
    az_time_blk = np.full(num_blk_az, az_time[0] + half_az_blk_dur * pri,
                          dtype=float)
    tm_int_pri_prod = len_tm_int * pri

    # doppler centroid map is azimuth block by slant-range block
    dop_cnt_map = np.zeros((num_blk_az, num_blk_rg), dtype='float32')

    # loop over azimuth blocks /range line blocks
    for n_azblk, slice_line in enumerate(slice_lines):
        num_lines = slice_line.stop - slice_line.start
        logger.info(
            f'(start, stop) of AZ block # {n_azblk + 1} -> '
            f'{slice_line.start, slice_line.stop}'
        )
        logger.info(
            'Block size (lines, ranges) for Doppler estimation -> '
            f'({num_lines, num_rgb_avg})'
        )
        # azimuth time at mid part of the azimuth block
        az_time_blk[n_azblk] += n_azblk * tm_int_pri_prod

        # compute position, velocity and quaternion of the spacecraft at
        # mid time of the azimuth block
        quat_mid = attitude.interpolate(az_time_blk[n_azblk])
        pos_mid, vel_mid = orbit.interpolate(az_time_blk[n_azblk])

        # get decoded raw echoes of one azimuth block and for all range bins
        mask_bad = False
        if is_multi_chanl:
            echo, rgb_limits = form_single_tap_dbf_echo(
                raw_dset, slice_line, el_trans, az_trans,
                pos_mid, vel_mid, quat_mid, sr_lsp, dem
                )
            if exclude_beams is not None:
                mask_bad = np.zeros(rgb_limits[-1], dtype='bool')
                for n_b in exclude_beams:
                    rgb_slice = slice(rgb_limits[n_b - 1], rgb_limits[n_b])
                    mask_bad[rgb_slice] = True
                    logger.info(
                        f'For AZ block # {n_azblk + 1}, exclude slant ranges '
                        f'(m, m) within -> ({sr_lsp[rgb_slice.start] :.3f}, '
                        f'{sr_lsp[rgb_slice.stop - 1]:.3f}) '
                        )
        else:  # single channel
            echo = raw_dset[slice_line]

        # create a mask for invalid/bad range bins for any reason
        # invalid values are either nan or zero but this does not include
        # TX gaps that may be filled with TX chirp!
        mask_bad |= (np.isnan(echo) | np.isclose(echo, 0)).sum(axis=0) > 0

        # build a mask array of range bins assuming fixed PRF within
        # each azimuth block. This is needed in case the TX gaps are filled
        # with TX chirp rather than invalid/bad value!
        mask_valid_rgb = _form_mask_valid_range(
            tot_rgbs, valid_sbsw_all[:, slice_line.start, :])
        mask_valid_rgb &= _form_mask_valid_range(
            tot_rgbs, valid_sbsw_all[:, slice_line.stop - 1, :])

        # Update valid mask with invalid range bins over all range lines
        mask_valid_rgb[mask_bad] = False

        # decimate the range bins mask to fill in mask for averaged range bins
        # per azimuth block. Make sure a valid averaged block contains all
        # valid range bins otherwise set to invalid.
        mask_rgb_avg_all[n_azblk] = mask_valid_rgb[
            :num_blk_rg * num_rgb_avg].reshape((num_blk_rg, num_rgb_avg)).sum(
                axis=1) == num_rgb_avg

        # form mask for NaN values in echo and replace it with 0
        echo[np.isnan(echo)] = 0.0

        if subband:
            echo_sub_first = np.zeros(echo.shape, dtype=echo.dtype)
            echo_sub_last = np.copy(echo_sub_first)
            # Loop over range lines for one azimuth block
            for line in range(num_lines):
                # apply subband BPF in freq domain
                rgc_line_fft = fft.fft(echo[line], nfft)
                # first band
                rgc_line_fft_edge = rgc_line_fft * coef_bpf_fft_first
                echo_sub_first[line] = fft.ifft(
                    rgc_line_fft_edge)[slice_grp_del]
                # last band
                rgc_line_fft_edge = rgc_line_fft * coef_bpf_fft_last
                echo_sub_last[line] = fft.ifft(
                    rgc_line_fft_edge)[slice_grp_del]
                # mid band
                rgc_line_fft *= coeff_lpf_fft
                # go back to time and get rid of all group delays
                echo[line] = fft.ifft(rgc_line_fft)[slice_grp_del]

        # estimate doppler per band, per azimuth block over all range blocks
        dop_cnt = np.zeros(num_blk_rg, dtype="float32")
        for n_blk in range(num_blk_rg):
            slice_rgb = slice(n_blk * num_rgb_avg, (n_blk + 1) * num_rgb_avg)
            # CDE or SDE
            dop_cnt[n_blk], corr_coef[n_azblk, n_blk] = time_dop_est(
                echo[:, slice_rgb], prf)

        if subband:
            dop_cnt_bands = np.zeros((3, num_blk_rg), dtype="float32")
            dop_cnt_bands[1] = dop_cnt
            for n_blk in range(num_blk_rg):
                slice_rgb = slice(n_blk * num_rgb_avg,
                                  (n_blk + 1) * num_rgb_avg)
                # CDE or SDE for each subband
                dop_cnt_bands[0, n_blk], corr_coef_low = time_dop_est(
                    echo_sub_first[:, slice_rgb], prf)
                dop_cnt_bands[-1, n_blk], corr_coef_high = time_dop_est(
                    echo_sub_last[:, slice_rgb], prf)
                # sum correlation coeff among all three bands
                corr_coef[n_azblk, n_blk] += (corr_coef_low + corr_coef_high)
            # average correlation coeff among all three bands
            corr_coef[n_azblk] /= 3.0

            # perform doppler unwrapping over three bands
            dop_cnt_bands = unwrap_doppler(dop_cnt_bands, prf)

            # perform linear (1st degree) polyfit over 3 bands for
            # all range blocks
            # IF version: np.polyfit(fcnt_subbands, dop_cnt_bands, 1)
            pf_coef_subbands = np.polyfit(fcnt_rf_subbands, dop_cnt_bands, 1)

            # eval doppler centroid at the center freq of the chirp
            # IF version: pf_coef_subbands[1]
            dop_cnt = np.polyval(pf_coef_subbands, centerfreq)

        # get valid dopplers in range
        dop_cnt_valid = dop_cnt[mask_rgb_avg_all[n_azblk]]

        # Unwrap only valid Doppler values along range
        dop_cnt_valid = unwrap_doppler(dop_cnt_valid, prf)

        # Polyfit valid-only doppler centroids over slant ranges
        if polyfit:  # replace actual value by polyfitted ones
            sr_valid = slrg_per_blk[mask_rgb_avg_all[n_azblk]]
            # check if the number of valid range blocks > polyfit_deg
            if sr_valid.size <= polyfit_deg:
                raise RuntimeError(
                    'Too many bad range bins! Polyfit requires at least '
                    f'{polyfit_deg + 1} valid range blocks or '
                    f'{(polyfit_deg + 1) * num_rgb_avg} valid range bins!'
                )
            pf_coef_dop_cnt = np.polyfit(sr_valid, dop_cnt_valid, polyfit_deg)
            dop_cnt_map[n_azblk] = np.polyval(pf_coef_dop_cnt, slrg_per_blk)
            # replace zero correlation factors with averaged ones over
            # TX gaps or invalid regions given interpolation of Doppler
            # via polyfit/smoothing of valid regions.
            corr_coef_avg = corr_coef[n_azblk,
                                      mask_rgb_avg_all[n_azblk]].mean()
            corr_coef[n_azblk, ~mask_rgb_avg_all[n_azblk]] = corr_coef_avg
        else:  # keep the actual values
            # store the valid Dopplers unwrapped over ranges
            dop_cnt[mask_rgb_avg_all[n_azblk]] = dop_cnt_valid
            dop_cnt_map[n_azblk] = dop_cnt

        # plot ambiguous Doppler centroid purely extracted from echo
        # per azimuth block
        if plot:
            _plot_save_dop(n_azblk, slrg_per_blk, dop_cnt,
                           az_time_blk[n_azblk], epoch_utc_str, out_path,
                           freq_band, txrx_pol, polyfit_deg,
                           mask_rgb_avg_all[n_azblk], prf)

        # calculate absolute doppler and its ambiguity number to be added
        # to estimated ambiguous doppler centroid for final LUT2d
        # Use median (or mean) of measured ambiguous doppler over slant ranges
        # obatained from echo to be used in doppler ambiguity calculation.
        # Perhaps Median is more suited in case of skewed Doppler outliers
        # due to presence of man-made or non-homogenous targets in homogenous
        # scene.
        dop_echo = np.median(dop_cnt_valid)
        dop_abs, dop_amb_num = _compute_doppler_abs_ambiguity(
            vel_mid, quat_mid, look_vec_ant, prf, wavelength, dop_echo)

        logger.info('Absolute Doppler calculated from attitude for block # '
                    f'{n_azblk + 1} -> {dop_abs:.1f} (Hz)')
        logger.info('Calculated Doppler ambiguity number from attitude '
                    f'for block # {n_azblk + 1} -> {dop_amb_num}')
        # Adjust estimated Doppler centroids if doppler ambiguity is non zero.
        if dop_amb_num:
            dop_cnt_map[n_azblk] += dop_amb_num * prf

    # form Doppler LUT2d object
    dop_lut = LUT2d(slrg_per_blk, az_time_blk, dop_cnt_map)

    # given estimation of invalid range bins from polyfit,
    # set the mask to be all True after polyval!
    if polyfit:
        mask_rgb_avg_all[:] = True

    return dop_lut, epoch_utc, mask_rgb_avg_all, corr_coef, txrx_pol, \
        centerfreq, coeff_lpf


# list of  private helper functions:


def _compute_look_vector_ant(raw, ant, freq_band, txrx_pol):
    """
    Compute look vector or antenna pointing unit vector in antenna
    Cartesian coordinate.

    In case of multi-beam, EL angle is extratced from the middle beam
    (~mid swath) with RX polarization while the azimuth angle is the averaged
    one from Tx+Rx polarizations of middle beam!

    Parameters
    ----------
    raw : nisar.products.readers.Raw.RawBase
    ant : nisar.products.readers.antenna.AntennaParser
    freq_band : {'A', 'B'}
    txrx_pol : str

    Returns
    -------
    np.ndarray(float)
        Three-element unit look vector in antenna XYZ

    Notes
    -----
    In case `ant` is set to None, the EL and AZ angles of look vector
    will be assumed to be zero radians and the antenna frame will be assumed
    to be "EL-AND-AZ".

    """
    # check whether antenna object exists
    if ant is None:
        el_ang = 0  # in (rad)
        az_ang = 0  # in (rad)
        frame = Frame()
    else:
        # get mid beam number from raw object for a desired freq and pol
        list_rx_beams = raw.getListOfRxTRMs(freq_band, txrx_pol)
        mid_beam = list_rx_beams[list_rx_beams.size // 2]

        # Get elevation angle of the peak at around mid swath for RX pol only
        ant_rx_el = ant.el_cut(beam=mid_beam, pol=txrx_pol[1])
        idx_peak = abs(ant_rx_el.copol_pattern).argmax()
        el_ang = ant_rx_el.angle[idx_peak]  # in (rad)

        # Get azimuth angle from EL cut at around mid swath for Tx+Rx pol

        az_ang = ant_rx_el.cut_angle  # in (rad)
        # if TX pol is different from RX pol then take average of both
        if txrx_pol[0] != txrx_pol[1]:
            # To cover TX L/R circular pol, the following condition is needed
            if txrx_pol[1] == 'H':
                tx_pol = 'V'
            else:
                tx_pol = 'H'
            ant_tx_el = ant.el_cut(beam=mid_beam, pol=tx_pol)
            az_ang += ant_tx_el.cut_angle
            az_ang *= 0.5
        frame = ant.frame

    return frame.sph2cart(el_ang, az_ang)


def _compute_doppler_abs_ambiguity(vel, quat, look_vec,
                                   prf, wavelength, dop_echo):
    """
    Compute absolute Doppler centroid and its ambiguity number w.r.t nominal
    PRF at a desired azimuth time for a unit look vector in antenna frame.

    Both mechanical (attitude) and electrical (antenna) aspects
    are taken into account per knowledge.

    Parameters
    ----------
    vel : np.ndarray(float)
        3-element velocity vector of the spacecraft in ECEF
    quat : isce3.core.Quaternion
        Contains 4-element quaternion vector of the spacecraft attitude
    look_vec : np.ndarray(float)
        Three-element unit look vector in antenna Cartesian coordinate
    prf : float
        Pulse repetition frequency in (Hz).
        It is the nominal/mean value for dithered case.
    wavelength : float
        Wavelength at the center of the TX chirp in (m)
    dop_echo : float
        Doppler value estimated from echo in (Hz)

    Returns
    -------
    float
        Absolute doppler centroid in (Hz)
    int
        Ambiguity number of PRF, a signed value!

    Notes
    -----
    It's assumed that orbit and attitude objects have the same reference epoch
    and the azimuth time is seconds from their common reference epoch.


    """
    # convert look vector from antenna to ECEF
    lk_vec_ecef = quat.rotate(look_vec)
    # compute doppler
    dop = 2.0 / wavelength * vel.dot(lk_vec_ecef)
    # get doppler ambiguity
    amb_num = round((dop - dop_echo) / prf)
    return dop, amb_num


def _plot_subband_filters(samprate: float, centerfreq: float,
                          coef_bpf_fft_first: np.ndarray,
                          coeff_lpf_fft: np.ndarray,
                          coef_bpf_fft_last: np.ndarray,
                          name_plot: str):
    """Plot spectrum of three subbands filters"""
    # form RF frequency vector
    nfft = coef_bpf_fft_first.size
    min_rf_freq = centerfreq - 0.5 * samprate
    freq = min_rf_freq + (samprate / nfft) * np.arange(nfft)
    freq *= 1e-6  # (MHz)
    def amp2db_fft(amp): return 20 * np.log10(abs(fft.fftshift(amp)))
    plt.figure()
    plt.plot(freq, amp2db_fft(coef_bpf_fft_first), 'b',
             freq, amp2db_fft(coeff_lpf_fft), 'g',
             freq, amp2db_fft(coef_bpf_fft_last), 'r',
             linewidth=2)
    plt.legend(['First', 'Mid', 'Last'], loc='best')
    plt.xlabel('RF Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Spectrum of the three subband filters')
    plt.ylim([-50.0, 1.0])
    plt.grid(True)
    plt.savefig(name_plot)
    plt.close()


def _plot_save_dop(n_azblk: int, slrg_per_blk: np.ndarray, dop_cnt: np.ndarray,
                   az_time: float, epoch_utc_str: str, out_path: str,
                   freq_band: str, txrx_pol: str, polyfit_deg: int,
                   mask_valid_rgb: np.ndarray, prf: float):
    """Plot Doppler as a function Slant range and save it as PNG file"""
    fig = plt.figure(n_azblk, figsize=(8, 7))
    ax = fig.add_subplot(111)
    # only polyfit over the valid range blocks!
    dop_cnt = unwrap_doppler(dop_cnt, prf)
    pf_coeff_dop_rg = np.polyfit(slrg_per_blk[mask_valid_rgb],
                                 dop_cnt[mask_valid_rgb], polyfit_deg)
    pv_dop_rg = np.polyval(pf_coeff_dop_rg, slrg_per_blk)
    slrg_km = slrg_per_blk * 1e-3
    ax.plot(slrg_km, dop_cnt, 'r*--', slrg_km, pv_dop_rg, 'b--')
    ax.legend(["Echo", f"PF(order={polyfit_deg})"], loc='best')
    # get statistics of difference between measurement and polyfit
    # only over valid regions of the swath!
    diff_dop_pf = dop_cnt[mask_valid_rgb] - pv_dop_rg[mask_valid_rgb]
    plt_textstr = '\n'.join((
        'Deviation from PF:',
        r'$\mathrm{MEAN}$=%.1f Hz' % diff_dop_pf.mean(),
        r'$\mathrm{STD}$=%.1f Hz' % diff_dop_pf.std()))
    plt_props = dict(boxstyle='round', facecolor='green', alpha=0.5)
    ax.text(0.5, 0.2, plt_textstr, transform=ax.transAxes, fontsize=10,
            horizontalalignment='center', verticalalignment='center',
            bbox=plt_props)
    ax.grid(True)
    ax.set_title(
        'Doppler Centroids  (unwrapped ambiguous) from Raw Echo\n@ '
        f'azimuth-time = {az_time:.3f} sec\nsince {epoch_utc_str}'
    )
    ax.set_ylabel("Doppler (Hz)")
    ax.set_xlabel("Slant Range (Km)")
    fig.savefig(os.path.join(
        out_path, 'Doppler_SlantRange_Plot_Freq'
        f'{freq_band}_Pol{txrx_pol}_AzBlock{n_azblk + 1}.png'
    ))
    plt.close()


def _get_az_block_interval_len(num_pls: int, az_block_dur: float, prf: float,
                               time_interval: float):
    """Get block size and interval lengths for azimuth blocks

    Returns
    -------
    int
        length of a full azimuth block
    int
        length of time interval
    int
        total number of blocks , full + partial (last block).

    """
    # time interval shall be equal ot less than block duration
    time_interval = min(time_interval, az_block_dur)
    # make sure the min time interval is one PRI!
    len_tm_int = max(int(time_interval * prf), 1)
    len_az_blk_dur = int(az_block_dur * prf)
    # if the block_dur + interval is too large then raise an exception!
    if (len_tm_int + len_az_blk_dur) > num_pls:
        raise ValueError(
            'Sum of azimuth block duration and time interval is large than '
            f'echo duration {(num_pls - 1) / prf} (sec)!'
        )
    # get number of blocks which must be at least 2!
    num_blk_az = int(np.ceil((num_pls - len_az_blk_dur) / len_tm_int)) + 1
    if num_blk_az < 2:
        raise RuntimeError(
            'At least two azimuth blocks are required to form LUT2d! Try to '
            'reduce time interval!'
        )
    return len_az_blk_dur, len_tm_int, num_blk_az


def _azblk_slice_gen(num_pls: int, len_az_blk_dur: int, len_tm_int: int,
                     num_blk_az: int):
    """Slice index generator for azimuth blocks/range lines"""
    # generate slice for azimuth block indexing
    i_str = 0
    i_stp = len_az_blk_dur
    for bb in range(num_blk_az):
        yield slice(i_str, i_stp)
        i_str += len_tm_int
        i_stp = min(i_str + len_az_blk_dur, num_pls)


def _form_mask_valid_range(tot_rgbs, rgb_valid_sbsw):
    """Form valid mask for range bins for a specific range line.

    Parameters
    ----------
    tot_rgbs : int
        Total number of range bins
    rgb_valid_sbsw : np.ndarray(np.ndarray(int))
        2-D array-like integers for valid range bins of a specific range line

    Returns
    -------
    np.ndarray(bool)
        Mask array for valid range bins

    """
    msk_valid_rg = np.zeros(tot_rgbs, dtype=bool)
    for start_stop in rgb_valid_sbsw:
        msk_valid_rg[slice(*start_stop)] = True
    return msk_valid_rg
