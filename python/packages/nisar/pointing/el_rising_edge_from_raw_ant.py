"""
Function to estimate roll angle offset from rising edge of two-way power
patterns extracted from L0b and antenna products via edge method.
"""
import os
import bisect
import numpy as np
from numbers import Number
from scipy.interpolate import interp1d
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from isce3.antenna import (roll_angle_offset_from_edge, ElPatternEst,
                           ant2rgdop, Frame)
from isce3.geometry import DEMInterpolator, look_inc_ang_from_slant_range
from nisar.log import set_logger
from isce3.core import Ellipsoid, speed_of_light, Poly1d, TimeDelta
from nisar.antenna import TxTrmInfo, compute_transmit_pattern_weights


def el_rising_edge_from_raw_ant(raw, ant, *, dem_interp=None,
                                freq_band='A', txrx_pol=None,
                                orbit=None, attitude=None,
                                az_block_dur=3.0, beam_num=None,
                                dbf_pow_norm=True, apply_weight=True,
                                plot=False, out_path='.', logger=None):
    """
    Estimate off-nadir angle offset in elevation (EL) direction, called
    roll offset, as a function of azimuth time from either composite
    digitally beam formed (DBF) SweepSAR raw echo or single-channel SAR
    raw echo.

    See references [1]_ and [2]_ for the algorithm and the demo.

    Parameters
    ----------
    raw : nisar.products.readers.Raw.RawBase
        Raw L0B product parser base object
    ant : nisar.products.readers.antenna.AntennaParser
        Antenna HDF5 product parser object
    dem_interp : isce3.core.DEMInterpolator, optional
        DEMInterpolator instance. If not provided, WGS84 reference
        Ellipsoid w/o any topography will be assumed.
    freq_band : {'A', 'B'}
        Frequency band in multi-band TX chirp.
    txrx_pol : {'HH', 'HV', 'VH', 'VV'}, optional
        TxRx polarization. If not provided the first product under
        `freq_band` will be used.
    orbit : isce3.core.Orbit, optional
        If specified, this will be used in place of the orbit data
        stored in L0B.
    attitude : isce3.core.Attitude, optional
        If specified, this will be used in place of attitude data
        stored in L0B.
    az_block_dur : float, default=3.0
        Duration of azimuth block in seconds. The last azimuth block
        can have slightly larger duration depending on total azimuth
        duration.
        The max block duration will be limited to total azimuth
        duration of echo if it is too large.
        The min block duration must be equal or larger than nominal mean
        PRI (pulse repetition interval).
    beam_num : int, optional
        Beam number used for fetching a desired beam from antenna object simply
        for single-channel raw echo. It will be ignored for SweepSAR case!
    dbf_pow_norm : bool, default=True
        Whether or not DBF weighting coefficients used in forming 2-way Rx DBF
        power pattern shall be power normalized. This feature is only used for
        SweepSAR case.
    apply_weight : bool, default=True
        Whether or not to apply SNR-based weighting to the cost function.
    plot : bool, default=False
        If True, it will generate one PNG plot per azimuith block to compare
        polyfitted echo data versus that of antenna one in EL over rising edge
        region. If the block is invalid, then no plot will be generated for
        that block.
    out_path : str, default='.'
        Ouput directory for dumping PNG files, if `plot` is True.
    logger : logging.Logger, optional
        If not provided a logger with StreamHandler will be set.

    Returns
    -------
    np.ndarray(float32)
        Off-nadir/Roll angle offset in (rad) to be added to antenna off-nadir
        angle to line up with that of echo. One value per azimuth block.
    np.ndarray(isce3.core.DateTime)
        Azimuth DateTime object representing mid azimuth date/time of each
        azimuth block.
    np.ndarray(float32)
        2-D array containing corrected (true) EL angle coverage [first, last]
        in (rad). One pair per azimuth block.
        The shape is (azimuth blocks by 2).
    np.ndarray(float32)
        2-D array containing slant range coverage [first, last] of echo in (m).
        One pair per azimuth block. The shape is (azimuth blocks by 2).
    np.ndarray(float32)
        2-D array containing look(off-nadir) angle coverage [first, last] of
        echo in (rad). One pair per azimuth block. The shape is
        (azimuth blocks by 2).
        Note that these angles are not corrected by angle offset. They simply
        correspond to slant ranges of echo in the radar domain.
    np.ndarray(bool)
        Mask array for valid azimth blocks.
    np.ndarray(bool)
        Convergence flag in cost function
    np.ndarray(isce3.core.Poly1d)
        3rd-order Poly1d object to represent rising edge of echo power in (dB)
        as a function of look angle in (rad). One object per azimuth block.
    np.ndarray(isce3.core.Poly1d)
        3rd-order Poly1d object to represent rising edge of two-way antenna
        power pattern in (dB) as a function look angle in (rad). One object
        per azimuth block.
    np.ndarray(isce3.core.Poly1d)
        3rd-order Poly1d object to represent weighting factor in (dB) as a
        function of look angle in (rad). One object per azimuth block.

    Raises
    ------
    ValueError
        For bad input parameters or non-existent polarization and/or
        frequency band.
        Azimuth block duration is smaller than the mean PRI.
    RuntimeError
        Missing first 3 active TX/RX channels for NISAR SweepSAR case,
        and missing antenna pattern for the desired beam number for
        single-channel SAR cases such as ALOS1 PALSAR FDB/FSB.

        Not enough EL coverage of "ant" object at rising edge!

        Missing HPA CAL data for SweepSAR case.
    NotImplementedError
        For circular polarization on either Tx or RX side!

        X-Pol patterns do not have equal EL angle coverage!

    Notes
    -----
    For SweepSAR case, the first two channels shall be active on both TX and
    RX side and their respective antenna patterns must exist.

    For single-channel SAR case, the antenna pattern for the specified
    `beam_num` must exist.

    See reference [3]_ in regards to TX calibration components HPA and BYPASS.

    References
    ----------
    .. [1] Ghaemi H., and Durden S. "Pointing Estimation Algorithms &
        Simulation Results," JPL Report, Rev D., 2018.
    .. [2] Ghaemi H., "Formulation of Rising-Edge Method in EL Pointing
        Estimation & its Application to ALOS1 PALSAR Data," JPL Report, RevB,
        October 2020.
    .. [3] Ghaemi H., "DSI SweepSAR On-Board DSP Algorithms Description,"
        JPL D-95646, Rev 14.

    """
    # List of Const

    # prefix for EL rising edge PNG plot
    prefix = 'EL_Rising_Edge_Plot'
    # polyfit degree for echo, antenna, weights used in cost function
    pf_deg = 3
    # Number of taps in DBF process
    num_taps_dbf = 3
    # EL angle margin on either ends of antenna EL coverage of rising edge.
    # Shall be set to Max error = +/- ~0.5
    el_margin_deg = 0.5
    # Number of range bins of range block to be averaged
    size_rgb_avg = 8
    # EL angle resolution in cost function
    el_res_deg = 1e-3
    # 1-D Interpolation method used in EL direction for look angle -> EL angle
    interp_method_el = 'linear'

    # set logger
    if logger is None:
        logger = set_logger("ElRisingEdgeFromRawAnt")

    # check if there is matplotlib package needed for plotting if requested
    if plot:
        if plt is None:
            logger.warning('No plots due to missing package "matplotlib"!')
            plot = False

    # Check inputs

    # Check frequency band
    if freq_band not in raw.polarizations:
        raise ValueError('Wrong frequency band! The available bands -> '
                         f'{list(raw.polarizations)}')
    logger.info(f'Frequency band -> "{freq_band}"')
    #  check for txrx_pol
    list_txrx_pols = raw.polarizations[freq_band]
    if txrx_pol is None:
        txrx_pol = list_txrx_pols[0]
    else:
        if txrx_pol not in list_txrx_pols:
            raise ValueError('Wrong TxRx polarization! The available ones -> '
                             f'{list_txrx_pols}')
    # TODO: add Left/Right TX pattern formation to support compact Pol!
    if txrx_pol not in ('HH', 'VV', 'HV', 'VH'):
        raise NotImplementedError('The compact polarization is not supported.')
    logger.info(f'TxRx Pol -> "{txrx_pol}"')

    # check beam number value
    if beam_num is None:
        beam_num = 1
    else:
        if (beam_num < 1 or beam_num > ant.num_beams(txrx_pol[1])):
            raise ValueError('Beam number is out of range!')

    # Get raw dataset
    raw_dset = raw.getRawDataset(freq_band, txrx_pol)
    if raw_dset.ndim != 2:
        raise RuntimeError('Expected 2-D Science Raw Echo!')
    num_rgls, num_rgbs = raw_dset.shape
    logger.info('Shape of the Raw echo data (pulses, ranges) -> '
                f'({num_rgls, num_rgbs})')

    # EL angle margin  and resolution in radians
    el_margin = np.deg2rad(el_margin_deg)
    el_res = np.deg2rad(el_res_deg)

    # Get chirp parameters
    centerfreq, samprate, chirp_rate, chirp_dur = raw.getChirpParameters(
        freq_band, txrx_pol[0])
    logger.info(f'Fast-time sampling rate -> {samprate * 1e-6:.2f} (MHz)')
    logger.info(f'Chirp pulsewidth -> {chirp_dur * 1e6:.2f} (us)')
    logger.info(f'Chirp center frequency -> {centerfreq * 1e-6:.2f} (MHz)')
    # wavelength in meters
    wavelength = speed_of_light / centerfreq

    # slant range within chirp duration in (m)
    sr_chirp = 0.5 * speed_of_light * chirp_dur

    # get mean PRF and check for dithering
    prf = raw.getNominalPRF(freq_band, txrx_pol[0])
    dithered = raw.isDithered(freq_band, txrx_pol[0])
    logger.info(f'Mean PRF -> {prf:.3f} (Hz)')
    if dithered:
        logger.warning(
            'Dithered PRF! TX-gap related valid mask will all be set to True!'
            ' Mean PRF will be used to get number of lines in azimuth block!'
        )

    # get number of range lines per azimuth block and check if within
    # [1, num_rgls]
    num_rgl_block = round(az_block_dur * prf)
    if num_rgl_block > num_rgls:
        logger.warning('Azimuth block duration exceeds max duration. '
                       'It will be limited to total azimuth duration!')
        num_rgl_block = num_rgls
    if num_rgl_block < 1:
        raise ValueError('Azimuth block duration is smaller than mean PRI!')

    # get number of azimuth blocks
    num_azimuth_block = num_rgls // num_rgl_block
    logger.info(f'Total number of azimuth blocks  -> {num_azimuth_block}')

    # get list of active RX/TX channels to find out whether it is
    # single-channel SAR or a DBFed multi-channel SweepSAR.
    list_tx = raw.getListOfTxTRMs(freq_band, txrx_pol[0])
    logger.info(f'List of active TX channels -> {list_tx}')

    list_rx = raw.getListOfRxTRMs(freq_band, txrx_pol)
    logger.info(f'List of active RX channels -> {list_rx}')

    # check if certain channels exist on RX/TX side of echo and
    # if desired beam(s) exist in antenna file
    if list_rx.size > 1:  # SweepSAR case
        is_sweepsar = True
        logger.info(
            'It is a SweepSAR echo. All antenna beams will be used.'
        )
        # check the first "num_taps_dbf" channel exist in both RX/TX lists
        # needed for rising edge method in N-tap SweepSAR case!
        list_txrx = set(list_rx).intersection(list_tx)
        for cc in range(1, num_taps_dbf + 1):
            if cc not in list_txrx:
                raise RuntimeError(
                    f'Channel # {cc} is missing in either Rx/Tx lists!'
                )
        # get the first "num_taps_dbf" RX channel numbers versus antenna
        # beams per RX pol
        rx_beam_tags = {
            f'RX{rx:02d}{txrx_pol[1]}' for rx in range(1, num_taps_dbf + 1)}
        # get all RX beam tags used to form TX BMF pattern per TX pol
        # number of beams with TX pol used on TX side
        num_tx_beams = ant.num_beams(txrx_pol[0])
        tx_beam_tags = {
            f'RX{rx:02d}{txrx_pol[0]}' for rx in range(1, num_tx_beams)
        }
        # beam number of the last RX beam whose peak loc will set upper limit
        # of EL angle of rising edge
        beam_num_peak = 2

    else:  # single-channel SAR case
        is_sweepsar = False
        logger.info('It is a Single-channel SAR echo. The antenna beams # '
                    f'{beam_num} will be used.')
        rx_beam_tags = {f'RX{beam_num:02d}{txrx_pol[1]}'}
        tx_beam_tags = {f'RX{beam_num:02d}{txrx_pol[0]}'}
        # beam number of the last RX beam whose peak loc will set upper limit
        # of EL angle of rising edge
        beam_num_peak = beam_num

    logger.info(
        f'Expected RX beam tags for forming RX DBF ->\n {rx_beam_tags}')
    if not rx_beam_tags.issubset(ant.rx_beams):
        raise RuntimeError(
            f'Missing one or more of beams {rx_beam_tags} in antenna file '
            f'needed to form rising edge of {num_taps_dbf}-Tap RX DBF pattern!'
        )
    logger.info(f'Expected RX beam tags to form TX BMF ->\n {tx_beam_tags}')
    if not tx_beam_tags.issubset(ant.rx_beams):
        raise RuntimeError(
            f'Missing one or more of beams {tx_beam_tags} in antenna file '
            'required to form TX BMF pattern!'
        )
    # Parse El-cut info of Rx and TX individual beams
    rx_beams_el = ant.el_cut_all(txrx_pol[1])
    if txrx_pol[0] != txrx_pol[1]:
        tx_beams_el = ant.el_cut_all(txrx_pol[0])
        # RX and TX beams must have the same EL angle coverage/array for
        # patterns of X-Pol product
        is_tx_equal_rx_el = (np.isclose(rx_beams_el.angle[0],
                                        tx_beams_el.angle[0]) and
                             np.isclose(rx_beams_el.angle[-1],
                                        tx_beams_el.angle[-1]) and
                             rx_beams_el.angle.size == tx_beams_el.angle.size)
        if not is_tx_equal_rx_el:
            raise NotImplementedError(
                'Does not support unequal EL angles for Tx and Rx patterns of'
                ' x-pol product!'
            )
    else:
        tx_beams_el = rx_beams_el

    # Find the peak location of beam # "beam_num_peak" This location defines
    # upper limit for rising edge of the final 2-way pattern in EL dirction.
    idx_el_peak = abs(rx_beams_el.copol_pattern[beam_num_peak - 1]).argmax()
    el_peak = rx_beams_el.angle[idx_el_peak]
    logger.info('EL peak location used for upper limit of EL -> '
                f'{np.rad2deg(el_peak):.2f} (deg)')
    # Set the max EL angle for antenna patterns and the echo data.
    # For single-channel, the max EL angle shall be below peak location.
    # Fo the sweepSAR case, the first peak (ripple) of the DBFed pattern
    # occurs somewhere between the peak of beam # 2 and # 3. It may or
    # may not be within desired leading edge in flight. Thus, the peak
    # of second beam along with a marigin is used to defined max EL angle
    # for antenna in sweepsar SAR case.
    # In either case, make sure EL angle coverage for echo is smaller than
    # that of antenna, and both are within rising edge below the first peak.
    if is_sweepsar:
        el_ant_max = el_peak + el_margin
        el_echo_max = el_peak
    else:  # single-channel SAR
        el_ant_max = el_peak
        el_echo_max = el_peak - el_margin

    # get mean azimuth angle between TX and RX EL-cut beams
    az_ang_cut = 0.5 * (rx_beams_el.cut_angle + tx_beams_el.cut_angle)
    logger.info('Averaged azimuth angle for EL cuts TX/RX -> '
                f'{np.rad2deg(az_ang_cut):.2f} (deg)')

    # Get Pulse/azimuth time and ref epoch
    ref_epoch_echo, aztime_echo = raw.getPulseTimes(freq_band, txrx_pol[0])

    # get orbit object and update its ref epoch if necessary to match
    # that of L0B echo for both internal and external cases.
    if orbit is None:
        logger.info('Orbit data stored in L0B will be used.')
        orbit = raw.getOrbit()
    else:  # use external orbit data
        logger.info('External orbit data will be used.')
    if orbit.reference_epoch != ref_epoch_echo:
        logger.warning(
            'Reference epoch of orbit, '
            f'{orbit.reference_epoch.isoformat()}, and that'
            f' of L0B pulse time, {ref_epoch_echo.isoformat()}, '
            'does not match!'
        )
        logger.warning('Reference epoch of L0B pulsetime will be used!')
        orbit = orbit.copy()
        orbit.update_reference_epoch(ref_epoch_echo)

    # get attitude object and update its ref epoch if necessary to match
    # that of L0B echo for both internal and external cases.
    if attitude is None:
        logger.info('Attitude data stored in L0B will be used.')
        attitude = raw.getAttitude()
    else:  # use external attitude data
        logger.info('External attitude data will be used.')
    if attitude.reference_epoch != ref_epoch_echo:
        logger.warning(
            'Reference epoch of attitude, '
            f'{attitude.reference_epoch.isoformat()}, and that'
            f' of L0B pulse time, {ref_epoch_echo.isoformat()}, '
            'does not match!'
        )
        logger.warning('Reference epoch of L0B pulsetime will be used!')
        attitude = attitude.copy()
        attitude.update_reference_epoch(ref_epoch_echo)

    # build DEM object if not provided
    if dem_interp is None:
        dem_interp = DEMInterpolator()
    else:
        # precompute mean DEM needed for antenna geometry
        if dem_interp.have_raster and not dem_interp.have_stats:
            dem_interp.compute_min_max_mean_height()
    logger.info(
        f'Ref height of DEM object -> {dem_interp.ref_height:.3f} (m)')

    # Get slant range
    sr_linspace = raw.getRanges(freq_band, txrx_pol[0])
    sr_spacing = sr_linspace.spacing
    sr_start = sr_linspace.first
    logger.info('slant range (start, spacing) -> '
                f'({sr_start:.3f}, {sr_spacing:.3f}) (m, m)')

    # parse valid subswath index for all range lines used later
    valid_sbsw_all = raw.getSubSwaths(freq_band, txrx_pol[0])

    # Use TX Calibration components HPA and BYPASS to build complex TX
    # weighting defined by the averaged ratio of the HPA CAL to BYPASS CAL.
    # This approach includes both mag/phase of the key parts of TX path.
    # Alternatively, one may use the TX phases covering the phase of
    # entire TX path and ignoring the magnitude by assuming the TX channels
    # are pretty-well balanced magnitude-wise due to HPAs operating in
    # saturation mode.
    # Note that phase imbalance is key driving factor in forming TX pattern.
    # However, the channel imbalances only affect overlapped regions of
    # beams (part of leading edge).
    # TX patterns.
    if is_sweepsar:
        # Last TX beam needed to form TX BMF pattern valid within rising edge
        beam_num_stop_tx = min(beam_num_peak + num_taps_dbf,
                               ant.num_beams(txrx_pol[0]))
        # Last beam number needed for forming RX DBF pattern within rising edge
        beam_num_stop_rx = num_taps_dbf
        logger.info('The last beam # used for (Tx BMF, RX DBF) -> '
                    f'({beam_num_stop_tx}, {beam_num_stop_rx})')

        # Get HPA/BYP Cal ratio for desired TX beams over all range lines
        # form TX TRM to be used for TX weights
        # One can use TX phase by adding it to the "tx_trm_info"
        tx_trm_info = TxTrmInfo(
            raw.getPulseTimes(freq_band, txrx_pol[0])[1],
            np.arange(1, beam_num_stop_tx + 1),
            raw.getChirpCorrelator(freq_band, txrx_pol[0])[..., 1],
            raw.getCalType(freq_band, txrx_pol[0])
            )
        # get TX weights to be used to form Tx pattern in EL.
        tx_cal_ratio = compute_transmit_pattern_weights(
            tx_trm_info, norm=True)

    # generate rangeline slices
    rgl_slices = _rgl_slice_gen(num_rgls, num_azimuth_block, num_rgl_block)

    # build ElPatternEst object used for 2-way power pattern est from echo
    # within only rising edge with 3rd-order to be used in roll cost function
    el_pat_est = ElPatternEst(sr_start, orbit, dem_interp=dem_interp,
                              polyfit_deg=pf_deg)

    # check status of the SNR-based weighting applied to the cost function
    if apply_weight:
        logger.info('SNR-based weights will be applied to the cost function.')
    else:  # no weighting
        logger.warning('No weighting will be applied to the cost function!')
        pf_wgt = Poly1d([1.0])

    # initialize return values for containers
    mask_valid = np.ones(num_azimuth_block, dtype=bool)
    cvg_flag = np.zeros(num_azimuth_block, dtype=bool)
    el_ofs = np.zeros(num_azimuth_block, dtype='f4')
    sr_first_last = np.zeros((num_azimuth_block, 2), dtype='f4')
    el_first_last = np.zeros((num_azimuth_block, 2), dtype='f4')
    lka_first_last = np.zeros((num_azimuth_block, 2), dtype='f4')
    az_datetime = []
    pf_ant_all = []
    pf_echo_all = []
    pf_wgt_all = []
    mask_valid_rgb = True

    # loop over all azimuth blocks
    for nn, s_rgl in enumerate(rgl_slices):
        n_azblk = nn + 1
        logger.info(
            f'(start, stop) range lines for azimuth block # {n_azblk} -> '
            f'({s_rgl.start}, {s_rgl.stop})'
        )
        # mid azimuth time of the block
        azt_mid = aztime_echo[s_rgl].mean()

        # get S/C position/veclity and attitude at mid azimuth time
        pos_ecef_mid, vel_ecef_mid = orbit.interpolate(azt_mid)
        quat_ant2ecef_mid = attitude.interpolate(azt_mid)

        # compute approximate first(starting) look angle corresponds to
        # starting slant range at mid azimuth time of each azimuth block
        lka_first, _ = look_inc_ang_from_slant_range(sr_start, orbit, azt_mid,
                                                     dem_interp)
        # Get approximate first (starting) antenna EL angle per estimated
        # Mechanical boresight (MB) in radians
        mb_ang = compute_mb_angle(pos_ecef_mid, quat_ant2ecef_mid)
        el_echo_first = lka_first - mb_ang
        logger.info(
            'Estimated mechanical boresight angle -> '
            f'{np.rad2deg(mb_ang):.2f} (deg)'
        )
        # check if the antenna starting EL angle is equal or less than the
        # "el_echo_first". If not issue an error due to lack of enough
        # antenna pattern coverage in antenna file
        if rx_beams_el.angle[0] > el_echo_first:
            raise RuntimeError('Not enough EL angle coverage in "ant" '
                               'at rising edge of pattern!')

        # get [start, stop) EL angle indices for antenna pattern
        # make sure to include el margin on each end to go slightly
        # beyond EL coverage of echo data!
        el_ant_min = el_echo_first - el_margin
        idx_el_ant_str = bisect.bisect_left(rx_beams_el.angle, el_ant_min)
        idx_el_ant_stp = bisect.bisect_right(rx_beams_el.angle, el_ant_max)
        el_ant_slice = slice(idx_el_ant_str, idx_el_ant_stp)
        logger.info(
            f'(min, max) antenna EL coverage -> ({np.rad2deg(el_ant_min):.2f}'
            f', {np.rad2deg(el_ant_max):.2f}) (deg, deg)'
        )

        # form rising-edge two-way antenna power pattern (dB) as a
        # function of EL (rad) within rising edge limit
        # [el_ant_min, el_ant_max]
        ant_el = rx_beams_el.angle[el_ant_slice]
        if is_sweepsar:
            # get multi-channel complex antenna pattern for both TX and RX side
            # within [start, stop) EL angle "el_ant_slice" for selected beams.
            ant_pat_rx = rx_beams_el.copol_pattern[:beam_num_stop_rx,
                                                   el_ant_slice]
            ant_pat_tx = tx_beams_el.copol_pattern[:beam_num_stop_tx,
                                                   el_ant_slice]

            # Get block-averaged TX complex weighting to build TX BMF pattern
            tx_wgt = np.nanmean(tx_cal_ratio[s_rgl], axis=0)

            # form 2-way power pattern (dB) of rising edge only
            antpat2w = _form_ant2way_sweepsar(
                ant_pat_tx, ant_pat_rx, ant_el, tx_wgt, dbf_pow_norm)

        else:  # single-channel SAR
            # get antenna pattern for a specific beam number on both TX/RX side
            ant_pat_rx = rx_beams_el.copol_pattern[beam_num - 1, el_ant_slice]
            ant_pat_tx = tx_beams_el.copol_pattern[beam_num - 1, el_ant_slice]
            # form two-way power pattern of rising edge only
            antpat2w = 2 * pow2db(abs(ant_pat_rx * ant_pat_tx))

        # peak normalized 2-way power pattern
        antpat2w -= antpat2w.max()

        # get exact look angle (off-nadir) for antenna at mid azimuth block
        # perform 3rd-order polyfit of antenna power pattern (dB) as a function
        # of look angle in radians
        ant_lka = ela_to_offnadir(ant_el, quat_ant2ecef_mid, pos_ecef_mid,
                                  az_cut=az_ang_cut, frame=ant.frame)
        pf_coef_ant = np.polyfit(ant_lka, antpat2w, deg=pf_deg)
        pf_ant = Poly1d(pf_coef_ant[::-1])

        # convert antenna EL angles to slant range and then compute weighting
        # factor based on relative SNR used in weighing cost function of
        # rising edge method. SNR ~ (pow_pat2w / sr**3) for random scene.
        ant_sr, _, _ = ant2rgdop(
            ant_el, az_ang_cut, pos_ecef_mid, vel_ecef_mid,
            quat_ant2ecef_mid, wavelength, dem_interp
        )
        logger.info(
            f'(min, max) antenna slant range coverage -> ({ant_sr[0]:.3f}'
            f', {ant_sr[-1]:.3f}) (m, m)'
        )
        if apply_weight:
            # cost function weights based on SNR in (dB)
            cf_wgt = antpat2w - 3 * pow2db(ant_sr)
            # peak normalized
            cf_wgt -= cf_wgt.max()
            # form 3rd-order poly1d for weighting of cost function, relative
            # power in (dB) as a function of look angle in (rad)
            pf_coef_wgt = np.polyfit(ant_lka, cf_wgt, deg=pf_deg)
            pf_wgt = Poly1d(pf_coef_wgt[::-1])

        # convert the (first, last) el angle (within first EL peak location)
        # to slant range and range bin per azimuth block
        sr_last, _, _ = ant2rgdop(
            el_echo_max, az_ang_cut, pos_ecef_mid, vel_ecef_mid,
            quat_ant2ecef_mid, wavelength, dem_interp
        )

        # update last slant range with chirp duration to compensate
        # for range bins of 'valid' mode in rangecomp of "ElPatternEst()"
        sr_last += sr_chirp
        # convert to range bin
        rgb_last = round((sr_last - sr_start) / sr_spacing)
        logger.info(f'(start, stop) range bins of echo -> (0, {rgb_last})')
        if rgb_last > num_rgbs:
            raise RuntimeError('Not enough range bins in echo data!')
        # get echo for a subset of range bins for each azimuth block
        echo = raw_dset[s_rgl, :rgb_last]
        # replace bad values in place with some random with proper std given
        # homogenous random scene
        replace_badval_echo(echo)
        # get 2-way power pattern from the echo
        p2w_echo, sr_echo, lka_echo, inc_echo, pf_echo = \
            el_pat_est.power_pattern_2way(
                echo, sr_spacing, chirp_rate, chirp_dur, azt_mid,
                size_avg=size_rgb_avg
            )
        logger.info(
            f'(min, max) echo slant range coverage -> ({sr_echo[0]:.3f}'
            f', {sr_echo[-1]:.3f}) (m, m)'
        )
        logger.info(
            '(min, max) echo look angle coverage -> '
            f'({np.rad2deg(lka_echo[0]):.2f}, {np.rad2deg(lka_echo[-1]):.2f})'
            ' (deg, deg)'
        )

        # perform roll offset estimation
        roll_ofs, roll_fval, roll_flag, roll_iter = \
            roll_angle_offset_from_edge(
                pf_echo, pf_ant, lka_echo[0], lka_echo[-1], el_res,
                pf_wgt
            )
        logger.info('Estimated roll angle offset -> '
                    f'{rad2mdeg(roll_ofs):.1f} (mdeg)')
        # Plot poly-fitted echo power pattern v.s. antenna one w/ & w/o roll
        # angle correction per azimuth block.
        az_dtm = ref_epoch_echo + TimeDelta(azt_mid)
        if plot:
            plt_name = (f'{prefix}_Freq{freq_band}_Pol{txrx_pol}_'
                        f'AzBlock{n_azblk}.png')
            plt_filename = os.path.join(out_path, plt_name)
            _plot_echo_vs_ant_pat(
                pf_echo, pf_ant, (lka_echo[0], lka_echo[-1]),
                roll_ofs, azt_mid, ref_epoch_echo.isoformat(), plt_filename
                )

        # If PRF is constant then find out if rising edge region is valid. That
        # is, whether or not it overlaps with TX gap!
        # Checking only the first and last range lines assuming that the
        # current azimuth block is small compared to the rate with which the
        # locations of subswaths vary within the swath.
        if not dithered:  # const PRF
            mask_valid_rgb = _is_rising_edge_valid(
                (0, rgb_last), valid_sbsw_all[:, s_rgl.start, :])
            mask_valid_rgb &= _is_rising_edge_valid(
                (0, rgb_last), valid_sbsw_all[:, s_rgl.stop - 1, :])

        # Convert look angles of echo to antenna EL angles via interpolation
        # of exisiting (antenna EL -> off-nadir) arrays.
        # Note that antenn EL/look angle has a wider coverage than that of echo
        # and it is monotonically sorted! Thus, "fill_value" is unnecessary.
        # Define a function (interpolation kernel) to evaluate EL angle (rad)
        # from Look angle (rad)
        func_lka2el = interp1d(
            ant_lka, ant_el, kind=interp_method_el, copy=False,
            fill_value='extrapolate', assume_sorted=True
            )

        # fill up the output containers per azimuth block
        el_ofs[nn] = roll_ofs
        az_datetime.append(az_dtm)
        # True EL angle is roll offset added to off-nadir angle in
        # antenna frame. Given "el_fl" is obtained from slant ranges in
        # echo domain, then roll offset shall be subtratced from the EL
        # angle to get the true/corrected EL angle for the respective
        # slant range at leading edge.
        el_first_last[nn] = func_lka2el((lka_echo[0] - roll_ofs,
                                         lka_echo[-1] - roll_ofs))
        sr_first_last[nn] = (sr_echo[0], sr_echo[-1])
        lka_first_last[nn] = (lka_echo[0], lka_echo[-1])
        mask_valid[nn] = mask_valid_rgb
        cvg_flag[nn] = roll_fval
        pf_echo_all.append(pf_echo)
        pf_ant_all.append(pf_ant)
        pf_wgt_all.append(pf_wgt)

    return (el_ofs, np.asarray(az_datetime), el_first_last, sr_first_last,
            lka_first_last, mask_valid, cvg_flag, np.asarray(pf_echo_all),
            np.asarray(pf_ant_all), np.asarray(pf_wgt_all))


def pow2db(x):
    return 10 * np.log10(x)


def rad2mdeg(x):
    return 1000 * np.rad2deg(x)


def nadir_unit_vector(pos, ellips=Ellipsoid()):
    """Get nadir unit vector in ECEF(X,Y,Z) from spacecraft position.

    Parameters
    ----------
    pos : np.ndarray(float)
        3-D position vector of spacecraft in ECEF
    ellips : isce3.core.Ellipsoid, default=WGS84

    Returns
    -------
    np.ndarray(float)
        Local geodetic nadir unit vector in ECEF

    """
    lon, lat, _ = ellips.xyz_to_lon_lat(pos)
    return -ellips.n_vector(lon, lat)


def compute_mb_angle(pos, quat, ellips=Ellipsoid()):
    """
    Compute mechanical boresight (MB) angle with respect to the geodetic nadir
    vector from both spacecraft position and antenna-to-spacecraft unit
    quaternions.

    Parameters
    ----------
    pos : np.ndarray(float)
        3-D position vector of spacecraft in ECEF
    quat : isce3.core.Quaternion
        Represent unit quaternions for conversion from antenna/radar coordinate
        to spacecraft coordinate.
    ellips : isce3.core.Ellipsoid, default=WGS84

    Returns
    -------
    float :
        Mechanical boresight angle in radians

    """
    boresight_ecef = quat.rotate([0, 0, 1])
    nadir_ecef = nadir_unit_vector(pos, ellips)

    return np.arccos(nadir_ecef.dot(boresight_ecef))


def ela_to_offnadir(el, quat, pos,  az_cut=0.0, frame=Frame(),
                    ellips=Ellipsoid()):
    """Convert EL-cut angle(s) to off-nadir angle(s).

    Parameters
    ----------
    el : float or array of float
        antenn EL angles defined by antenna "frame" in radians.
    quat : isce3.core.Quaternion
        Represent unit quaternions for conversion from antenna/radar coordinate
        to spacecraft coordinate.
    pos : np.ndarray(float)
        3-D position vector of spacecraft in ECEF
    az_cut : float, default=0.0
        Azimuth angle in radians at which EL-cut angles are obtained.
        Azimuth angle is defined per antenna "frame".
    frame : isce3.antenna.Frame, default='EL-AND-AZ'
    ellips : isce3.core.Ellipsoid, default=WGS84

    Returns
    -------
    float or np.ndarray(float)
        Off-nadir angle(s) in radians

    """

    # get nadir vector in ECEF
    nadir_ecef = nadir_unit_vector(pos, ellips)
    # get a list of pointing unit verctors in antenna XYZ frame
    pnt_ant_vec = frame.sph2cart(el, az_cut)
    if isinstance(el, Number):
        pnt_ant_vec = [pnt_ant_vec]
    # get off-nadir angle from dot product of nadir and pointing vector
    off_nadir = []
    for pnt_ant in pnt_ant_vec:
        pnt_ecef = quat.rotate(pnt_ant)
        off_nadir.append(np.arccos(nadir_ecef.dot(pnt_ecef)))

    if len(off_nadir) == 1:
        return off_nadir[0]
    return np.asarray(off_nadir)


def replace_badval_echo(echo, rnd_seed=10):
    """
    Replace bad values such as NaN or zero values by a Gaussian random noise
    whose STD deteremined by std of non bad values per range line.
    The input array is modified in place.

    Parameters
    ----------
    echo : np.ndarray(float)
        The echo is modified in place if it contains bad values.
    rnd_seed : int, default=10
        seed number for random generator

    Notes
    -----
    Invalid values, NaNs or zeros, are replaced by random values given each
    range line contains homogenous clutter.

    """
    const_iq = 1. / np.sqrt(2.)
    # seed number for Gaussian noise random generator
    # to replace bad values of echo if any.
    rnd_gen = np.random.RandomState(rnd_seed)
    # get number of range lines and range bins
    nrgl, _ = echo.shape
    # replace bad values (NaN) with Gaussian noise with std determined
    # by non-nan range bins per range line
    for line in range(nrgl):
        mask_bad = np.isnan(echo[line]) | np.isclose(echo[line], 0)
        num_bad = mask_bad.sum()
        if num_bad > 0:
            std_iq = const_iq * np.std(echo[line, ~mask_bad])
            echo[line, mask_bad] = std_iq * (rnd_gen.randn(num_bad) +
                                             1j*rnd_gen.randn(num_bad))


# list of private helper functions below this line

def _form_ant2way_sweepsar(ant_pat_tx, ant_pat_rx, ant_el, tx_wgt,
                           pow_norm=True):
    """
    Form 2-way peak-normalized antenna power pattern in EL direction
    for multi-beam SweepSAR antenna.

    Parameters
    ----------
    ant_pat_tx : np.ndarray(complex)
        2-D Multi-channel complex antenna patterns on TX side.
        Shape is beams by angles.
    ant_pat_rx : np.ndarray(complex)
        2-D Multi-channel complex antenna patterns on RX side.
        Shape is beams by angles.
    ant_el : np.ndarray(float)
        EL angles in radians.
    tx_wgt : np.ndarray(complex)
        Complex TX weighting coeffs with size equal to number of beams.
    pow_norm : bool, default=True
        Whether or not RX DBF pattern shall be power normalized.

    Returns
    -------
    np.ndarray(float)
        2-way power pattern within specified EL angle in (dB)

    """
    # form perfect N-tap RX DBF power pattern
    rx_dbf = (abs(ant_pat_rx)**2).sum(axis=0)
    if not pow_norm:
        rx_dbf *= rx_dbf
    # form TX BMF power pattern
    tx_bmf = abs(np.matmul(tx_wgt, ant_pat_tx))**2
    # form peak-normalized two-way power pattern in dB
    ant_powpat_2way = tx_bmf * rx_dbf
    max_pow = ant_powpat_2way.max()
    if not (max_pow > 0.0):
        raise RuntimeError('2-way Power Pattern is zero for SweepSAR!')
    ant_powpat_2way /= max_pow
    return pow2db(ant_powpat_2way)


def _rgl_slice_gen(num_rgls, num_azimuth_block, num_rgl_block):
    """Generates pair number and respective range line slice.

    Parameters
    ----------
    num_rgls : int
        Total number of range lines
    num_azimuth_block : int
        Number of azimuth blocks.
    num_rgl_block : int
        Number of range lines per azimuth block

    Yields
    ------
    slice
        Range line slices

    """
    i_start = 0
    i_stop = 0
    for cc in range(1, num_azimuth_block + 1):
        if cc == num_azimuth_block:
            i_stop = num_rgls
        else:
            i_stop += num_rgl_block
        yield slice(i_start, i_stop)
        i_start += num_rgl_block


def _is_rising_edge_valid(rgb_fl, rgb_valid_sbsw):
    """Check whether edge is valid or not.

    An edge is considered valid if it does not overlap the location of
    any Tx gaps in the swath.

    Parameters
    ----------
    rgb_fl : Tuple[int, int]
        (first, last) range bins defining the rising edge region
    rgb_valid_sbsw : np.ndarray[int]
        2-D array-like integers for valid range bins of a specific range line

    Returns
    -------
    bool

    """
    rgb_edge_region = set(rgb_fl)
    # check if edge overlaps with TX gaps
    flag = False
    for start_stop in rgb_valid_sbsw:
        flag |= rgb_edge_region.issubset(range(*start_stop))
    return flag


def _plot_echo_vs_ant_pat(pf_echo, pf_ant, lka_fl, roll_ofs,
                          az_time, epoch, filename):
    """Plot poly-fitted echo v.s. antenna w/ and w/o roll angle correction.

    Parameters
    ----------
    pf_echo : isce3.core.Poly1d
    pf_ant : isce3.core.Poly1d
    lka_fl : tuple(float, float)
        [Fist, Last] look angles in (rad)
    roll_ofs : float
        Roll angle ofset in (rad)
    az_time : float
        Seconds since "epoch" related to mid AZ time of the
        block whose EL rising-edge to be plotted.
    epoch : str
        Reference epoch UTC time.
    filename : str
        Filename of the plot with ext "png".

    """
    el_res_deg = 0.01
    num_lka = round((lka_fl[1] - lka_fl[0]) / np.deg2rad(el_res_deg)) + 1
    lka_vec = np.linspace(*lka_fl, num=num_lka)
    echo_pow = pf_echo.eval(lka_vec)
    ant_pow = pf_ant.eval(lka_vec)
    ant_pow_cor = pf_ant.eval(lka_vec + roll_ofs)
    lka_vec_deg = np.rad2deg(lka_vec)

    plt.figure(figsize=(8, 7))
    plt.plot(lka_vec_deg, echo_pow, 'b',
             lka_vec_deg, ant_pow, 'r-.',
             lka_vec_deg, ant_pow_cor, 'g--', linewidth=2)
    plt.grid(True)
    plt.xlabel('Look Angles (deg)')
    plt.ylabel('Relative Power (dB)')
    plt.legend(['ECHO', 'ANT',
                f'EL-Adj={rad2mdeg(roll_ofs):.0f}(mdeg)'],
               loc='best')
    plt.title(
        f'Echo v.s. Antenna Rising Edge w/ & w/o EL Adjustment\n'
        f'@ AZ-Time={az_time:.3f} sec\nsince {epoch}'
        )
    plt.savefig(filename)
    plt.close()
