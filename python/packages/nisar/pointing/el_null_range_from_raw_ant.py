"""
Function to get the null locations in EL as a function of range from
multi-channel Raw echoes and in EL angles from multi-beam antenna patterns.
"""
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from isce3.antenna import ElNullRangeEst, ant2rgdop
from isce3.geometry import DEMInterpolator
from isce3.core import speed_of_light
from nisar.log import set_logger


@dataclass(frozen=True)
class AntElPair:
    """
    Immutable struct for a pair of adjacent antenna
    elevation-cut (EL) patterns with some overlap.

    Atrributes
    ----------
    pat_left : np.ndarray(complex)
        A vector of complex EL pattern for left beam.
    pat_right : np.ndarray(complex)
        A vector of complex EL pattern for right beam.
        The same size as `pat_left`.
    el_ang : np.ndarray(float)
        A vector of uniformly-spaced EL angles in radians.
        The same size as `pat_*`
    az_ang_cut : float
        AZ angle in radians at which EL cuts left/right are taken.
        The averaged value between two azimuth cuts left/right will be
        stored.

    """
    pat_left: Sequence[complex]
    pat_right: Sequence[complex]
    el_ang: Sequence[float]
    az_ang_cut: float


def el_null_range_from_raw_ant(raw, ant, *, dem_interp=None,
                               freq_band='A', txrx_pol=None,
                               orbit=None, attitude=None,
                               az_block_dur=3.0,
                               apply_caltone=False, logger=None,
                               plot=False, out_path='.',
                               polyfit_deg=6):
    """
    Get estimated null locations in elevation (EL) direction as a function
    of slant range and azimuth time from multi-channel raw echo
    (at least two channels) as well as get expected null location in EL angles
    from multi-beam antenna patterns (at least two beams).

    See Ref [1]_ for algorithm and demo.

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
    txrx_pol : str, optional
        TxRx polarization such as {'HH', 'HV',...}. If not provided the first
        product under `freq_band` will be used.
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
    apply_caltone : bool, default=False
        Apply caltone coefficients to RX channels prior to Null formation.
    logger : logging.Logger, optional
        If not provided a logger with StreamHandler will be set.
    plot : bool, default=False
        If True, it will generate one PNG plot per null and per azimuith block
        to compare null echo data (measured) versus that of antenna one
        (reference) in EL.
    out_path : str, default='.'
        Ouput directory for dumping PNG files, if `plot` is True.
    polyfit_deg : int, default=6
        Polyfit degree used in poly fitting echo null power pattern in
        elevation for the sake of smoothing and null location estimation.
        The degree must be an even number equal or larger than 2.

    Returns
    -------
    np.ndarray(uint8)
        Null number with a value equal or greater than 1 and less than
        total number of RX channels. Null=1 denotes the null formed
        using the first two channels.
    np.ndarray(float32)
        Estimated Nulls locations in slant ranges (m) from Raw echo.
    np.ndarray(float32)
        Expected Nulls locations in antenna EL angles (deg) from antenna.
    np.ndarray(float32)
        Normalized power of echo null in linear scale.
        The normalized power is the ratio of null (min) power to the max power
        within null pattern.
        This value can be used as a simple quality metric of the echo null.
        The smaller the value the better/deeper the null formed by echo pairs.
        Theoretically, this value is within [0, 1].
    np.ndarray(isce3.core.DateTime)
        Azimuth UTC Date-time tag
    np.ndarray(bool)
        Null overall convergence flag
    np.ndarray(bool)
        Mask array for valid nulls.
        A null is considered invalid within a given azimuth block if it
        overlaps with a TX gap for > 50% of the azimuth extent of the block.
        In case of PRF dithering, all nulls are assumed to be valid.
    str
        TxRx polarization of the product
    float
        wavelength of the `freq_band` in (m)

    Raises
    ------
    ValueError
        For bad input parameters or non-existent polarization and/or
        frequency band.
        Azimuth block duration is smaller than the mean PRI.
        Degree of polyfit is not an even-number integer equal or larger
        than 2.
    RuntimeError
        If raw echo dataset is not 3-D (multi-channel) or antenna beam numbers
        don't match that of active RX channel numbers of Raw.
        If active RX channels are not adjacent ones.

    Notes
    -----
    `ant` shall have one RX antenna pattern per Raw echo channel. That is
    the number of beams shall be equal or larger than number of echo channels.

    To avoid possible poor estimation of null location, it is recommended to
    set azimuth block duration to a value (way) larger than 10*PRI.

    References
    ----------
    .. [1] H. Ghaemi, "Demonstration of Null Pattern Formation & EL Pointing
        Estimation over Heterogeneous Scene," JPL Report, RevB, October 2020.

    """
    # Const
    # prefix name used in PNG Null plot if requested
    null_prefix = 'EL_Null_PowPattern_Plot'

    # set logger
    if logger is None:
        logger = set_logger("ElNullRangeFromRawAnt")

    # check inputs

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
    logger.info(f'TxRx Pol -> "{txrx_pol}"')

    # Get raw dataset
    raw_dset = raw.getRawDataset(freq_band, txrx_pol)
    if raw_dset.ndim != 3:
        raise RuntimeError(
            'Expected Multi-channel Raw echo aka Diagnostic Mode #2 (DM2)!')
    num_channels, num_rgls, num_rgbs = raw_dset.shape
    logger.info('Shape of the echo data (channels, pulses, ranges) -> '
                f'({num_channels, num_rgls, num_rgbs})')

    # get mean PRF and check for dithering
    prf = raw.getNominalPRF(freq_band, txrx_pol[0])
    dithered = raw.isDithered(freq_band, txrx_pol[0])
    logger.info(f'Mean PRF -> {prf:.3f} (Hz)')
    if dithered:
        logger.warning(
            'Dithered PRF! TX-gap related valid mask will all be set to True!'
            ' Use mean PRF for number of lines in azimuth block!'
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
    if (num_rgls > 10 and num_rgl_block < 10):
        logger.warning('Azimuth block duration is smaller than "10xPRI".'
                       ' This can lead to poor null estimation!')

    # get number of azimuth blocks
    num_azimuth_block = num_rgls // num_rgl_block
    logger.info('Total number of azimuth blocks per null -> '
                f'{num_azimuth_block}')

    # get list of active RX channels
    list_rx = raw.getListOfRxTRMs(freq_band, txrx_pol)
    logger.info(f'List of active RX channels -> {list_rx}')
    # check to see if channels are adjacent in ascending order
    if np.any(np.diff(list_rx) != 1):
        raise RuntimeError(
            'RX channels are not adjacent in ascending order!')

    # check RX channel numbers versus antenna beams per a desired RX pol
    rx_beam_tags = {f'RX{rx:02d}{txrx_pol[1]}' for rx in list_rx}
    logger.info(f'Expected RX beam tags in antenna object -> {rx_beam_tags}')
    if rx_beam_tags.intersection(ant.rx_beams) != rx_beam_tags:
        raise RuntimeError(
            f'Missing one or more of {rx_beam_tags} in antenna object!')

    # number of nulls
    num_nulls = num_channels - 1
    logger.info(f'Number of nulls -> {num_nulls}')

    # build DEM object if not provided
    if dem_interp is None:
        dem_interp = DEMInterpolator()
    else:
        # precompute mean DEM needed for antenna geometry
        if dem_interp.have_raster and not dem_interp.have_stats:
            dem_interp.compute_min_max_mean_height()
    logger.info(
        f'Ref height of DEM object -> {dem_interp.ref_height:.3f} (m)')

    # Get chirp parameters
    centerfreq, samprate, chirp_rate, chirp_dur = \
        raw.getChirpParameters(freq_band,
                               txrx_pol[0])
    wavelength = speed_of_light / centerfreq
    logger.info(f'Fast-time sampling rate -> {samprate * 1e-6:.2f} (MHz)')
    logger.info(f'Chirp pulsewidth -> {chirp_dur * 1e6:.2f} (us)')
    logger.info(f'Chirp center frequency -> {centerfreq * 1e-6:.2f} (MHz)')
    logger.info(f'Wavelength -> {wavelength:.4f} (m)')

    # Get slant range
    sr_linspace = raw.getRanges(freq_band, txrx_pol[0])
    sr_spacing = sr_linspace.spacing
    sr_start = sr_linspace.first
    logger.info(
        f'slant range (start, spacing) -> ({sr_start:.3f}, {sr_spacing:.3f})')

    # Get Pulse/azimuth time and ref epoch
    ref_epoch_echo, aztime_echo = raw.getPulseTimes(freq_band, txrx_pol[0])

    # get orbit object and update its ref epoch if necessary to match
    # that of L0B echo
    if orbit is None:
        logger.info('Orbit data stored in L0B will be used.')
        orbit = raw.getOrbit()
    else:  # use external orbit data
        logger.info('External orbit data will be used.')
        if orbit.reference_epoch != ref_epoch_echo:
            logger.warning(
                'Reference epoch of external orbit, '
                f'{orbit.reference_epoch.isoformat()}, and that'
                f' of L0B pulse time, {ref_epoch_echo.isoformat()}, '
                'does not match!'
            )
            logger.warning('Reference epoch of L0B pulsetime will be used!')
            orbit = orbit.copy()
            orbit.update_reference_epoch(ref_epoch_echo)

    # get attitude object and update its ref epoch if necessary to match
    # that of L0B echo
    if attitude is None:
        logger.info('Attitude data stored in L0B will be used.')
        attitude = raw.getAttitude()
    else:  # use external attitude data
        logger.info('External attitude data will be used.')
        if attitude.reference_epoch != ref_epoch_echo:
            logger.warning(
                'Reference epoch of external attitude, '
                f'{attitude.reference_epoch.isoformat()}, and that'
                f' of L0B pulse time, {ref_epoch_echo.isoformat()}, '
                'does not match!'
            )
            logger.warning('Reference epoch of L0B pulsetime will be used!')
            attitude = attitude.copy()
            attitude.update_reference_epoch(ref_epoch_echo)

    # form EL-Null object
    logger.info(
        f'Polyfit degree used for echo null power pattern -> {polyfit_deg}')
    el_null_obj = ElNullRangeEst(wavelength, sr_spacing, chirp_rate, chirp_dur,
                                 orbit, attitude, polyfit_deg=polyfit_deg)

    # build all pairs related to active RX channels with common EL angles
    # within peak-to-peak overlapped regions. Number of pairs = number of nulls
    list_ant_pairs = form_overlap_antenna_pairs(ant, list_rx, txrx_pol[1])

    # parse valid subswath index for all range lines used later
    valid_sbsw_all = raw.getSubSwaths(freq_band, txrx_pol[0])

    # parse all caltone coeffs for all channels at once
    if apply_caltone:
        logger.info(
            'Inverse of averaged caltone coeffs will be applied to RX echoes!')
        caltone = raw.getCaltone(freq_band, txrx_pol)

    # get pair number and range line slice generator
    pairnum_rglslice = _pair_num_rgl_slice_gen(
        num_rgls, num_nulls, num_azimuth_block, num_rgl_block)

    # check if there is matplotlib package needed for plotting if requested
    if plot and plt is None:
        logger.warning('No plots due to missing package "matplotlib"!')
        plot = False

    # containers for return values
    null_num = []  # (-)
    sr_echo = []  # (m)
    el_ant = []  # (deg)
    mag_ratio = []  # (linear)
    az_datetime = []  # (isce3.core.DateTime)
    mask_valid = []  # (-)
    null_flag = []  # (-)

    # loop over all sets of range lines, one set per pair of RXs/beams
    for nn, s_rgl, n_azblk in pairnum_rglslice:
        pair_num = nn + 1
        logger.info(f'(start, stop) range lines for null # {pair_num} -> '
                    f'({s_rgl.start}, {s_rgl.stop})')
        # retrieve a pair of beams object for a specific null
        beams = list_ant_pairs[nn]
        # get a pair of echo for a subset of range lines
        echo_left = raw_dset[nn, s_rgl, :]
        echo_right = raw_dset[nn + 1, s_rgl, :]
        # if requested, apply inverse of slow-time averaged
        # caltone coeffs to echo pairs (left, right)
        if apply_caltone:
            # calibrate left RX channel
            cal_coef_avg_left = caltone[s_rgl, nn].mean()
            if abs(cal_coef_avg_left) > 0.0:
                echo_left *= (1./cal_coef_avg_left)
            # calibrate right RX channel
            cal_coef_avg_right = caltone[s_rgl, nn + 1].mean()
            if abs(cal_coef_avg_right) > 0.0:
                echo_right *= (1./cal_coef_avg_right)
            # get mid azimuth time within desired range lines in seconds
        az_time_mid = aztime_echo[s_rgl].mean()
        # If PRF is constant then find out if null estimation is within
        # valid regions that is if it overlaps with TX gap or not!
        if dithered:
            mask_valid.append(True)
        else:  # const PRF
            # per a subset of range lines get [min, max] range bins between
            # two antenna beams and compare them to range bins of validsubswath

            # get state vectors and quaternions of radar platform at mid
            # azimuth time within a subset of range lines
            pos_ecef_mid, vel_ecef_mid = orbit.interpolate(az_time_mid)
            quat_ant2ecef_mid = attitude.interpolate(az_time_mid)

            # get peak-to-peak range bins (first, last) for a pair of beams
            sr_p2p, _, _ = ant2rgdop((beams.el_ang[0], beams.el_ang[-1]),
                                     beams.az_ang_cut, pos_ecef_mid,
                                     vel_ecef_mid, quat_ant2ecef_mid,
                                     wavelength, dem_interp)

            rgb_p2p = np.int_(np.round((sr_p2p - sr_start) / sr_spacing))
            # check if the TX gap overlaps fully or partially with null zone,
            # that is center of the region defined by rgb_p2p.
            # get valid range bins of subswath at middle of a subset of
            # range lines to be used in checking null validity.
            # Assumption: TX gap location won't change faster than
            # azimuth block duration.
            rgl_mid_sub = s_rgl.start + (s_rgl.stop - s_rgl.start) // 2
            rgb_valid_sbsw = valid_sbsw_all[:, rgl_mid_sub, :]
            mask_valid.append(_is_null_valid(rgb_p2p, rgb_valid_sbsw))

        # estimate null locations in both Echo and Antenna domain
        tm_null, echo_null, ant_null, flag_null, pow_pat_null = \
            el_null_obj.genNullRangeDoppler(echo_left,
                                            echo_right,
                                            beams.pat_left,
                                            beams.pat_right,
                                            sr_start,
                                            beams.el_ang[0],
                                            beams.el_ang[1] - beams.el_ang[0],
                                            beams.az_ang_cut,
                                            az_time_mid)

        # report echo null power as quality check
        # the smaller the value the better/deeper the null!
        # In other words, the closer two nulls the better the performance!
        logger.info(
            f'Echo null power for # {pair_num} '
            f' @ {tm_null.isoformat()} -> '
            f'{_pow2db(echo_null.magnitude):.1f} (dB)'
        )
        # collect outputs
        null_num.append(pair_num)
        sr_echo.append(echo_null.slant_range)
        el_ant.append(np.rad2deg(ant_null.el_angle))
        mag_ratio.append(echo_null.magnitude)
        az_datetime.append(tm_null)
        null_flag.append(flag_null.geometry_antenna &
                         flag_null.geometry_echo &
                         flag_null.newton_solver)

        # plot null echo v.s. null ant profile if requested
        if plot:
            null_plt_name = (f'{null_prefix}_Pair{pair_num}_Freq{freq_band}'
                             f'_Pol{txrx_pol}_AzBlock{n_azblk}.png')
            null_filename = Path(out_path).joinpath(null_plt_name)
            _plot_null_pow_patterns(
                pow_pat_null, pair_num, null_filename, az_time_mid,
                ref_epoch_echo)

    # return tuple of np.ndarray
    return (np.asarray(null_num, dtype='uint8'),
            np.asarray(sr_echo, dtype='float32'),
            np.asarray(el_ant, dtype='float32'),
            np.asarray(mag_ratio, dtype='float32'),
            np.asarray(az_datetime),
            np.asarray(null_flag),
            np.asarray(mask_valid),
            txrx_pol,
            wavelength)


def form_overlap_antenna_pairs(ant, list_rx, rx_pol, is_half_p2p=False):
    """Form overlap antenna pattern pair object for all pairs used in Null.

    Get a pair of antenna patterns within the same overlap EL angles.
    The shortest required overlap region is within peak-to-peak.
    No need to assume that adjacent beams cover the same EL angles!

    Parameters
    ----------
    ant : nisar.products.readers.antenna.AntennaParser
    list_rx : list of int or array of int
        List/array of RX channel numbers
    rx_pol : RX polarization
    is_half_p2p : bool, default=False
        If True, the overlap region is defined as middle half
        of the peak-to-peak region where two beams are 50-50 overlapped and
        closer in terms of relative gain and shape w/ opposite slope.
        This is useful for relative channel imbalance estimation!

    Returns
    -------
    list of AntElPair
        One pair object per null in ascending order

    """
    list_pairs = []
    for n_rx in list_rx[:-1]:
        beam_left = ant.el_cut(n_rx, rx_pol)
        beam_right = ant.el_cut(n_rx + 1, rx_pol)

        el_first = beam_left.angle[abs(beam_left.copol_pattern).argmax()]
        el_last = beam_right.angle[abs(beam_right.copol_pattern).argmax()]

        if is_half_p2p:
            el_dur_quart = 0.25 * (el_last - el_first)
            el_first += el_dur_quart
            el_last -= el_dur_quart

        el_spacing = 0.5 * (np.diff(beam_left.angle[:2])[0] +
                            np.diff(beam_right.angle[:2])[0])
        az_ang_avg = 0.5 * (beam_left.cut_angle + beam_right.cut_angle)

        num_ang = int((el_last - el_first) / el_spacing) + 1
        el_ang = np.linspace(el_first, el_last, num_ang)
        pat_left = np.interp(el_ang, beam_left.angle, beam_left.copol_pattern,
                             left=0.0, right=0.0)
        pat_right = np.interp(el_ang, beam_right.angle,
                              beam_right.copol_pattern, left=0.0, right=0.0)
        list_pairs.append(AntElPair(pat_left, pat_right, el_ang, az_ang_avg))
    return list_pairs

# list of private helper functions below this line


def _pair_num_rgl_slice_gen(num_rgls, num_nulls, num_azimuth_block,
                            num_rgl_block):
    """Generates pair number and respective range line slice.

    Parameters
    ----------
    num_rgls : int
        Total number of range lines
    num_nulls : int
        Total number of nulls or pair of beams/RXs
    num_azimuth_block : int
        Number of azimuth blocks.
    num_rgl_block : int
        Number of range lines per azimuth block

    Yields
    ------
    int
        Pair number starting from 0
    slice
        Range line slices
    int
        AZ block number starting from 1

    """
    i_start = 0
    i_stop = 0
    for cc in range(1, num_azimuth_block + 1):
        if cc == num_azimuth_block:
            i_stop = num_rgls
        else:
            i_stop += num_rgl_block
        for nn in range(num_nulls):
            yield nn, slice(i_start, i_stop), cc
        i_start += num_rgl_block


def _is_null_valid(rgb_p2p, rgb_valid_sbsw):
    """Check wehther null is valid or not.

    Parameters
    ----------
    rgb_p2p : list[int, int]
        (first, last) range bins of peaks of a pair of adjacent beams
    rgb_valid_sbsw : np.ndarray[np.ndarray[int]]
        2-D array-like integers for valid range bins of a specific range line

    Returns
    -------
    bool

    """
    # A DIVISOR to define left/right margins to be excluded.
    # an integer equal or greater than 4!
    # Note that the current requirement for EL pointing error is around 250
    # mdeg which is one-quarter of p2p EL coverage (~ 1 deg) for NISAR.
    margin_divisor = 6
    # Null region is defined two-third of p2p centered at mid region
    rgb_margin = (rgb_p2p[1] - rgb_p2p[0]) // margin_divisor
    rgb_null_region = {rgb_p2p[0] + rgb_margin, rgb_p2p[1] - rgb_margin}
    # check if null overlaps with TX gap
    flag = False
    for start_stop in rgb_valid_sbsw:
        flag |= rgb_null_region.issubset(range(*start_stop))
    return flag


def _pow2db(p):
    """Linear power to dB"""
    return 10 * np.log10(p)


def _plot_null_pow_patterns(pow_pat_null, null_num, null_file, az_time,
                            epoch):
    """Plot null power patterns for echo and antenna.

    Parameters
    ----------
    pow_pat_null : isce3.antenna.NullPowPatterns
    null_num : int
        Null number.
    null_file : str
        Filename of null patterns with ext "png".
    az_time : float
        Seconds since "epoch" related to mid AZ time of the
        block whose null to be plotted.
    epoch : str
        Reference epoch UTC time.

    """
    el_deg = np.rad2deg(pow_pat_null.el)
    # set (min, max) and el spacing for x-axis ticks
    min_el = round(el_deg.min(), ndigits=1)
    max_el = round(el_deg.max(), ndigits=1)
    d_el = 0.1

    # get the min loc from antenna to be used as a reference Null location!
    idx_min = np.nanargmin(pow_pat_null.ant)
    pow_ant = _pow2db(pow_pat_null.ant)

    plt.figure(figsize=(8, 6))
    plt.plot(el_deg, pow_ant, 'b--',
             el_deg, _pow2db(pow_pat_null.echo), 'r',
             linewidth=2)
    plt.axvline(x=el_deg[idx_min], color='g', linestyle='-.', linewidth=2)
    plt.legend(['ANT', 'ECHO', 'Ref Null Loc'], loc='best')
    plt.xticks(ticks=np.arange(min_el, max_el + d_el, d_el))
    plt.xlim([min_el, max_el])
    plt.xlabel('Elevation (deg)')
    plt.ylabel('Norm. Magnitude (dB)')
    plt.title(f'EL Null Power Pattern, Echo v.s. Ant \n'
              f'for Null # {null_num} @ AZ-Time={az_time:.3f} sec\n'
              f'since {epoch}')
    plt.grid(True)
    plt.savefig(null_file)
    plt.close()
