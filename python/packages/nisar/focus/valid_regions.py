#!python3
import enum
import logging
import isce3
import numpy as np
from nisar.mixed_mode import find_overlapping_channel

log = logging.getLogger("focus")


class _PolBit(enum.IntEnum):
    """
    Bit index assigned to each polarization channel in the valid-data mask
    written by save_valid_data_mask.
    """
    HH = 8
    HV = 9
    VH = 10
    VV = 11
    LH = 12
    LV = 13
    RH = 14
    RV = 15


class PolValidMask(enum.IntFlag):
    """
    Bitmask value identifying which polarization channel(s) contributed a
    valid pixel to the mask written by save_valid_data_mask.
    """
    HH = 1 << _PolBit.HH
    HV = 1 << _PolBit.HV
    VH = 1 << _PolBit.VH
    VV = 1 << _PolBit.VV
    LH = 1 << _PolBit.LH
    LV = 1 << _PolBit.LV
    RH = 1 << _PolBit.RH
    RV = 1 << _PolBit.RV


def get_raw_sub_swath_bboxes(rawlist, out_chan, orbit, num_ignore=25,
                             min_segment_length=2000, max_pulse_gap=1,
                             max_observation_gap=0.002,
                             use_rx_pulse_mask=False):
    """
    Determine bounding boxes of raw data sub-swaths for each observation,
    matched to the requested output channel, and force azimuth continuity
    between observations that are seamless.

    Parameters
    ----------
    rawlist : list[nisar.products.readers.Raw.Raw]
        List of raw data files (observations) that will be processed.
    out_chan : nisar.mixed_mode.PolChannel
        Desired channel to process (will be matched with available raw data
        using mixed-mode logic).
    orbit : isce3.core.Orbit
        Trajectory of antenna phase center.  Used only for its reference
        epoch, to which raw data time stamps are converted.
    num_ignore : int, optional
        Number of pulses to ignore when calculating the valid data region at the
        end of a fixed-PRF observation.  This is relevant when a fixed-PRF
        observation is immediately followed by a dithered observation, as the
        dithered pulses in the air will overlap the last few receive windows of
        the fixed-PRF one.
    min_segment_length : int, optional
        Segments with fewer than this number of consecutive valid pulses
        will not be returned.  This helps avoid unnecessary bookkeeping when
        lots of missing pulses are sprinkled throughout an observation. Must
        be >= 2 pulses.
    max_pulse_gap : int, optional
        Segments (each of which must be at least min_segment_length pulses)
        separated by max_pulse_gap or fewer invalid pulses will be joined
        together.  This provides an easy way to ignore isolated gaps of a
        single invlid pulse, for example.  Set to 0 to disable merging.
    max_observation_gap : float, optional
        Max allowed time (in seconds) between the last pulse of one observation
        and the first pulse of the following observation for the two to be
        considered seamless.  Larger raw data gaps may result in a synthetic
        aperture being marked invalid in the RSLC.
    use_rx_pulse_mask : bool, optional
        When True read the pulseHasValidSamples metadata to help determine
        which pulses are valid for the receive polarization in `out_chan`.
        Otherwise only the validSamplesSubSwath metadata will be used, which
        is not specific to the receive polarization.

    Returns
    -------
    raw_bbox_lists : list[list[isce3.focus.RadarBoundingBox]]
        Bounding boxes of all subswaths for each raw data file/observation.
        Azimuth times are specified in seconds relative to the orbit
        reference epoch.
    chirp_durations : list[float]
        Duration of transmit chirp for each raw data file, in seconds.
    """
    # Need raw files sorted in time so we can reason about gaps between them.
    rawlist = sorted(rawlist, key=lambda raw: raw.identification.zdStartTime)

    raw_bbox_lists = []
    chirp_durations = []
    for raw in rawlist:
        raw_chan = find_overlapping_channel(raw, out_chan)

        freq = raw_chan.freq_id
        bbox_lists = raw.getSubSwathBboxes(freq, polarization=raw_chan.pol,
            epoch=orbit.reference_epoch, num_ignore=num_ignore,
            min_segment_length=min_segment_length, max_pulse_gap=max_pulse_gap,
            use_rx_pulse_mask=use_rx_pulse_mask)
        raw_bbox_lists.extend(bbox_lists)

        txpol = raw_chan.pol[0]
        T = raw.getChirpParameters(freq, txpol)[3]
        chirp_durations.extend(len(bbox_lists) * [T])

    # Force azimuth continuity since Raw.getSubSwathBboxes doesn't know final
    # PRI so there's a 1-pulse gap between observations.  Note that there
    # should be no gap between 10-second DWP updates.
    for i in range(len(raw_bbox_lists) - 1):
        # Each subswath should have the same start/end time, just different
        # ranges.
        t_cur = raw_bbox_lists[i][0].last.time
        t_next = raw_bbox_lists[i + 1][0].first.time
        dt = t_next - t_cur
        if dt <= max_observation_gap:
            if dt > 0.0:
                log.info(f"Merging observations separated by {dt * 1e6:.2f} us "
                    f"at {orbit.reference_epoch + isce3.core.TimeDelta(t_cur)}")
            elif dt < 0.0:
                # The time difference should always be positive since there's at
                # least one PRI between the end of one observation and the start
                # of the next one.  However, as of 2026-05-04, L0B time stamps
                # are derived from LRCLK counts using a model that's updated
                # every downlink pass.  If the observations were downlinked on
                # separate passes, it's conceivable that time could go backwards
                # (though this would violate requirements).  If that happens it
                # seems safe to assume that's a seamless transition, so just log
                # it and proceed.
                log.warning("Time decremented between observations.  "
                    "Assuming seamless transition.")
            for bbox in raw_bbox_lists[i]:
                bbox.last.time = max(t_next, t_cur)
        else:
            log.warning(f"Gap between observations {dt:7f} s exceeds threshold "
                f"for seamless observations ({max_observation_gap} s).")

    return raw_bbox_lists, chirp_durations


def get_focused_sub_swaths(rawlist, out_chan, grid, orbit, doppler, dem, azres,
                           rdr2geo_params=dict(), geo2rdr_params=dict(),
                           ignore_failure=False, polygon_segment_length=50.0,
                           num_ignore=25, max_observation_gap=0.002,
                           min_segment_length=2000, max_pulse_gap=1):
    """
    Determine fully-focused regions of the image in a format suitable for
    populating the validSamplesSubSwathX RSLC datasets.

    Parameters
    ----------
    rawlist : list[nisar.products.readers.Raw.Raw]
        List of raw data files (observations) that will be processed.
    out_chan : nisar.mixed_mode.PolChannel
        Desired channel to process (will be matched with available raw data
        using mixed-mode logic).
    grid : isce3.product.RadarGridParameters
        Grid for focused image (zero-Doppler).
    orbit : isce3.core.Orbit
        Trajectory of antenna phase center.  Its time span must cover the entire
        collection of raw data plus any reskew time offset between the native-
        and zero-Doppler radar coordinate systems.
    doppler : isce3.core.LUT2d
        Doppler centroid of raw data, in Hz.
    dem : isce3.geometry.DEMInterpolator
        Digital elevation model.
    azres : float
        Processed azimuth resolution, in meters.
    rdr2geo_params : dict
        Parameters for rdr2geo_bracket
    geo2rdr_params : dict
        Parameters for geo2rdr_bracket
    ignore_failure : bool
        If set to True and isce3.focus.get_focused_sub_swaths fails for any
        reason, then a mask corresponding to all-pixels-valid will be returned.
        Otherwise an exception will be raised on failures.  This can be useful
        for datasets where the orbit data covers all the raw data but without
        enough extra for the reskew to the zero-Doppler image grid.
    polygon_segment_length : float, optional
        Length scale over which subswath boundary can be considered linear,
        in meters.
    num_ignore : int, optional
        Number of pulses to ignore when calculating the valid data region at the
        end of a fixed-PRF observation.  This is relevant when a fixed-PRF
        observation is immediately followed by a dithered observation, as the
        dithered pulses in the air will overlap the last few receive windows of
        the fixed-PRF one.
    max_observation_gap : float, optional
        Max allowed time (in seconds) between the last pulse of one observation
        and the first pulse of the following observation for the two to be
        considered seamless.  Larger raw data gaps may result in a synthetic
        aperture being marked invalid in the RSLC.
    min_segment_length : int, optional
        Segments with fewer than this number of consecutive valid pulses
        will not be returned.  This helps avoid unnecessary bookkeeping when
        lots of missing pulses are sprinkled throughout an observation. Must
        be >= 2 pulses.
    max_pulse_gap : int, optional
        Segments (each of which must be at least min_segment_length pulses)
        separated by max_pulse_gap or fewer invalid pulses will be joined
        together.  This provides an easy way to ignore isolated gaps of a
        single invlid pulse, for example.  Set to 0 to disable merging.

    Returns
    -------
    swaths : numpy.ndarray[np.uint32]
        Array of [start, stop) valid data regions, shape = (nswath, npulse, 2)
        where nswath is the number of valid sub-swaths and npulse is the length
        of the focused image grid.
    """
    raw_bbox_lists, chirp_durations = get_raw_sub_swath_bboxes(rawlist,
        out_chan, orbit, num_ignore=num_ignore,
        min_segment_length=min_segment_length, max_pulse_gap=max_pulse_gap,
        max_observation_gap=max_observation_gap)

    try:
        swaths = isce3.focus.get_focused_sub_swaths(raw_bbox_lists,
            chirp_durations, orbit, doppler, azres, grid, dem=dem,
            rdr2geo_params=rdr2geo_params, geo2rdr_params=geo2rdr_params,
            max_segment_length=polygon_segment_length)
    except Exception as e:
        if ignore_failure:
            log.error("Failed to calculate valid subswath masks!  "
                "The entire radar grid will be assumed valid.")
            swaths = np.zeros((1, grid.length, 2), dtype=np.uint32)
            swaths[..., 1] = grid.width
        else:
            raise e
    return swaths


def _mark_all_valid(image, bit, blocksize=None):
    """
    Set `bit` in every pixel of `image`, writing in row blocks so the whole
    array need not be held in memory at once.

    Returns
    -------
    num_valid : int
        Total number of pixels in the image.
    """
    nrows, ncols = image.shape
    if blocksize is None:
        blocksize = getattr(image, "chunks", (512,))[0]
    mask = np.array(1 << bit, dtype=image.dtype)
    for i0 in range(0, nrows, blocksize):
        rows = slice(i0, min(i0 + blocksize, nrows))
        block = image[rows] | mask
        if hasattr(image, "write_direct"):
            image.write_direct(block, None, rows)
        else:
            image[rows] = block
    return nrows * ncols


def save_valid_data_mask(rawlist, out_chan, grid, orbit, doppler, dem, azres,
                         image, rdr2geo_params=dict(), geo2rdr_params=dict(),
                         ignore_failure=False, polygon_segment_length=50.0,
                         num_ignore=25, max_observation_gap=0.002,
                         min_segment_length=2000, max_pulse_gap=1,
                         blocksize=None):
    """
    Determine fully-focused regions of the image and write the result as a
    per-pixel boolean mask.  The bit used to mark valid pixels is chosen
    according to `out_chan.pol` (see `PolValidMask`), so that mask images for
    different polarization channels can be safely OR-ed into the same image.

    Parameters
    ----------
    rawlist : list[nisar.products.readers.Raw.Raw]
        List of raw data files (observations) that will be processed.
    out_chan : nisar.mixed_mode.PolChannel
        Desired channel to process (will be matched with available raw data
        using mixed-mode logic).
    grid : isce3.product.RadarGridParameters
        Grid for focused image (zero-Doppler).
    orbit : isce3.core.Orbit
        Trajectory of antenna phase center.  Its time span must cover the entire
        collection of raw data plus any reskew time offset between the native-
        and zero-Doppler radar coordinate systems.
    doppler : isce3.core.LUT2d
        Doppler centroid of raw data, in Hz.
    dem : isce3.geometry.DEMInterpolator
        Digital elevation model.
    azres : float
        Processed azimuth resolution, in meters.
    image : array_like
        Output mask, must have shape matching `grid.shape`.  May be an HDF5
        dataset, in which case writes are chunk-aligned.  Should be
        initialized to zero (at least in the bit position assigned to
        `out_chan.pol`, see `PolValidMask`).
    rdr2geo_params : dict
        Parameters for rdr2geo_bracket
    geo2rdr_params : dict
        Parameters for geo2rdr_bracket
    ignore_failure : bool
        If set to True and isce3.focus.save_valid_data_mask fails for any
        reason, then the mask will be set to all-pixels-valid.  Otherwise an
        exception will be raised on failures.  This can be useful for
        datasets where the orbit data covers all the raw data but without
        enough extra for the reskew to the zero-Doppler image grid.
    polygon_segment_length : float, optional
        Length scale over which subswath boundary can be considered linear,
        in meters.
    num_ignore : int, optional
        Number of pulses to ignore when calculating the valid data region at the
        end of a fixed-PRF observation.  This is relevant when a fixed-PRF
        observation is immediately followed by a dithered observation, as the
        dithered pulses in the air will overlap the last few receive windows of
        the fixed-PRF one.
    max_observation_gap : float, optional
        Max allowed time (in seconds) between the last pulse of one observation
        and the first pulse of the following observation for the two to be
        considered seamless.  Larger raw data gaps may result in a synthetic
        aperture being marked invalid in the RSLC.
    min_segment_length : int, optional
        Segments with fewer than this number of consecutive valid pulses
        will not be returned.  This helps avoid unnecessary bookkeeping when
        lots of missing pulses are sprinkled throughout an observation. Must
        be >= 2 pulses.
    max_pulse_gap : int, optional
        Segments (each of which must be at least min_segment_length pulses)
        separated by max_pulse_gap or fewer invalid pulses will be joined
        together.  This provides an easy way to ignore isolated gaps of a
        single invlid pulse, for example.  Set to 0 to disable merging.
    blocksize : int, optional
        Number of rows to rasterize and write at a time.  Defaults to the
        chunk size of `image` if it is an HDF5 dataset, otherwise 512.

    Returns
    -------
    num_valid : int
        Total number of valid pixels in the image.
    """
    bit = int(_PolBit[out_chan.pol])

    raw_bbox_lists, chirp_durations = get_raw_sub_swath_bboxes(rawlist,
        out_chan, orbit, num_ignore=num_ignore,
        min_segment_length=min_segment_length, max_pulse_gap=max_pulse_gap,
        max_observation_gap=max_observation_gap, use_rx_pulse_mask=True)

    try:
        num_valid = isce3.focus.save_valid_data_mask(raw_bbox_lists,
            chirp_durations, orbit, doppler, azres, grid, image, dem=dem,
            rdr2geo_params=rdr2geo_params, geo2rdr_params=geo2rdr_params,
            max_segment_length=polygon_segment_length, blocksize=blocksize,
            bit=bit)
    except Exception as e:
        if ignore_failure:
            log.error("Failed to calculate valid subswath mask!  "
                "The entire radar grid will be assumed valid.")
            num_valid = _mark_all_valid(image, bit, blocksize=blocksize)
        else:
            raise e
    return num_valid
