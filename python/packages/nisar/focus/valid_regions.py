#!python3
import logging
import isce3
from nisar.mixed_mode import find_overlapping_channel

log = logging.getLogger("focus")

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
    # Need raw files sorted in time so we can reason about gaps between them.
    rawlist = sorted(rawlist, key=lambda raw: raw.identification.zdStartTime)

    raw_bbox_lists = []
    chirp_durations = []
    for raw in rawlist:
        raw_chan = find_overlapping_channel(raw, out_chan)

        freq = raw_chan.freq_id
        bbox_lists = raw.getSubSwathBboxes(freq, epoch=orbit.reference_epoch,
            num_ignore=num_ignore, min_segment_length=min_segment_length,
            max_pulse_gap=max_pulse_gap)
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
                    f"at {orbit.reference_epoch + TimeDelta(t_cur)}")
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