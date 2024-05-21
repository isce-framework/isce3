
def crop_external_orbit(external_orbit_obj,
                        rslc_orbit_obj,
                        npad = 7):
    """
    Crop the external orbit using the start and end time of the RSLC
    internal RSLC orbit with the number of padding

    Parameters
    ----------
    external_orbit_obj : isce3.core.Orbit
        The external orbit object
    rslc_orbit_obj : isce3.core.Orbit
        The RSLC interal orbit object
    npad : int
        The number of padding (default: 7)

    Returns
    -------
    isce3.core.Orbit
        the cropped new orbit object
    """

    # adjust npad based on the "10 seconds interval"
    actual_npad = int(npad  * rslc_orbit_obj.spacing / external_orbit_obj.spacing)

    # Deal with the case that start or end time of the external orbit
    # is outside the RSLC internal orbit, in addition,
    # Both  external and internal orbits have the same reference epoch.
    start_datetime = rslc_orbit_obj.start_datetime
    t0 = (external_orbit_obj.start_datetime -
          rslc_orbit_obj.reference_epoch).total_seconds()
    t1 = (rslc_orbit_obj.start_datetime -
          rslc_orbit_obj.reference_epoch).total_seconds()
    if t1 < t0:
        start_datetime = external_orbit_obj.start_datetime

    end_datetime = rslc_orbit_obj.end_datetime
    t0 = (external_orbit_obj.end_datetime -
          rslc_orbit_obj.reference_epoch).total_seconds()
    t1 = (rslc_orbit_obj.end_datetime -
          rslc_orbit_obj.reference_epoch).total_seconds()
    if t1 > t0:
        end_datetime = external_orbit_obj.end_datetime

    # Crop the external orbit using the internal RSLC orbit
    # with additional the number of npad state vectors
    cropped_orbit = external_orbit_obj.crop(
        start_datetime,
        end_datetime,
        npad = actual_npad)

    return cropped_orbit