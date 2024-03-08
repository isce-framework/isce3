'''
Package to compute TEC LUT from JSON file
'''
from datetime import datetime, timedelta
import json
import os

import numpy as np
import journal

import isce3
from isce3.geometry import compute_incidence_angle

# constants used compute ionospheric range delay
K = 40.31 # its a constant in m3/s2
TECU = 1e16 # its a constant to convert the TEC product to electrons / m2

def _compute_ionospheric_range_delay(utc_time: np.ma.MaskedArray,
                                     tec_json_dict: dict, nr_fr: str,
                                     nr_fr_rg: float, center_freq: float,
                                     orbit: isce3.core.Orbit,
                                     doppler_lut: isce3.core.LUT2d,
                                     radar_grid: isce3.product.RadarGridParameters,
                                     dem_interp: isce3.geometry.DEMInterpolator,
                                     ellipsoid: isce3.core.Ellipsoid) -> np.ndarray:
    '''
    Compute near or far TEC delta range

    Parameters
    ----------
    utc_time: numpy.ma.MaskedArray
        UTC times as seconds after SLC epoch to compute TEC delta range over
        with values outside radar grid or orbit masked.
    tec_json_dict: dict
        TEC JSON as a dict. Keys are TEC parameters and values are the values
        of said parameter.
    nr_fr: ['Nr', 'Fr']
        String values used to access Near or Far TEC parameters in TEC JSON
        dict
    nr_fr_rg: float
        Near or far range of the radar grid
    center_freq: float
        Processed center frequency of swath (Hz)
    orbit: isce3.core.Orbit
        Orbit for associated SLC
    doppler_lut: isce3.core.LUT2d
        Doppler centroid of SLC
    radar_grid: isce3.product.RadarGridParameters
        Radar grid for associated SLC
    dem_interp: isce3.geometry.DEMInterpolator
        Digital elevation model, m above ellipsoid. Defaults to h=0.
    ellipsoid: isce3.core.Ellipsoid
        Ellipsoid with same EPSG as DEM interpolator

    Returns
    -------
    np.ndarray
        TEC delta range
    '''
    # compute sub orbital TEC from total and top TEC in JSON
    sub_orbital_tec = _get_suborbital_tec(tec_json_dict, nr_fr,
                                          utc_time.mask)

    incidence = [compute_incidence_angle(t, nr_fr_rg, orbit, doppler_lut,
                                         radar_grid, dem_interp, ellipsoid)
                 for t in utc_time.compressed()]

    delta_r = K * sub_orbital_tec * TECU / center_freq**2 / np.cos(incidence)

    return delta_r


def _get_suborbital_tec(tec_json_dict: dict,
                        nr_fr: str,
                        tec_time_mask: np.ndarray) -> np.ndarray:
    '''
    Get the suborbital TEC from IMAGEN TEC product parsed as a dictionary by
    subtracting the total TEC by top (i.e. above the satellite) TEC

    Parameters
    ----------
    tec_json_dict: dict
        IMAGEN TEC product as a dictionary.
    nr_fr: ['Nr', 'Fr']
        Near-range or Far-range
    tec_time_mask: numpy.ndarray
        Mask of TEC values that fall within radar grid or orbit time span. Mask
        follows NumPy masked array convention where True is masked and False is
        not.

    Returns
    -------
    sub_orbital_tec: np.ndarray
        Suborbital TEC
    '''
    # compute sub orbital TEC from total and top TEC in JSON
    tot_tec = np.array(tec_json_dict[f'totTec{nr_fr}'])
    top_tec = np.array(tec_json_dict[f'topTec{nr_fr}'])
    sub_orbital_tec = tot_tec - top_tec
    sub_orbital_tec = sub_orbital_tec[~tec_time_mask]

    return sub_orbital_tec


def tec_lut2d_from_json_srg(json_path: str, center_freq: float,
                            orbit: isce3.core.Orbit,
                            radar_grid: isce3.product.RadarGridParameters,
                            doppler_lut: isce3.core.LUT2d, dem_path: str,
                            margin: float=40.0) -> isce3.core.LUT2d:
    '''
    Create a TEC LUT2d for slant range correction from a JSON source

    Parameters
    ----------
    json_path: str
        Path to JSON file containing TEC data
    center_freq: float
        Processed center frequency of swath (Hz)
    orbit: isce3.core.Orbit
        Orbit for associated SLC
    radar_grid: isce3.product.RadarGridParameters
        Radar grid for associated SLC
    doppler_lut: isce3.core.LUT2d
        Doppler centroid of SLC
    dem_path: str
        Digital elevation model, m above ellipsoid. Defaults to h=0.
    margin: float
        Margin (seconds) to pad to sensing start and stop times when extracting
        TEC data. Default 40 seconds.

    Returns
    -------
    isce3.core.LUT2d
        LUT2d object for geolocation correction in slant range delay in meters
    '''
    error_channel = journal.error('tec_product.tec_lut2d_from_json_srg')

    if not os.path.isfile(json_path):
        err_str = f'TEC JSON path not found: {json_path}'
        error_channel.log(err_str)

    # Perform sanity checks.
    _check_orbit_contains_radar_grid(radar_grid, orbit)
    _check_orbit_and_rdr_grid_common_ref_epoch(orbit, radar_grid)

    with open(json_path, 'r') as fid:
        tec_json_dict = json.load(fid)

    # Get the UTC time and mask for tec_data that covers
    # sensing start / stop with margin applied
    t_since_epoch_masked = _get_tec_time(tec_json_dict, radar_grid, orbit,
                                         margin, doppler_lut)

    # Load DEM into interpolator and get ellipsoid object from DEM EPSG
    dem_raster = isce3.io.Raster(dem_path)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Using zero DEM in current implementation to account for TEC file bounds
    # being larget than that of the scene DEM
    dem_interp = isce3.geometry.DEMInterpolator()

    # Compute near and far delta range for near and far TEC
    # Use radar grid start/end range for near/far range
    # Transpose stacked output to get shape to be consistent with coordinates
    rg_vec = [radar_grid.starting_range, radar_grid.end_range]
    delta_r = np.vstack([_compute_ionospheric_range_delay(t_since_epoch_masked,
                                                          tec_json_dict,
                                                          nr_fr, rg,
                                                          center_freq, orbit,
                                                          doppler_lut,
                                                          radar_grid,
                                                          dem_interp,
                                                          ellipsoid)
                         for nr_fr, rg in zip(['Nr', 'Fr'], rg_vec)]).T

    return isce3.core.LUT2d(rg_vec, t_since_epoch_masked.compressed(), delta_r)


def tec_lut2d_from_json_az(json_path: str, center_freq: float,
                           orbit: isce3.core.Orbit,
                           radar_grid: isce3.product.RadarGridParameters,
                           margin: float=40.0) -> isce3.core.LUT2d:
    '''
    Create a TEC LUT2d for azimuth time correction from a JSON source

    Parameters
    ----------
    json_path: str
        Path to JSON file containing TEC data
    center_freq: float
        Processed center frequency of swath (Hz)
    orbit: isce3.core.Orbit
        Orbit for associated SLC
    radar_grid: isce3.product.RadarGridParameters
        Radar grid for associated SLC
    margin: float
        Margin (seconds) to pad to sensing start and stop times when extracting
        TEC data. Default 40 seconds.

    Returns
    -------
    isce3.core.LUT2d
        LUT2d object for geolocation correction in azimuth time delay in seconds
    '''
    error_channel = journal.error('tec_product.tec_lut2d_from_json_az')

    # Perform sanity checks.
    _check_orbit_contains_radar_grid(radar_grid, orbit)
    _check_orbit_and_rdr_grid_common_ref_epoch(orbit, radar_grid)

    if not os.path.isfile(json_path):
        err_str = f'TEC JSON path not found: {json_path}'
        error_channel.log(err_str)

    # Load TEC from NISAR TEC JSON file
    with open(json_path, 'r') as fid:
        tec_json_dict = json.load(fid)

    # get the UTC time and mask for tec_data that covers
    # sensing start / stop with margin applied
    t_since_epoch_masked = _get_tec_time(tec_json_dict, radar_grid, orbit,
                                         margin)

    # Load the TEC information from IMAGEN parsed as dictionary
    # Use radar grid start/end range for near/far range
    # Transpose stacked output to get shape to be consistent with coordinates
    rg_vec = [radar_grid.starting_range, radar_grid.end_range]
    tec_suborbital = np.vstack([_get_suborbital_tec(tec_json_dict,
                                                    nr_fr,
                                                    t_since_epoch_masked.mask)
                                for nr_fr in ['Nr', 'Fr']]).T

    # set up up the LUT grids for az. iono. delay
    t_since_epoch = t_since_epoch_masked.compressed()
    tec_gradient_az_spacing = (t_since_epoch[-1] - t_since_epoch[0]) / (len(t_since_epoch) - 1)

    # staggered grid to compute TEC gradient
    tec_gradient_t_since_epoch = t_since_epoch[1:] - tec_gradient_az_spacing / 2

    tec_gradient = np.diff(tec_suborbital, axis=0) / tec_gradient_az_spacing

    speed_vec_lut = np.array([np.linalg.norm(orbit.interpolate(t)[1]) for t in
                              tec_gradient_t_since_epoch])
    az_fm_rate = np.outer(-2 * speed_vec_lut**2 * (center_freq / isce3.core.speed_of_light),
                          1 / np.array(rg_vec))

    t_az_delay = (-2 * K
                  / (isce3.core.speed_of_light * az_fm_rate * center_freq)
                  * tec_gradient * TECU)
    return isce3.core.LUT2d(rg_vec, tec_gradient_t_since_epoch, t_az_delay)


def _get_tec_time(tec_json_dict: dict,
                  radar_grid: isce3.product.RadarGridParameters,
                  orbit: isce3.core.Orbit,
                  margin: float,
                  doppler_lut: isce3.core.LUT2d=isce3.core.LUT2d()
                  ) -> np.ma.MaskedArray:
    '''
    Extract the TEC UTC times masked so that valid times lie within the radar
    grid start/stop times plus margin, doppler LUT azimuth times, or orbit
    start/stop times. Doppler defaults to default LUT2d if one is not provided.

    Parameters
    ----------
    tec_json_dict: dict
        IMAGEN TEC JSON file parsed as dictionary
    radar_grid: isce3.product.RadarGridParameters
        Radar grid of the SLC data
    orbit: isce3.core.Orbit
        Orbit for associated SLC data
    margin: float
        Temporal margin in seconds to be applied to radar grid start and stop times
        to ensure sufficient TEC data is assembled.
    doppler_lut: isce3.core.LUT2d
        Doppler centroid of SLC

    Returns
    -------
    t_since_ref_epoch: numpy.ma.MaskedArray
        1D array of time of JSON TEC times in seconds since reference epoch
        where values that are outside either the radar grid or orbit time
        domain masked.
    '''
    # Get string UTC times from JSON as isce3.core.DateTime objects.
    json_utc_datetimes = [isce3.core.DateTime(iso_t_str)
                          for iso_t_str in tec_json_dict['utc']]

    # Compute lower and upper bounds as datetime objects to filter JSON UTC
    # datetime objects based on radar grid start and stop with margin applied.
    # Pad radar grid start and stop with margin to ensure enough TEC data is
    # collected.
    rdr_grid_start_minus_margin = radar_grid.ref_epoch + \
        isce3.core.TimeDelta(seconds=radar_grid.sensing_start - margin)
    rdr_grid_end_plus_margin = radar_grid.ref_epoch + \
        isce3.core.TimeDelta(seconds=radar_grid.sensing_stop + margin)

    # Determine mask start based on larger between orbit start,
    # rdr grid start - margin
    mask_start = max(orbit.start_datetime, rdr_grid_start_minus_margin)

    # Determine mask start based on smaller between orbit stop,
    # rdr grid start + margin
    mask_stop = min(orbit.end_datetime, rdr_grid_end_plus_margin)

    # If correct doppler LUT provided, check LUT start/stop against mask
    # start/stop.
    if doppler_lut.bounds_error and doppler_lut.have_data:
        doppler_start = radar_grid.ref_epoch + \
            isce3.core.TimeDelta(seconds=doppler_lut.y_start)
        doppler_stop = radar_grid.ref_epoch + \
            isce3.core.TimeDelta(seconds=doppler_lut.y_start + doppler_lut.y_spacing * (doppler_lut.length - 1))
        mask_start = max(mask_start, doppler_start)
        mask_stop = min(mask_stop, doppler_stop)

    # Compute mask of JSON UTC times that are within the bounds computed above.
    time_mask = [mask_start <= t <= mask_stop
                 for t in json_utc_datetimes]

    # Compute JSON UTC times as deltatime total seconds since reference epoch.
    t_since_ref_epoch = [(t - radar_grid.ref_epoch).total_seconds()
                         for t in json_utc_datetimes]

    t_since_ref_epoch = np.ma.MaskedArray(data=t_since_ref_epoch,
                                          mask=np.logical_not(time_mask))

    return t_since_ref_epoch


def _check_orbit_contains_radar_grid(radar_grid: isce3.product.RadarGridParameters,
                                     orbit: isce3.core.Orbit) -> None:
    """
    Helper function to check if radar grid time domain lies within orbit time
    domain. Raises an exception if radar grid does not lie within the orbit.

    Parameters
    ----------
    radar_grid: isce3.product.RadarGridParameters
        Radar grid whose time domain to be inspected to see if it lies within
        that of the orbit time domain.
    orbit: isce3.core.Orbit
        Orbit whose time domain to fully contain that of the radar grid.

    Raises
    ------
    ValueError
        If the radar grid times do not fall strictly within the orbit time domain.
    """
    # Radar grid start/stop as DateTime objects.
    rdr_grid_start = radar_grid.ref_epoch + \
        isce3.core.TimeDelta(seconds=radar_grid.sensing_start)
    rdr_grid_stop = radar_grid.ref_epoch + \
        isce3.core.TimeDelta(seconds=radar_grid.sensing_stop)

    # Determine if radar grid start and stop lie within orbit time domain.
    rdr_grid_start_in_orbit = orbit.start_datetime < rdr_grid_start < orbit.end_datetime
    rdr_grid_stop_in_orbit = orbit.start_datetime < rdr_grid_stop < orbit.end_datetime

    # Raise exception if radar grid start or stop do not lie within orbit time
    # domain.
    if not(rdr_grid_start_in_orbit and rdr_grid_stop_in_orbit):
        error_channel = journal.error(
            "tec_product._check_orbit_contains_radar_grid")

        # Error string that mentions start or stop being out of bounds.
        err_str = f'Radar grid does not lie within orbit.'
        for is_oob, start_stop in zip([rdr_grid_start_in_orbit,
                                       rdr_grid_stop_in_orbit],
                                      ["start", "stop"]):
            if not is_oob:
                err_str += f" Radar grid {start_stop} out of bounds."

        error_channel.log(err_str)
        raise ValueError(err_str)


def _check_orbit_and_rdr_grid_common_ref_epoch(orbit: isce3.core.Orbit,
                                               radar_grid: isce3.product.RadarGridParameters) -> None:
    """
    Helper function to ensure orbit reference epoch has the same as the
    radar grid. If orbit reference epoch differs from that of radar grid,
    raise an exception.

    Parameters
    ----------
    orbit: isce3.core.Orbit
        Orbit whose reference epoch is to be checked and possibly corrected.
    radar_grid: isce3.product.RadarGridParameters
        Radar grid whose reference epoch will be checked against that of the
        given orbit.

    Raises
    ------
    ValueError
        If the reference epoch of the radar grid and the orbit are not the same.
    """
    radar_grid_ref_epoch = radar_grid.ref_epoch
    orbit_ref_epoch = orbit.reference_epoch

    if radar_grid_ref_epoch != orbit_ref_epoch:
        error_channel = journal.error(
            "tec_product._orbit_and_rdr_grid_common_ref_epoch")
        err_str = f"Reference epoch of radar grid ({radar_grid_ref_epoch}) and orbit ({orbit_ref_epoch}) are not the same"
        error_channel.log(err_str)
        raise ValueError(err_str)
