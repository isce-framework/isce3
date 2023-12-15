'''
Package to compute TEC LUT from JSON file
'''
from datetime import datetime
import json
import os

import numpy as np
import journal

import isce3
from isce3.geometry import compute_incidence_angle

# constants used compute ionospheric range delay
K = 40.31 # its a constant in m3/s2
TECU = 1e16 # its a constant to convert the TEC product to electrons / m2

def _compute_ionospheric_range_delay(utc_time: list, tec_json_dict: dict, nr_fr: str,
                                     nr_fr_rg: float, center_freq: float,
                                     orbit: isce3.core.Orbit,
                                     doppler_lut: isce3.core.LUT2d,
                                     radar_grid: isce3.product.RadarGridParameters,
                                     dem_interp: isce3.geometry.DEMInterpolator,
                                     ellipsoid: isce3.core.Ellipsoid,
                                     tec_time_mask: list):
    '''
    Compute near or far TEC delta range

    Parameters
    ----------
    utc_time: list
        UTC times as seconds after SLC epoch to compute TEC delta range over
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
    tec_time_mask: list
        List of bools to extract valid total and top TEC values

    Returns
    -------
    np.ndarray
        TEC delta range
    '''
    # compute sub orbital TEC from total and top TEC in JSON
    sub_orbital_tec = _get_suborbital_tec(tec_json_dict, nr_fr, tec_time_mask)

    incidence = [compute_incidence_angle(t, nr_fr_rg, orbit, doppler_lut,
                                         radar_grid, dem_interp, ellipsoid)
                 for t in utc_time]

    delta_r = K * sub_orbital_tec * TECU / center_freq**2 / np.cos(incidence)

    return delta_r


def _get_suborbital_tec(tec_json_dict: dict,
                        nr_fr: str,
                        tec_time_mask: list):
    '''
    Get the suborbital TEC from IMAGEN TEC product parsed as a dictionary by
    subtracting the total TEC by top (i.e. above the satellite) TEC

    Parameters
    ----------
    tec_json_dict: dict
        IMAGEN TEC product as a dictionary.
    nr_fr: ['Nr', 'Fr']
        Near-range or Far-range
    tec_time_mask: list
        List of bools to extract valid total and top TEC values

    Returns
    -------
    sub_orbital_tec: np.ndarray
        Suborbital TEC
    '''
    # compute sub orbital TEC from total and top TEC in JSON
    tot_tec = np.array(tec_json_dict[f'totTec{nr_fr}'])
    top_tec = np.array(tec_json_dict[f'topTec{nr_fr}'])
    sub_orbital_tec = tot_tec - top_tec
    sub_orbital_tec = np.array([tec for tec, m in
                                zip(sub_orbital_tec, tec_time_mask) if m])

    return sub_orbital_tec


def tec_lut2d_from_json_srg(json_path: str, center_freq: float,
                        orbit: isce3.core.Orbit,
                        radar_grid: isce3.product.RadarGridParameters,
                        doppler_lut: isce3.core.LUT2d, dem_path: str,
                        margin: float=40.0) -> isce3.core.LUT2d:
    '''
    Create a TEC LUT2d from a JSON source

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
    error_channel = journal.error('tec_product.tec_lut2d_from_json')

    if not os.path.isfile(json_path):
        err_str = f'TEC JSON path not found: {json_path}'
        error_channel.log(err_str)

    with open(json_path, 'r') as fid:
        tec_json_dict = json.load(fid)

    # get the UTC time and mask for tec_data that covers
    # sensing start / stop with margin applied
    utc_time, time_mask = get_tec_time(tec_json_dict, radar_grid, margin)

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
    delta_r = np.vstack([_compute_ionospheric_range_delay(utc_time, tec_json_dict,
                                                          nr_fr, rg,
                                                          center_freq, orbit,
                                                          doppler_lut,
                                                          radar_grid,
                                                          dem_interp,
                                                          ellipsoid,
                                                          time_mask)
                         for nr_fr, rg in zip(['Nr', 'Fr'], rg_vec)]).T

    return isce3.core.LUT2d(rg_vec, utc_time, delta_r)


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
    error_channel = journal.error('tec_product.tec_lut2d_from_json')

    if not os.path.isfile(json_path):
        err_str = f'TEC JSON path not found: {json_path}'
        error_channel.log(err_str)

    # Load TEC from NISAR TEC JSON file
    with open(json_path, 'r') as fid:
        tec_json_dict = json.load(fid)
    
    # get the UTC time and mask for tec_data that covers
    # sensing start / stop with margin applied
    utc_time, time_mask = get_tec_time(tec_json_dict, radar_grid, margin)

    # Load the TEC information from IMAGEN parsed as dictionary
    # Use radar grid start/end range for near/far range
    # Transpose stacked output to get shape to be consistent with coordinates
    rg_vec = [radar_grid.starting_range, radar_grid.end_range]
    tec_suborbital = np.vstack([_get_suborbital_tec(tec_json_dict,
                                                    nr_fr, time_mask)
                                for nr_fr in ['Nr', 'Fr']]).T

    # set up up the LUT grids for az. iono. delay
    tec_gradient_az_spacing = (utc_time[-1] - utc_time[0]) / (len(utc_time) - 1)

    # staggered grid to compute TEC gradient
    tec_gradient_utc_time = utc_time[1:] - tec_gradient_az_spacing / 2

    tec_gradient = np.diff(tec_suborbital, axis=0) / tec_gradient_az_spacing

    speed_vec_lut = np.array([np.linalg.norm(orbit.interpolate(t)[1]) for t in
                              tec_gradient_utc_time])
    az_fm_rate = np.outer(-2 * speed_vec_lut**2 * (center_freq / isce3.core.speed_of_light),
                          1 / np.array(rg_vec))

    t_az_delay = (-2 * K
                  / (isce3.core.speed_of_light * az_fm_rate * center_freq)
                  * tec_gradient * TECU)
    return isce3.core.LUT2d(rg_vec, tec_gradient_utc_time, t_az_delay)


def get_tec_time(tec_json_dict: dict,
                 radar_grid: isce3.product.RadarGridParameters,
                 margin: float):
    '''
    Extract the UTC time and TEC time mask that covers
    the radar grid's start / stop time with margin

    Parameters
    ----------
    tec_json_dict: dict
        IMAGEN TEC JSON file parsed as dictionary
    radar_grid: isce3.product.RadarGridParameters
        Radar grid of the data
    margin: float
        Temporal margin in seconds

    Returns
    -------
    utc_time: np.array
        1D array of UTC time that covers the sensing start / stop time with margin
    
    time_mask: list
        A mask to identify the TEC data that corresponds to `utc_time`
    '''
    # Save UTC time as total seconds since reference epoch of radar grid
    ref_epoch = datetime.fromisoformat(radar_grid.ref_epoch.isoformat()[:-3])
    utc_time = [(datetime.fromisoformat(t) - ref_epoch).total_seconds()
                for t in tec_json_dict['utc']]

    # Filter utc_time to only save times near radar_grid.
    # Pad data before and after to ensure enough TEC data is collected
    # to cover sensing start / stop with margin.
    t_lower_bound = radar_grid.sensing_start - margin
    t_upper_bound = radar_grid.sensing_stop + margin
    time_mask = [t_lower_bound <= t <= t_upper_bound for t in utc_time]
    utc_time = np.array([t for t, m in zip(utc_time, time_mask) if m])

    return utc_time, time_mask
