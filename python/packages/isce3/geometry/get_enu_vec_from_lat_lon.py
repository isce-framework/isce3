import numpy as np


def get_enu_vec_from_lat_lon(lon, lat, units='degrees'):
    '''
    Calculate the east, north, and up vectors in ECEF for lon / lat provided

    Parameters
    ----------
    lon: np.ndarray
        Longitude of the points to calculate ENU vectors
    lat: np.ndarray
        Latitude of the points to calculate ENU vectors
    units: str
        Units of the `lon` and `lat`.
        Acceptable units are `radians` or `degrees`, (Default: degrees)

    Returns
    -------
    vec_e: np.ndarray
        unit vector of "east" direction in ECEF
    vec_n: np.ndarray
        unit vector of "north" direction in ECEF
    vec_u: np.ndarray
        unit vector of "up" direction in ECEF
    '''
    if units == 'degrees':
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
    elif units == 'radians':
        lon_rad = lon
        lat_rad = lat
    else:
        raise ValueError(f'"{units}" was provided for `units`, '
                         'which needs to be either `degrees` or `radians`')

    # Calculate up, north, and east vectors
    # reference: https://github.com/isce-framework/isce3/blob/944eba17f4a5b1c88c6a035c2d58ddd0d4f0709c/cxx/isce3/core/Ellipsoid.h#L154-L157 # noqa
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU # noqa
    vec_u = np.array([np.cos(lon_rad) * np.cos(lat_rad),
                      np.sin(lon_rad) * np.cos(lat_rad),
                      np.sin(lat_rad)])

    vec_n = np.array([-np.cos(lon_rad) * np.sin(lat_rad),
                      -np.sin(lon_rad) * np.sin(lat_rad),
                      np.cos(lat_rad)])

    vec_e = np.cross(vec_n, vec_u, axis=0)

    return vec_e, vec_n, vec_u
