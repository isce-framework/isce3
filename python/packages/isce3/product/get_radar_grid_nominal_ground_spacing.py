import numpy as np
from numpy.linalg import norm

import isce3
from isce3.core import Orbit
from isce3.product import RadarGridParameters


def get_radar_grid_nominal_ground_spacing(grid: RadarGridParameters,
                                          orbit: Orbit,
                                          **kw):
    """Calculate along-track and ground-range spacing at middle of swath.

    Parameters
    ----------
    grid : isce3.product.RadarGridParameters
        Radar grid
    orbit : isce3.core.Orbit
        Radar orbit
    threshold : float, optional
    maxiter : int, optional
    extraiter : int, optional
        See rdr2geo

    Returns
    -------
    azimuth_spacing : float
        Along-track spacing in meters at mid-swath
    ground_range_spacing : float
        Ground range spacing in meters at mid-swath
    """
    if orbit.reference_epoch != grid.ref_epoch:
        raise ValueError("Need orbit and grid to have same reference epoch")
    pos, vel = orbit.interpolate(grid.sensing_mid)
    doppler = 0.0
    target_llh = isce3.geometry.rdr2geo(grid.sensing_mid,
                                        grid.mid_range,
                                        orbit,
                                        grid.lookside,
                                        doppler,
                                        grid.wavelength,
                                        **kw)
    ell = isce3.core.Ellipsoid()
    target_xyz = ell.lon_lat_to_xyz(target_llh)
    azimuth_spacing = norm(vel) / grid.prf * norm(target_xyz) / norm(pos)
    los_enu = isce3.geometry.enu_vector(target_llh[0], target_llh[1],
        target_xyz - pos)
    cos_inc = -(los_enu[2] / norm(los_enu))
    sin_inc = np.sqrt(1 - cos_inc**2)
    ground_range_spacing = grid.range_pixel_spacing / sin_inc
    return azimuth_spacing, ground_range_spacing
