import numpy as np

from isce3.core import xyz_to_enu, Ellipsoid, LUT2d, Orbit
from isce3.geometry import rdr2geo, DEMInterpolator
from isce3.product import RadarGridParameters

def compute_incidence_angle(t: float, srange: float, orbit: Orbit,
                            doppler_lut: LUT2d, rdr_grid: RadarGridParameters,
                            dem_interp: DEMInterpolator, ellipsoid: Ellipsoid):

    '''
    Compute incidence angle for given parameters

    Parameters
    ----------
    t: float
        UTC time in seconds after radar grid epoch
    srange: float
        Slant range to compute incidence angle for
    orbit: isce3.core.Orbit
        Orbit used to compute incidence angle
    doppler_lut: isce3.core.LUT2d
        Doppler centroid used to compute incidence angle
    rdr_grid: isce3.product.RadarGridParameters
        Radar grid for used to compute incidence angle
    dem_interp: isce3.geometry.DEMInterpolator
        Digital elevation model, m above ellipsoid. Defaults to h=0.
    ellipsoid: isce3.core.Ellipsoid
        Ellipsoid with same EPSG as DEM interpolator

    Returns
    -------
    float
        Incidence angle in radians
    '''
    # Get satellite position at time t
    sat_pos, _ = orbit.interpolate(t)

    # Get doppler at given time and slant range
    doppler = doppler_lut.eval(t, srange)

    # Compute target llh for given time and slant range
    llh = rdr2geo(t, srange, orbit, rdr_grid.lookside, doppler,
                  rdr_grid.wavelength, dem_interp)

    # Covert llh to xyz for ground point
    target_xyz = ellipsoid.lon_lat_to_xyz(llh)

    # Compute vector from satellite to ground
    sat_to_ground = target_xyz - sat_pos

    # Compute ENU coordinates around target
    xyz2enu = xyz_to_enu(llh[1], llh[0])
    enu = np.dot(xyz2enu, sat_to_ground)

    # Return incidence angle in radians
    cosalpha = np.abs(enu[2]) / np.linalg.norm(enu)
    return np.arccos(cosalpha)
