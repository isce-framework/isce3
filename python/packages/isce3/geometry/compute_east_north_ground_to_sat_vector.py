import numpy as np

from isce3.core import xyz_to_enu, Ellipsoid, LUT2d, Orbit
from isce3.geometry import rdr2geo, DEMInterpolator, Rdr2GeoParams
from isce3.product import RadarGridParameters


def compute_east_north_ground_to_sat_vector(
    az_time: float,
    srange: float,
    orbit: Orbit,
    doppler_lut: LUT2d,
    rdr_grid: RadarGridParameters,
    dem_interp: DEMInterpolator,
    ellipsoid: Ellipsoid,
    rdr2geo_params: Rdr2GeoParams = Rdr2GeoParams()
):
    """
    Compute east and north components of ground to satellite unit vector for
    given parameters

    Parameters
    ----------
    az_time: float
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
    east: float
        East component of the ground to satellite ENU unit vector
    north: float
        North component of the ground to satellite ENU unit vector
    """
    # Get satellite position at given azimuth time
    sat_pos, _ = orbit.interpolate(az_time)

    # Get doppler at given time and slant range
    doppler = doppler_lut.eval(az_time, srange)

    # Compute target llh for given time and slant range
    llh = rdr2geo(
        az_time,
        srange,
        orbit,
        rdr_grid.lookside,
        doppler,
        rdr_grid.wavelength,
        dem_interp,
        threshold=rdr2geo_params.threshold,
        maxite=rdr2geo_params.maxiter,
        extraite=rdr2geo_params.extraiter
    )

    # Covert llh to xyz for ground point
    target_xyz = ellipsoid.lon_lat_to_xyz(llh)

    # Compute vector from ground to satellite
    ground_to_sat = sat_pos - target_xyz

    # Compute ENU coordinates of ground to satellite
    xyz2enu = xyz_to_enu(llh[1], llh[0])

    # Compute ground to satellite ENU unit vector
    enu = np.dot(xyz2enu, ground_to_sat)
    east, north, _ = enu / np.linalg.norm(enu)

    return east, north
