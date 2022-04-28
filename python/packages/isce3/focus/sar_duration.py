import isce3
from isce3.core import Orbit, Ellipsoid
import numpy as np


def get_sar_duration(t: float, r: float, orbit: Orbit, ellipsoid: Ellipsoid,
                     azres: float, wavelength: float):
    """
    Get approximate synthetic aperture duration (coherent processing interval).

    Parameters
    ---------
    t : float
        Azimuth time, in seconds since orbit epoch
    r : float
        Range, in meters
    orbit : isce3.core.Orbit
    ellipsoid : isce3.core.Ellipsoid
    azres : float
        Desired azimuth resolution, in meters

    Returns
    -------
    cpi : float
        Synthetic aperture duration (approximate), s
    """
    pos, vel = orbit.interpolate(t)
    vs = np.linalg.norm(vel)
    # assume radar and target are close enough that r_dir() is constant
    lon, lat, h = ellipsoid.xyz_to_lon_lat(pos)
    hdg = isce3.geometry.heading(lon, lat, vel)
    a = ellipsoid.r_dir(hdg, lat)
    # assume circular orbit (no nadir component of velocity)
    vg = vs / (1 + h / a)
    # small angle approximations and ignore window factor
    return wavelength / (2 * azres) * r / vg
