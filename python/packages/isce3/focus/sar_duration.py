from __future__ import annotations
import isce3
from isce3.core import Orbit, Ellipsoid
import numpy as np
from scipy.special import j1
from typing import Optional


def get_radar_velocities(orbit: Orbit, time: Optional[float] = None,
                        ellipsoid: Ellipsoid = Ellipsoid()) -> tuple[float, float]:
    """
    Calculate radar along-track velocity at altitude and along ground-track.

    Parameters
    ----------
    orbit : isce3.core.Orbit
        Orbit object
    time : Optional[float]
        Time at which to calculate velocity.  If None then use orbit midpoint.
    ellipsoid : Optional[isce3.core.Ellipsoid]
        Ellipsoid to use for determining the surface normal/nadir vector.
        If None then use WGS84 ellipsoid.

    Returns
    -------
    vs : float
        Satellite along-track velocity in m/s
    vg : float
        Ground track velocity in m/s

    Notes
    -----
    The velocity component along the geodetic nadir is set to zero.
    """
    t = time if time is not None else orbit.mid_time
    pos, vel = orbit.interpolate(t)
    tcn = isce3.core.geodetic_tcn(pos, vel, ellipsoid)
    nadir = tcn.x2
    vs = np.linalg.norm(vel - vel.dot(nadir))
    lon, lat, h = ellipsoid.xyz_to_lon_lat(pos)
    hdg = isce3.geometry.heading(lon, lat, vel)
    a = ellipsoid.r_dir(hdg, lat)
    return vs, vs / (1 + h / a)


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
    _, vg = get_radar_velocities(orbit, t, ellipsoid)
    # small angle approximations and ignore window factor
    return wavelength / (2 * azres) * r / vg


def predict_azimuth_envelope(azres, prf, vs, L=12, n=256, circular=True):
    """
    Generate azimuth spectral envelope based on analytic antenna pattern
    assuming no extra apodization.

    Parameters
    ----------
    azres : float
        Azimuth resolution in meters
    prf : float
        Pulse repetition frequency in Hz
    vs : float
        Satellite along-track velocity in m/s
    L : float, optional
        Antenna diameter (or azimuth length) in meters
    n : int, optional
        Number of frequency bins
    circular : bool, optional
        True for circular aperture (jinc pattern) or False for rectangular
        aperture (sinc pattern)

    Returns
    -------
    azspec : numpy.ndarray[float]
        Azimuth spectral envelope (amplitude units).  Doppler centroid is in
        the middle of the array, as if fftshift has been called on the baseband
        spectrum.
    """
    # Wavelength cancels out in arithmetic but makes formulas clearer
    wvl = 0.24

    # processed_angle is the angular extent corresponding to processed
    # azimuth bandwidth.
    processed_angle = wvl / (2 * azres)
    # total_angle is the angular extent corresonding to given PRF.
    az_spacing = vs / prf
    total_angle = wvl / (2 * az_spacing)
    theta = np.fft.fftfreq(n) * total_angle
    theta = np.fft.fftshift(theta)

    azspec = np.zeros(n)
    x = theta * L / wvl
    # Avoid dividing by zero.  The x->0 limit of both functions is one.
    small = abs(x) < 1e-15
    azspec[small] = 1.0
    # Only calculate over frequencies in processed band (and not small values).
    mask = (abs(theta) <= processed_angle / 2.0) & (~small)
    # One-way pattern is jinc or sinc, square for two-way pattern.
    if circular:
        px = np.pi * x[mask]
        azspec[mask] = (2 * j1(px) / px)**2
    else:
        azspec[mask] = np.sinc(x[mask])**2
    return azspec
