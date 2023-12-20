from isce3.core import Ellipsoid, LookSide, LUT2d, Orbit
from isce3.geometry import DEMInterpolator, rdr2geo_bracket, geo2rdr_bracket
from typing import Union, Optional


def rdr2rdr(t: float,
            r: float,
            orbit: Orbit,
            side: Union[str, LookSide],
            doppler: LUT2d,
            wavelength: float,
            dem: DEMInterpolator = DEMInterpolator(),
            ellipsoid: Ellipsoid = Ellipsoid(),
            doppler_out: Optional[LUT2d] = None,
            orbit_out: Optional[Orbit] = None,
            rdr2geo_params=dict(),
            geo2rdr_params=dict()):
    """
    Convert coordinates from one radar geometry to another with a different
    Doppler (reskew) or orbit (motion compensation).

    Parameters
    ----------
    t : float
	    Azimuth time, seconds past orbit epoch
    r : float
	    Range, m
    orbit : isce3.core.Orbit
	    Orbit defining radar motion on input path
    side : {"left", "right", isce3.core.LookSide.Left, isce3.core.LookSide.Right}
	    Flag desribing which side the radar is looking.
    doppler : isce3.core.LUT2d
        Doppler look up table vs input range and azimuth time, Hz
    wavelength : float
        Wavelength associated with Doppler, m
    dem : isce3.geometry.DEMInterpolator, optional
        Digital elevation model, m above ellipsoid.  Defaults to h=0.
    ellipsoid : isce3.core.Ellipsoid, optional
        Ellipsoid describing surface.  Defaults to WGS-84.
    doppler_out : isce3.core.LUT2d, optional
        Doppler look up table vs output range and azimuth time, Hz
        Defaults to input `doppler`.
    orbit_out : isce3.core.Orbit, optional
        Orbit defining radar motion on output path.  Defaults to input `orbit`.
    rdr2geo_params : dict, optional
        Dictionary specifying convergence paramters for rdr2geo solver.
        Keys among {"tol_aztime", "look_min", "look_max"}
        See isce3.geometry.rdr2geo_bracket
    geo2rdr_params: dict, optional
        Dictionary specifying convergence paramters for geo2rdr solver.
        Keys among {"tol_height", "time_start", "time_start"}
        See isce3.geometry.geo2rdr_bracket

    Returns
    -------
    t_out : float
        Azimuth time in output geometry, seconds past orbit epoch
    r_out : float
        Range in output geometry, m
    """
    if (orbit_out is None) and (doppler_out is None):
        return t, r
    if orbit_out is None:
        orbit_out = orbit
    if doppler_out is None:
        doppler_out = doppler
    doppler_in = doppler.eval(t, r)
    xyz = rdr2geo_bracket(t, r, orbit, side, doppler_in, wavelength, dem,
                          **rdr2geo_params)
    tout, rout = geo2rdr_bracket(xyz, orbit_out, doppler_out, wavelength, side,
                                 **geo2rdr_params)
    return tout, rout
