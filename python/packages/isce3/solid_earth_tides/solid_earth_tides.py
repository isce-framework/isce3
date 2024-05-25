from datetime import datetime

import numpy as np
from osgeo import osr
import pysolid
from scipy.interpolate import RegularGridInterpolator

import isce3
from isce3.geometry import get_enu_vec_from_lat_lon


def solid_earth_tides(
    radar_grid: isce3.product.RadarGridParameters,
    lon_radar_grid: np.ndarray,
    lat_radar_grid: np.ndarray,
    hgt_radar_grid: np.ndarray,
    orbit: isce3.core.Orbit,
    ellipsoid: isce3.core.Ellipsoid,
    geo2rdr_params: isce3.geometry.Geo2RdrParams = None,
):
    """
    Compute displacement due to Solid Earth Tides (SET)
    in slant range and azimuth directions

    Parameters
    ---------
    radar_grid: isce3.product.RadarGridParameters
        Radar grid to compute Solid Earth Tides correction over
    lat_radar_grid: np.ndarray
        Latitude array on radar grid
    lon_radar_grid: np.ndarray
        Longitude array on radar grid
    hgt_radar_grid: np.ndarray
        Height array on radar grid
    ellipsoid: isce3.core.Ellipsoid
        Ellipsoid defined by DEM
    geo2rdr_params: isce3.geometry.Geo2RdrParams
        Threshold, maximum iterations, and delta range to run geo2rdr with

    Returns
    ------
    rg_set: np.ndarray
        2D array with SET displacement along LOS
    az_set: np.ndarray
        2D array with SET displacement along azimuth
    """
    # Produce geogrid with EPSG 4326 and 0.23 deg / ~5.1km resolution to
    # compute Solid Earth Tide over. Apply 0.23 deg margin geogrid to ensure
    # SET grid covers lat/lon grid to be interpolated to.
    grid_size = 0.23
    geogrid = isce3.product.bbox_to_geogrid(radar_grid,
                                            orbit,
                                            isce3.core.LUT2d(),
                                            grid_size,
                                            -grid_size,
                                            4326,
                                            margin=grid_size)

    geogrid_attrs = {
        "LENGTH": geogrid.length,
        "WIDTH": geogrid.width,
        "X_FIRST": geogrid.start_x,
        "Y_FIRST": geogrid.start_y,
        "X_STEP": geogrid.spacing_x,
        "Y_STEP": geogrid.spacing_y
    }

    # Run pySolid and get SET in ENU coordinate system
    sensing_start = datetime.fromisoformat((radar_grid.ref_epoch + isce3.core.TimeDelta(radar_grid.sensing_mid)).isoformat_usec())
    set_e, set_n, set_u = pysolid.calc_solid_earth_tides_grid(
        sensing_start, geogrid_attrs, display=False, verbose=True
    )

    # Resample SET from geographical grid to radar grid
    # Generate the lat/lon arrays for the SET geogrid
    lat_geo_vec = (
        geogrid_attrs["Y_FIRST"]
        + np.arange(geogrid_attrs["LENGTH"]) * geogrid_attrs["Y_STEP"]
    )
    lon_geo_vec = (
        geogrid_attrs["X_FIRST"]
        + np.arange(geogrid_attrs["WIDTH"]) * geogrid_attrs["X_STEP"]
    )

    # Use scipy RGI to resample SET from geocoded to radar coordinates
    pts_src = (np.flipud(lat_geo_vec), lon_geo_vec)
    pts_dst = (lat_radar_grid.flatten(), lon_radar_grid.flatten())

    rdr_set_e, rdr_set_n, rdr_set_u = [
        _resample_set(set_enu, pts_src, pts_dst).reshape(lat_radar_grid.shape)
        for set_enu in [set_e, set_n, set_u]
    ]

    # Convert SET from ENU to range/azimuth coordinates
    set_rg, set_az = _enu2rgaz(
        radar_grid,
        orbit,
        ellipsoid,
        lon_radar_grid,
        lat_radar_grid,
        hgt_radar_grid,
        rdr_set_e,
        rdr_set_n,
        rdr_set_u,
        geo2rdr_params,
    )

    return set_rg, set_az


def _enu2rgaz(
    radargrid,
    orbit,
    ellipsoid,
    lon_arr,
    lat_arr,
    hgt_arr,
    e_arr,
    n_arr,
    u_arr,
    geo2rdr_params=None,
):
    """
    Convert ENU displacement into range / azimuth displacement,
    based on the idea mentioned in ETAD ATBD, available in the link below:
    https://sentinels.copernicus.eu/documents/247904/4629150/ETAD-DLR-DD-0008_Algorithm-Technical-Baseline-Document_2.3.pdf/5cb45b43-76dc-8dec-04ef-ca1252ace434?t=1680181574715 # noqa

    Algorithm description
    ---------------------
    For all lon / lat / height of the array;
    1. Calculate the ECEF coordinates before applying SET
    2. Calculate the unit vector of east / north / up directions of the point (i.e. ENU vectors)
    3. Scale the ENU vectors in 2 with ENU displacement to
       get the displacement in ECEF
    4. Add the vectors calculated in 3 into 1.
       This will be the ECEF coordinates after applying SET
    5. Convert 4 into lat / lon / hgt.
       This will be LLH coordinates after applying SET
    6. Calculate the radar coordinate before SET applied using `geo2rdr`
    7. Calculate the radar coordinate AFTER SET applied using `geo2rdr`
    8. Calculate the difference between (7) and (6),
       which will be the displacement in radargrid by SET

    Parameters
    ----------
    radargrid: isce3.product.RadarGridParameters
        Radargrid of the burst
    orbit: isce3.core.Orbit
        Orbit of the burst
    ellipsoid: isce3.core.Ellipsoid
        Ellipsoid definition
    lon_arr, lat_arr, hgt_arr: np.nadrray
        Arrays for longitude, latitude, and height.
        Units for longitude and latitude are degree; unit for height is meters.
    e_arr, n_arr, u_arr: np.ndarray
        Displacement in east, north, and up direction in meters
    geo2rdr_params: SimpleNameSpace
        Parameters for geo2rdr

    Returns
    -------
    rg_arr: np.ndarray
        Displacement in slant range direction in meters.
    az_arr: np.ndarray
        Displacement in azimuth direction in seconds.

    Notes
    -----
    When `geo2rdr_params` is not provided, then the iteration
    threshold and max # iterations are set to
    `1.0e-8` and `25` respectively.

    """
    if geo2rdr_params is None:
        # default threshold and # iteration for geo2rdr
        threshold = 1.0e-9
        maxiter = 25
    else:
        threshold = geo2rdr_params.threshold
        maxiter = geo2rdr_params.numiter

    shape_arr = lon_arr.shape
    rg_arr = np.zeros(shape_arr)
    az_arr = np.zeros(shape_arr)

    # Calculate the ENU vector in ECEF for each point.
    for i, (lon_deg, lat_deg, hgt) in enumerate(np.nditer([lon_arr, lat_arr, hgt_arr])):
        vec_e, vec_n, vec_u = get_enu_vec_from_lat_lon(lon_deg, lat_deg)

        llh_without_set = np.array([np.deg2rad(lon_deg), np.deg2rad(lat_deg), hgt])

        xyz_without_set = ellipsoid.lon_lat_to_xyz(llh_without_set)

        index_arr = np.unravel_index(i, lon_arr.shape)
        xyz_with_set = (
            xyz_without_set
            + vec_e * e_arr[index_arr]
            + vec_n * n_arr[index_arr]
            + vec_u * u_arr[index_arr]
        )
        llh_with_set = ellipsoid.xyz_to_lon_lat(xyz_with_set)

        try:
            aztime_without_set, slant_range_without_set = isce3.geometry.geo2rdr(
                llh_without_set,
                ellipsoid,
                orbit,
                isce3.core.LUT2d(),
                radargrid.wavelength,
                radargrid.lookside,
                threshold=threshold,
                maxiter=maxiter,
            )
        except RuntimeError:
            print(f"geo2rdr not SET FUBAR {i}")

        aztime_with_set, slant_range_with_set = isce3.geometry.geo2rdr(
            llh_with_set,
            ellipsoid,
            orbit,
            isce3.core.LUT2d(),
            radargrid.wavelength,
            radargrid.lookside,
            threshold=threshold,
            maxiter=maxiter,
        )

        rg_arr[index_arr] = slant_range_with_set - slant_range_without_set
        az_arr[index_arr] = aztime_with_set - aztime_without_set

    return rg_arr, az_arr


def _resample_set(geo_tide, pts_src, pts_dest):
    """
    Use scipy RegularGridInterpolator to resample geo_tide
    from a geographical to a radar grid

    Parameters
    ----------
    geo_tide: np.ndarray
        Tide displacement component on geographical grid
    pts_src: tuple of ndarray
        Points defining the source rectangular regular grid for resampling
    pts_dest: tuple of ndarray
        Points defining the destination grid for resampling
    Returns
    -------
    rdr_tide: np.ndarray
        Tide displacement component resampled on radar grid
    """

    # Flip tide displacement component to be consistent with flipped latitudes
    geo_tide = np.flipud(geo_tide)
    rgi_func = RegularGridInterpolator(
        pts_src, geo_tide, method="linear", bounds_error=False, fill_value=0
    )
    rdr_tide = rgi_func(pts_dest)
    return rdr_tide
