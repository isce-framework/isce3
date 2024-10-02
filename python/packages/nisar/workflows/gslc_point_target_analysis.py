"""
Analyze point targets in a GSLC HDF5 file
"""
from __future__ import annotations

import os
import traceback
import warnings

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import shapely

from isce3.cal import (
    get_crs_in_polygon,
    get_target_observation_time_and_elevation,
)
from isce3.cal.point_target_info import (
    analyze_point_target_chip,
    generate_chip_on_slc,
)
from isce3.core import (
    DateTime,
    Ellipsoid,
    LLH,
    LUT2d,
    make_projection,
    Orbit,
    ProjectionBase,
    xyz_to_enu,
)
from isce3.geometry import (
    DEMInterpolator,
    dem_raster_to_interpolator,
    geo2rdr_bracket,
)
from isce3.io import Raster
from isce3.product import GeoGridParameters, RadarGridParameters

import nisar
from nisar.cal import est_cr_az_mid_swath_from_slc, filter_crs_per_az_heading
from nisar.products.readers import GSLC
from nisar.workflows.point_target_analysis import (
    add_pta_args,
    check_slc_freq_pols,
    CustomJSONEncoder,
    get_corner_reflectors_from_csv,
    to_json,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

desc = __doc__

def cmd_line_parse():
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "-i",
        "--input",
        metavar="GSLC_PATH",
        dest="gslc_filename",
        type=Path,
        help=(
            "Input GSLC product path."
        )
    )
    # Add a required XOR group for the `corner_reflector_csv` and `cr_llh` parameters
    # so that one or the other (but not both) must be specified.
    cr_group = parser.add_mutually_exclusive_group(required=True)
    cr_group.add_argument(
        "--csv",
        type=Path,
        dest="corner_reflector_csv",
        default=argparse.SUPPRESS,
        help=(
            "A CSV file containing corner reflector data, in the format defined by the"
            " --format flag. Required if -c (--LLH) is not specified."
        ),
    )
    cr_group.add_argument(
        "-c",
        "--LLH",
        nargs=3,
        dest="cr_llh",
        type=float,
        default=argparse.SUPPRESS,
        help=(
            "Geodetic coordinates (longitude in degrees, latitude in degrees,"
            " height above ellipsoid in meters). Required if --csv is not specified."
        ),
    )
    parser.add_argument(
        "--dem-path",
        type=Path,
        default=None,
        help=(
            "The path to the DEM file to unflatten with. Provides more accurate"
            " analysis of topographically flattened products. Unnecessary if the input"
            " GSLC product is not topographically flattened. If None, unflattening will"
            " not be performed."
        ),
    )
    parser.add_argument(
        "--in-rads",
        action="store_true",
        help="Use if your --llh input is in radians. Ignored if --csv is specified.",
    )
    parser.add_argument(
        "--format",
        type=str,
        dest="cr_format",
        choices=["nisar", "uavsar"],
        default="nisar",
        help=(
            "The corner reflector CSV file format. If 'nisar', the CSV file should be"
            " in the format described by the NISAR Corner Reflector Software Interface"
            " Specification (SIS) document, JPL D-107698. If 'uavsar', the CSV file is"
            " expected to be in the format used by the UAVSAR Rosamond Corner Reflector"
            " Array (https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl). This flag is"
            " ignored if --csv is not specified."
        ),
    )
    add_pta_args(parser, predict_null_option=False)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Display output PTA result plots. Ignored if --csv is specified.",
    )

    return parser.parse_args()


def get_llh_geo_grid_coords(
    llh: LLH,
    geogrid_params: GeoGridParameters,
) -> tuple[float, float]:
    """
    Get the coordinates of a lon/lat/height, in pixels, on a given geo grid.

    Parameters
    ----------
    llh : isce3.core.LLH
        The lon/lat/height object.
    geogrid_params : isce3.product.GeoGridParameters
        A description of the geo grid.

    Returns
    -------
    x_pixel, y_pixel : float
        The pixel coordinates that the corner reflector is expected to sit at on the
        geo grid.
    """
    x_origin_coord = geogrid_params.start_x
    y_origin_coord = geogrid_params.start_y
    x_spacing = geogrid_params.spacing_x
    y_spacing = geogrid_params.spacing_y
    epsg = geogrid_params.epsg

    # Project the LLH into the coordinates of the product.
    projection = make_projection(epsg)
    xyz = projection.forward(llh.to_vec3())

    # Get the position relative to the origin point of the raster, in the units of
    # the projection.
    x_rel = xyz[0] - x_origin_coord
    y_rel = xyz[1] - y_origin_coord

    # Adjust this position by the X and Y spacing values in order to get the expected
    # pixel location of the LLH.
    # This position is shifted by half a pixel in each direction, because the location
    # of pixels in a geocoded grid is at the center of the pixel, not the vertex.
    #
    #           o-------|
    #           |       |
    #           |   x   |
    #           |       |
    #           |-------|
    #
    # As a demonstration, in the diagram above, the value at position `o` in a geocoded
    # raster is representative of geographic location `x` at the center of the pixel.
    x_pixel = x_rel / x_spacing - 0.5
    y_pixel = y_rel / y_spacing - 0.5

    return (x_pixel, y_pixel)


def get_relative_heading_on_geo_grid(
    latitude: float,
    longitude: float,
    velocity: np.ndarray,
    grid_spacing: tuple[float, float],
) -> float:
    """
    Get the heading angle of a moving target at a given lon/lat position, skewed
    to the aspect ratio of a north-up, east-left image.

    Parameters
    ----------
    latitude, longitude : float
        The geodetic latitude and longitude of the target, in radians.
    velocity : numpy.ndarray
        The target's velocity vector in ECEF coordinates. Must be a vector of 3 floats.
    grid_spacing : tuple of 2 floats
        The x and y pixel spacing of the image.

    Returns
    -------
    float
        The heading angle, defined clockwise wrt the positive Y axis of the raster,
        adjusted by the spacing of the product raster, in radians from :math:`0` to
        :math:`2\pi`.
    """
    x_spacing, y_spacing = grid_spacing

    if len(velocity) != 3:
        raise ValueError(
            f"Velocity given with length {len(velocity)}. Must be a vector of 3 floats."
        )

    # This gives us a rotation matrix that lets us convert ECEF vector into ENU
    # vector by matrix multiplication
    ecef_to_enu_mat = xyz_to_enu(latitude, longitude)

    # Use the matrix to get an ENU vector
    enu = np.matmul(ecef_to_enu_mat, velocity)

    # Heading angle, defined clockwise wrt the positive Y axis of the raster,
    # adjusted by the spacing of the product raster, in radians.
    # If the GSLC image is North-up East-right, the `y_spacing` will be negative and
    # `x_spacing` will be positive. The ENU heading vector has the same sign convention
    # in the X direction but the opposite sign convention in the Y direction. Flip the
    # sign of `y_spacing` so that the skewed heading vector is still North-positive.
    heading = np.pi / 2 - np.arctan2(enu[1] * -y_spacing, enu[0] * x_spacing)

    # Return the heading, wrapped from 0 to 2π.
    return heading if heading > 0 else heading + 2 * np.pi


def get_midpoint_platform_velocity(gslc: GSLC) -> np.ndarray:
    """
    Acquire the velocity of the platform at the midpoint of the source grid of a GSLC
    product.

    Parameters
    ----------
    gslc : nisar.products.readers.GSLC
        The GSLC product.

    Returns
    -------
    numpy.ndarray of float
        The velocity of the platform at the midpoint of the product orbit.
    """
    orbit = gslc.getOrbit()
    ref_epoch = orbit.reference_epoch

    # The start time and end time are the number of seconds between the zero doppler
    # start and end datetimes and the reference epoch
    start_datetime : DateTime = gslc.identification.zdStartTime
    end_datetime : DateTime = gslc.identification.zdEndTime
    start_time = (start_datetime - ref_epoch).total_seconds()
    end_time = (end_datetime - ref_epoch).total_seconds()

    mid_time = start_time + (end_time - start_time) / 2

    # To get the orbit velocity, get the midpoint time and interpolate the orbit at
    # that time.
    _, velocity = orbit.interpolate(mid_time)

    return velocity


def geocoded_ij_to_lonlat(
    geogrid: GeoGridParameters, i: float, j: float
) -> tuple[float, float]:
    """
    Get the lon/lat position of the i,j position on a geogrid.

    Parameters
    ----------
    geogrid : isce3.product.GeoGridParameters
        The parameters of the geogrid.
    i : float
        The i position (Corresponds to the y axis on the grid).
    j : float
        The j position (Corresponds to the x axis on the grid).

    Returns
    -------
    lon, lat : float
        The longitude and latitude of the pixel center location, in radians.
    """
    x = geogrid.start_x + j * geogrid.spacing_x + geogrid.spacing_x / 2.0
    y = geogrid.start_y + i * geogrid.spacing_y + geogrid.spacing_y / 2.0

    xy = [x, y, 0.0]

    projection: ProjectionBase = make_projection(geogrid.epsg)

    # Transform the coordinates
    llh = projection.inverse(xy)

    return llh[0], llh[1]


def get_geogrid_heights(
    geogrid: GeoGridParameters,
    interpolator: DEMInterpolator,
) -> np.ndarray:
    """
    Given information about a region of geocoded data, find the heights of each
    position in the geogrid.

    Parameters
    ----------
    geogrid : isce3.product.GeoGridParameters
        The parameters of the geo grid to calculate heights over.
    interpolator : isce3.geometry.DEMInterpolator
        The DEM interpolator to get heights on.

    Returns
    -------
    numpy.ndarray of numpy.float64
        The heights of each position on the geocoded chip.
    """
    length = geogrid.length
    width = geogrid.width
    heights = np.full((length, width), fill_value=np.nan, dtype=np.float64)

    # Step over each position on the output grid and get the height of that position.
    for i in range(length):
        for j in range(width):
            # Get the lon/lat of the corresponding geocoded position
            lon, lat = geocoded_ij_to_lonlat(i=i, j=j, geogrid=geogrid)

            # Translate this lon/lat position into a height on the DEM
            heights[i, j] = interpolator.interpolate_lonlat(lon, lat)
    return heights


def get_geogrid_slant_ranges(
    height_data: np.ndarray,
    geogrid: GeoGridParameters,
    lookside: str,
    ellipsoid: Ellipsoid,
    orbit: Orbit,
) -> np.ndarray:
    """
    Acquire the slant ranges of all points on a geogrid.

    Parameters
    ----------
    height_data : numpy.ndarray of numpy.float64
        The heights at the position of each pixel of the geogrid, in meters.
    geogrid_params : isce3.product.GeoGridParameters
        The parameters of the geo grid to calculate slant ranges for.
    lookside : str
        The look side of the radar, either 'left' or 'right'.
    ellipsoid : isce3.core.Ellipsoid
        The ellipsoid of the DEM.
    orbit : isce3.core.Orbit
        The orbit of the satellite over the region of the geogrid.

    Returns
    -------
    numpy.ndarray of numpy.float64
        The slant range of each pixel of the geogrid, in meters.
    """
    length = geogrid.length
    width = geogrid.width

    if height_data.shape != (length, width):
        raise ValueError(
            "height_data shape differs from shape of geogrid."
            f" height_data: {height_data.shape}; geogrid: ({length}, {width})."
        )

    slant_ranges = np.full((length, width), fill_value=np.nan, dtype=np.float64)

    for i in range(length):
        for j in range(width):
            lon, lat = geocoded_ij_to_lonlat(i=i, j=j, geogrid=geogrid)
            height = height_data[i, j]

            llh = np.asarray([lon, lat, height])

            # Convert the llh to ECEF
            xyz_ecef = ellipsoid.lon_lat_to_xyz(llh)

            try:
                # Get the slant range using geo2rdr.
                # No wavelength is needed since a 0-doppler geometry will be used.
                _, slant_range = geo2rdr_bracket(
                    xyz=xyz_ecef,
                    orbit=orbit,
                    doppler=LUT2d(),
                    wavelength=1,
                    side=lookside,
                )
            except RuntimeError:
                slant_range = np.nan

            slant_ranges[i, j] = slant_range
    return slant_ranges


def get_geogrid_flattening_phase(
    dem_interpolator: DEMInterpolator,
    geogrid_params: GeoGridParameters,
    wavelength: float,
    lookside: str,
    orbit: Orbit,
) -> np.ndarray:
    """
    Get the topological flattening phase of a geogrid.

    Parameters
    ----------
    dem_interpolator : isce3.geometry.DEMInterpolator
        The DEM interpolator to use for estimating heights on the grid.
    geogrid_params : isce3.product.GeoGridParameters
        The parameters of the geo grid, sliced to the desired region.
    wavelength : float
        The wavelength of the radar, in meters.
    lookside : str
        The look side of the radar, either 'left' or 'right'.
    orbit : isce3.core.Orbit
        The orbit of the satellite over the region of the geogrid.

    Returns
    -------
    numpy.ndarray of numpy.float64
        The topological flattening phase of the grid, in radians.
    """
    height_data = get_geogrid_heights(
        geogrid=geogrid_params,
        interpolator=dem_interpolator,
    )

    slant_ranges = get_geogrid_slant_ranges(
        height_data=height_data,
        geogrid=geogrid_params,
        lookside=lookside,
        ellipsoid=dem_interpolator.ellipsoid,
        orbit=orbit,
    )

    # The flattening phase (in radians) is calculated as:
    # 4 * pi * slant range / wavelength; this is the same as the flattening phase
    # calculated by the C++ GeocodeSlc flattening algorithm.
    return 4 * (np.pi / wavelength) * slant_ranges


def analyze_gslc_point_targets_csv(
    gslc_filename: os.PathLike | str,
    output_file: os.PathLike | str | None,
    *,
    corner_reflector_csv: os.PathLike | str,
    freq: str | None,
    pol: str | None,
    dem_path: os.PathLike | str | None = None,
    nchip: int = 64,
    upsample_factor: int = 32,
    peak_find_domain: str = 'time',
    num_sidelobes: int = 10,
    cuts: bool = False,
    cr_format: str = "nisar",
):
    """
    Perform point target analysis on point targets in a NISAR GSLC product.

    Parameters
    ----------
    gslc_filename : path-like
        The filepath to the GSLC product.
    output_file : path-like or None
        The path to the JSON file where metrics will be output to, or None.
        If None, will print JSON outputs to the terminal.
    corner_reflector_csv : path-like
        The filepath to the corner reflector file.
    freq : {'A', 'B'} or None
        The frequency sub-band of the data. If None, defaults to the science band in the
        GSLC product ('A' if available, otherwise 'B').
    pol : {'HH', 'HV', 'VH', 'VV', 'LH', 'LV', 'RH', 'RV'} or None
        The transmit and receive polarization of the data. If None, defaults to the
        first co-polarization or compact polarization channel found in the specified
        band from the list ['HH', 'VV', 'LH', 'LV', 'RH', 'RV'].
    dem_path : path-like or None, optional
        The path to the DEM file, if one is needed for unflattening. If None,
        flattened GSLC products will not be unflattened for processing.
        Defaults to None.
    nchip : int, optional
        The width, in pixels, of the square block of image data centered around the
        target position to extract for oversampling and peak finding. Must be >= 1.
        Defaults to 64.
    upsample_factor : int, optional
        The upsampling ratio. Must be >= 1. Defaults to 32.
    peak_find_domain : {'time', 'freq'}
        Option controlling how the target peak position is estimated.
        Defaults to 'time'.

        'time':
          The peak location is found in the time domain by detecting the maximum value
          within a square block of image data around the expected target location. The
          signal data is upsampled to improve precision.

        'freq':
          The peak location is found by estimating the phase ramp in the frequency
          domain. This mode is useful when the target is well-focused, has high SNR, and
          is the only target in the neighborhood (often the case in point target
          simulations).
    num_sidelobes : int, optional
        The number of sidelobes, including the main lobe, to use for computing the
        integrated sidelobe ratio (ISLR). Must be > 1. Defaults to 10.
    cuts : bool, optional
        Whether to include range & azimuth cuts through the peak in the results.
        Defaults to False.
    cr_format : str, optional
        The format of the corner reflector file. Defaults to "nisar".
    """
    gslc = GSLC(hdf5file=gslc_filename)
    slc_data = gslc.getSlcDataset(frequency=freq, polarization=pol)

    # Check and acquire valid frequency and polarization values based on the given
    # inputs.
    freq, pol = check_slc_freq_pols(slc=gslc, freq=freq, pol=pol)

    # Acquire radar grid parameters from the source product
    source_radar_params: RadarGridParameters = gslc.getSourceRadarGridParameters(freq)
    wavelength = source_radar_params.wavelength
    look_side = source_radar_params.lookside

    # Get grid information
    geogrid = gslc.getGeoGridParameters(frequency=freq, polarization=pol)
    x_spacing = geogrid.spacing_x
    y_spacing = geogrid.spacing_y

    orbit = gslc.getOrbit()
    
    if dem_path is not None:
        dem_raster = Raster(os.fspath(dem_path))
        dem_interpolator = dem_raster_to_interpolator(dem_raster, geogrid)
    else:
        # If the product is topographically flattened and no DEM has been provided to
        # unflatten it, raise a warning to the user.
        if gslc.topographicFlatteningApplied:
            warnings.warn(
                "Flattened GSLC provided with no associated DEM. Point target analysis"
                " will unflatten using a zero-height DEM; outputs may be less"
                " accurate than if a DEM were provided."
            )
        dem_interpolator = DEMInterpolator()

    # Get the velocity vector of the platform.
    velocity = get_midpoint_platform_velocity(gslc=gslc)

    # Get corner reflector data.
    corner_reflectors = get_corner_reflectors_from_csv(
        filename=corner_reflector_csv,
        format=cr_format,
        observation_date=gslc.identification.zdStartTime,
    )

    # Filter out CRs outside the GSLC bounding polygon.
    polygon = shapely.from_wkt(gslc.identification.boundingPolygon)
    corner_reflectors = get_crs_in_polygon(corner_reflectors, polygon)

    cr_optimum_heading = est_cr_az_mid_swath_from_slc(slc=gslc)
    corner_reflectors = filter_crs_per_az_heading(corner_reflectors, cr_optimum_heading)

    results = []

    for cr in corner_reflectors:
        # Get the coordinates, in pixel indices on the GSLC product, of the corner
        # reflector.
        j, i = get_llh_geo_grid_coords(llh=cr.llh, geogrid_params=geogrid)

        # Get the heading of the satellite on the geocoded grid.
        cr_heading = get_relative_heading_on_geo_grid(
            cr.llh.latitude,
            cr.llh.longitude,
            velocity,
            grid_spacing=(x_spacing, y_spacing),
        )

        # Get the chip (i.e. a square region of data surrounding the corner reflector
        # with edge length = nchip)
        chip, min_i, min_j = generate_chip_on_slc(slc_data, i, j, chipsize=nchip)

        # If topographic flattening has been applied to the GSLC product,
        # remove the topographic flattening phase.
        if gslc.topographicFlatteningApplied:
            flattening_phase = get_geogrid_flattening_phase(
                dem_interpolator=dem_interpolator,
                geogrid_params=geogrid[min_i:min_i+nchip, min_j:min_j+nchip],
                wavelength=source_radar_params.wavelength,
                lookside=source_radar_params.lookside,
                orbit=orbit,
            )

            # Convert the calculated flattening phase into unit phasors.
            flattening_complex = np.cos(flattening_phase) + 1.j*np.sin(flattening_phase)
            chip *= np.conjugate(flattening_complex)

        try:
            # Pass the chip and supporting information along to the point target.
            perf_dict, _ = analyze_point_target_chip(
                chip=chip,
                i_pos=i,
                j_pos=j,
                nov=upsample_factor,
                chip_min_i=min_i,
                chip_min_j=min_j,
                cuts=cuts,
                plot=False,
                geo_heading=cr_heading,
                pixel_spacing=(y_spacing, x_spacing),
                num_sidelobes=num_sidelobes,
                predict_null=False,
                shift_domain=peak_find_domain,
            )
        except Exception:
            errmsg = traceback.format_exc()
            warnings.warn(
                f"an exception occurred while processing corner reflector {cr.id!r}:"
                f"\n\n{errmsg}",
                RuntimeWarning,
            )
            continue

        attitude = gslc.getAttitude()

        # Get the target's zero-Doppler UTC time and elevation angle.
        az_datetime, el_angle = get_target_observation_time_and_elevation(
            target_llh=cr.llh,
            orbit=orbit,
            attitude=attitude,
            wavelength=wavelength,
            look_side=look_side,
        )

        # Add some additional metadata.
        perf_dict.update(
            {
                "id": cr.id,
                "frequency": freq,
                "polarization": pol,
                "elevation_angle": el_angle,
                "timestamp": az_datetime,
            }
        )

        # Add NISAR-specific corner reflector metadata, if available.
        if isinstance(cr, nisar.cal.CornerReflector):
            perf_dict.update(
                {
                    "survey_date": cr.survey_date,
                    "validity": cr.validity,
                    "velocity": cr.velocity,
                }
            )

        results.append(perf_dict)

    to_json(results, output_file, encoder=CustomJSONEncoder)

    return results


def analyze_gslc_point_target_llh(
    gslc_filename: os.PathLike | str,
    output_file: os.PathLike | str | None,
    *,
    cr_llh: Iterable[float],
    freq: str | None,
    pol: str | None,
    dem_path: os.PathLike | str | None = None,
    nchip: int = 64,
    upsample_factor: int = 32,
    peak_find_domain: str = 'time',
    num_sidelobes: int = 10,
    plots: bool = False,
    cuts: bool = False,
    in_rads: bool = False,
):
    """
    Run point target analysis for a single point target on GSLC.

    Parameters:
    ------------
    gslc_filename : path-like
        The filepath to the GSLC product.
    output_file : path-like or None
        The path to the JSON file where metrics will be output to, or None.
        If None, will print JSON outputs to the terminal.
    cr_llh : Iterable of float
        The target's geodetic longitude, latitude, and (optionally) height. An iterable
        containing 2 or 3 floats in (lon, lat[, height]) order. Longitude and latitude
        should be provided in degrees (or radians if `in_rads` was true). Height is in
        meters above the reference ellipsoid. If height is not provided, the target is
        assumed to be on the ellipsoid.
    freq : {'A', 'B'} or None
        The frequency sub-band of the data. If None, defaults to the science band in the
        GSLC product ('A' if available, otherwise 'B').
    pol : {'HH', 'HV', 'VH', 'VV', 'LH', 'LV', 'RH', 'RV'} or None
        The transmit and receive polarization of the data. If None, defaults to the
        first co-polarization or compact polarization channel found in the specified
        band from the list ['HH', 'VV', 'LH', 'LV', 'RH', 'RV'].
    dem_path : path-like or None, optional
        The path to the DEM file, if one is needed for unflattening. If None,
        flattened GSLC products will not be unflattened for processing.
        Defaults to None.
    nchip : int, optional
        The width, in pixels, of the square block of image data centered around the
        target position to extract for oversampling and peak finding. Must be >= 1.
        Defaults to 64.
    upsample_factor : int, optional
        The upsampling ratio. Must be >= 1. Defaults to 32.
    peak_find_domain : {'time', 'freq'}
        Option controlling how the target peak position is estimated.
        Defaults to 'time'.

        'time':
          The peak location is found in the time domain by detecting the maximum value
          within a square block of image data around the expected target location. The
          signal data is upsampled to improve precision.

        'freq':
          The peak location is found by estimating the phase ramp in the frequency
          domain. This mode is useful when the target is well-focused, has high SNR, and
          is the only target in the neighborhood (often the case in point target
          simulations).
    num_sidelobes : int, optional
        The number of sidelobes, including the main lobe, to use for computing the
        integrated sidelobe ratio (ISLR). Must be > 1. Defaults to 10.
    plots : bool, optional
        If True, display a plots with point target analysis results. Defaults to False.
    cuts : bool, optional
        Whether to include range & azimuth cuts through the peak in the results.
        Defaults to False.
    in_rads : bool, optional
        True if `cr_llh` is in radians, False if it is in degrees. Defaults to False.
    """
    if len(cr_llh) < 2:
        raise ValueError("cr_llh must have at least a longitude and latitude.")
    if len(cr_llh) > 3:
        raise ValueError("cr_llh is too long - must be no more than 3 values.")
    if len(cr_llh) == 2:
        height = 0
    else:
        height = cr_llh[2]

    # Open GSLC data
    gslc = GSLC(hdf5file=gslc_filename)

    # Check and acquire valid frequency and polarization values based on the given
    # inputs.
    freq, pol = check_slc_freq_pols(slc=gslc, freq=freq, pol=pol)

    slc_data = gslc.getSlcDataset(frequency=freq, polarization=pol)

    if not in_rads:
        cr_llh = list(cr_llh)
        lon_deg, lat_deg = cr_llh[:2]
        lonlat_rads = np.deg2rad([lon_deg, lat_deg])
        cr_llh = [lonlat_rads[0], lonlat_rads[1], height]

    # Acquire radar grid parameters from the source product
    source_radar_params: RadarGridParameters = gslc.getSourceRadarGridParameters(freq)

    # Get grid information
    geogrid = gslc.getGeoGridParameters(frequency=freq, polarization=pol)
    x_spacing = geogrid.spacing_x
    y_spacing = geogrid.spacing_y

    orbit = gslc.getOrbit()
    
    if dem_path is not None:
        dem_raster = Raster(os.fspath(dem_path))
        dem_interpolator = dem_raster_to_interpolator(dem_raster, geogrid)
    else:
        # If the product is topographically flattened and no DEM has been provided to
        # unflatten it, raise a warning to the user.
        if gslc.topographicFlatteningApplied:
            warnings.warn(
                "Flattened GSLC provided with no associated DEM. Point target analysis"
                " will unflatten using a zero-height DEM; outputs may be less"
                " accurate than if a DEM were provided."
            )
        dem_interpolator = DEMInterpolator()

    # Get the velocity vector of the platform.
    velocity = get_midpoint_platform_velocity(gslc=gslc)

    cr_llh_obj = LLH(longitude=cr_llh[0], latitude=cr_llh[1], height=height)

    # Get the coordinates, in pixel indices on the GSLC product, of the corner
    # reflector.
    pixel_coord = get_llh_geo_grid_coords(llh=cr_llh_obj, geogrid_params=geogrid)
    j, i = pixel_coord

    # Get the heading of the satellite on the geocoded grid.
    cr_heading = get_relative_heading_on_geo_grid(
        longitude=cr_llh_obj.longitude,
        latitude=cr_llh_obj.latitude,
        velocity=velocity,
        grid_spacing=(x_spacing, y_spacing),
    )

    # Get the chip (i.e. a square region of data surrounding the corner reflector
    # with edge length = nchip)
    chip, min_i, min_j = generate_chip_on_slc(slc_data, i, j, chipsize=nchip)

    # If topographic flattening has been applied to the GSLC product,
    # remove the topographic flattening phase.
    if gslc.topographicFlatteningApplied:
        flattening_phase = get_geogrid_flattening_phase(
            dem_interpolator=dem_interpolator,
            geogrid_params=geogrid[min_i:min_i+nchip, min_j:min_j+nchip],
            wavelength=source_radar_params.wavelength,
            lookside=source_radar_params.lookside,
            orbit=orbit,
        )

        # Convert the calculated flattening phase into unit phasors.
        flattening_complex = np.cos(flattening_phase) + 1.j*np.sin(flattening_phase)
        chip *= np.conjugate(flattening_complex)

    try:
        # Pass the chip and supporting information along to the point target.
        perf_dict, _ = analyze_point_target_chip(
            chip=chip,
            i_pos=i,
            j_pos=j,
            nov=upsample_factor,
            chip_min_i=min_i,
            chip_min_j=min_j,
            cuts=cuts,
            plot=plots,
            geo_heading=cr_heading,
            pixel_spacing=(y_spacing, x_spacing),
            num_sidelobes=num_sidelobes,
            predict_null=False,
            shift_domain=peak_find_domain,
        )
    except Exception as err:
        errmsg = traceback.format_exc()
        raise RuntimeError(
            "an exception occurred while processing the corner reflector."
        ) from err

    # Write dictionary content to a json file if output is requested by user
    to_json(perf_dict, output_file, encoder=CustomJSONEncoder)

    if plots and plt is not None:
        plt.show()

    return perf_dict


if __name__ == '__main__':
    inputs = cmd_line_parse()

    if hasattr(inputs, "corner_reflector_csv"):
        if inputs.in_rads:
            warnings.warn(
                "--in-rads is not used with corner_reflector_csv; This argument will "
                "be ignored."
            )
        del inputs.in_rads
        del inputs.plots
        analyze_gslc_point_targets_csv(**vars(inputs))
    elif hasattr(inputs, "cr_llh"):
        del inputs.format
        analyze_gslc_point_target_llh(**vars(inputs))
    else:
        # Should be unreachable.
        assert False, "invalid arguments to gslc_point_target_analysis"
