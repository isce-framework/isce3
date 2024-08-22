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
from isce3.core import DateTime, LLH, make_projection, xyz_to_enu
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

    # TODO: Implement flattening and change this argument's help string to fit.
    parser.add_argument(
        "-i", 
        "--input", 
        metavar="GSLC_PATH",
        dest="gslc_filename",
        type=Path,
        help=(
            "Input GSLC product path. WARNING: Works best with products that are not"
            " topographically flattened."
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
    x_pixel = x_rel/x_spacing
    y_pixel = y_rel/y_spacing

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
    velocity : np.ndarray
        The target's velocity vector in ECEF coordinates. Must be a vector of 3 floats.
    grid_spacing : tuple of 2 floats
        The x and y pixel spacing of the image.

    Returns
    -------
    float
        The heading angle, defined clockwise wrt the positive Y axis of the raster,
        adjusted by the spacing of the product raster, in radians from 0 to 2π.
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
    heading = np.pi / 2 - np.arctan2(enu[1] * -y_spacing, enu[0] * x_spacing)

    # Return the heading, wrapped from 0 to 2π.
    return heading if heading > 0 else heading + 2 * np.pi


def get_approx_platform_velocity(gslc: GSLC) -> np.ndarray:
    """
    Acquire the velocity of the platform at the midpoint of the source grid of a GSLC
    product.

    Parameters
    ----------
    gslc : GSLC
        The GSLC product.

    Returns
    -------
    np.ndarray of float
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


def analyze_gslc_point_targets_csv(
    gslc_filename: os.PathLike | str,
    output_file: os.PathLike | str | None,
    *,
    corner_reflector_csv: os.PathLike | str,
    freq: str | None,
    pol: str | None,
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
    grid_params = gslc.getGeoGridParameters(frequency=freq, polarization=pol)
    x_spacing = grid_params.spacing_x
    y_spacing = grid_params.spacing_y

    orbit = gslc.getOrbit()

    # Get the velocity vector of the platform.
    velocity = get_approx_platform_velocity(gslc=gslc)

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

    # If topographic flattening has been applied to the GSLC product,
    # warn the user.
    # TODO: Implement unflattening and remove this warning.
    if gslc.topographicFlatteningApplied:
        warnings.warn(
            "The input GSLC product is topographically flattened. Flattened"
            " products cannot be guaranteed to be basebanded; this may cause"
            " point target analysis to produce inaccurate results."
        )

    results = []

    for cr in corner_reflectors:
        # Get the coordinates, in pixel indices on the GSLC product, of the corner
        # reflector.
        j, i = get_llh_geo_grid_coords(llh=cr.llh, geogrid_params=grid_params)

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

        # TODO: Unflattening code will eventually go here.

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

        # Get the target's zero-Doppler elevation angle.
        _, elevation_angle = get_target_observation_time_and_elevation(
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
                "elevation_angle": elevation_angle,
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


def analyze_gslc_point_target_llh(
    gslc_filename: os.PathLike | str,
    output_file: os.PathLike | str | None,
    *,
    cr_llh: Iterable[float],
    freq: str | None,
    pol: str | None,
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

    # If topographic flattening has been applied to the GSLC product,
    # warn the user.
    # TODO: Implement unflattening and remove this warning.
    if gslc.topographicFlatteningApplied:
        warnings.warn(
            "The input GSLC product is topographically flattened. Flattened"
            " products cannot be guaranteed to be basebanded; this may cause"
            " point target analysis to produce inaccurate results."
        )

    # Get grid information
    grid_params = gslc.getGeoGridParameters(frequency=freq, polarization=pol)
    x_spacing = grid_params.spacing_x
    y_spacing = grid_params.spacing_y

    velocity = get_approx_platform_velocity(gslc=gslc)

    cr_llh_obj = LLH(longitude=cr_llh[0], latitude=cr_llh[1], height=height)

    # Get the coordinates, in pixel indices on the GSLC product, of the corner
    # reflector.
    pixel_coord = get_llh_geo_grid_coords(llh=cr_llh_obj, geogrid_params=grid_params)
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

    # TODO: Unflattening code will eventually go here.

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
        if hasattr(inputs, "in_rads"):
            warnings.warn(
                "--in-rads is not used with corner_reflector_csv; This argument will be"
                "ignored."
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
