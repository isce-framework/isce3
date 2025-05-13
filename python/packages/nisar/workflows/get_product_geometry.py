#!/usr/bin/env python3

import os
import warnings
import argparse
from osgeo import gdal
import isce3
from nisar.products.readers import open_product
import numpy as np
import journal


def get_parser():
    '''
    Command line parser.
    '''
    descr = 'Get product geometry'
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument(type=str,
                        dest='input_file',
                        help='Input NISAR L1 or L2 file')

    parser.add_argument('--dem',
                        '--dem-file',
                        dest='dem_file',
                        required=True,
                        type=str,
                        help='Reference DEM file')

    parser.add_argument('--od',
                        '--output-dir',
                        dest='output_dir',
                        type=str,
                        default='.',
                        help='Output directory')

    parser.add_argument('--epsg',
                        action='store',
                        dest='epsg',
                        type=int,
                        default=None,
                        help='EPSG code for output coordinate X and Y'
                        ' (only applicable if the input'
                        ' is a NISAR L1 product). Default: same as DEM.')

    parser.add_argument('--frequency',
                        '--freq',
                        type=str,
                        default='A',
                        dest='frequency',
                        choices=['A', 'B'],
                        help='Frequency band: "A" or "B"')

    parser.add_argument('--dem-interp-method',
                        dest='dem_interp_method',
                        type=str,
                        choices=['SINC', 'BILINEAR', 'BICUBIC', 'NEAREST',
                                 'BIQUINTIC'],
                        help='DEM interpolation method. Options:'
                        ' "SINC", "BILINEAR", "BICUBIC", "NEAREST", and'
                        ' "BIQUINTIC"')

    parser.add_argument('--threshold-geo2rdr',
                        '--geo2rdr-threshold',
                        type=float,
                        dest='threshold_geo2rdr',
                        help='Convergence threshold for geo2rdr')

    parser.add_argument('--num-iter-geo2rdr',
                        '--geo2rdr-num-iter',
                        type=int,
                        dest='num_iter_geo2rdr',
                        help='Maximum number of iterations for geo2rdr')

    parser.add_argument('--delta-range-geo2rdr',
                        '--geo2rdr-delta-range',
                        type=float,
                        dest='delta_range_geo2rdr',
                        help='Delta range for geo2rdr')

    parser.add_argument('--threshold-rdr2geo',
                        '--rdr2geo-threshold',
                        type=float,
                        dest='threshold_rdr2geo',
                        help='Convergence threshold for rdr2geo')

    parser.add_argument('--num-iter-rdr2geo',
                        '--rdr2geo-num-iter',
                        type=int,
                        dest='num_iter_rdr2geo',
                        help='Maximum number of iterations for rdr2geo')

    parser.add_argument('--extra-iter-rdr2geo',
                        '--rdr2geo-num-extra',
                        type=int,
                        dest='extra_iter_rdr2geo',
                        help='Additional number of iterations for rdr2geo')

    parser.add_argument('--out-interpolated-dem',
                        action='store_true',
                        dest='flag_interpolated_dem',
                        help='Save interpolated DEM')

    parser.add_argument('--out-x',
                        '--out-coordinate-x',
                        action='store_true',
                        dest='flag_coordinate_x',
                        help='Save coordinate X (only applicable if the input'
                        ' is a NISAR L1 product)')

    parser.add_argument('--out-y',
                        '--out-coordinate-y',
                        action='store_true',
                        dest='flag_coordinate_y',
                        help='Save coordinate Y (only applicable if the input'
                        ' is a NISAR L1 product)')

    parser.add_argument('--out-slant-range',
                        action='store_true',
                        dest='flag_slant_range',
                        help='Save slant-range (only applicable if the input'
                        ' is a NISAR L2 product)')

    parser.add_argument('--out-azimuth-time',
                        '--out-az-time',
                        action='store_true',
                        dest='flag_azimuth_time',
                        help='Save azimuth time (only applicable if the input'
                        ' is a NISAR L2 product)')

    parser.add_argument('--out-inc-angle',
                        '--out-incidence-angle',
                        action='store_true',
                        dest='flag_incidence_angle',
                        help='Save interpolated DEM')

    parser.add_argument('--out-line-of-sight',
                        '--out-los',
                        action='store_true',
                        dest='flag_los',
                        help='Save line-of-sight unit vector')

    parser.add_argument('--out-along-track',
                        action='store_true',
                        dest='flag_along_track',
                        help='Save along-track unit vector')

    parser.add_argument('--out-elevation-angle',
                        action='store_true',
                        dest='flag_elevation_angle',
                        help='Save elevation angle')

    parser.add_argument('--out-ground-track-velocity',
                        action='store_true',
                        dest='flag_ground_track_velocity',
                        help='Save ground track velocity')

    parser.add_argument('--out-local-inc-angle',
                        '--out-local-incidence-angle',
                        action='store_true',
                        dest='flag_local_incidence_angle',
                        help='Save local-incidence angle (only implemented for'
                        ' NISAR L2 products)')

    parser.add_argument('--out-projection-angle',
                        action='store_true',
                        dest='flag_projection_angle',
                        help='Save projection angle (only implemented for'
                        ' NISAR L2 products)')

    parser.add_argument('--simulated-radar-brightness',
                        action='store_true',
                        dest='flag_simulated_radar_brightness',
                        help='Save simulated radar brightness (only'
                        ' implemented for NISAR L2 products)')

    return parser.parse_args()


def run(args):
    '''
    run main method
    '''
    # Get NISAR product
    nisar_product_obj = open_product(args.input_file)
    if nisar_product_obj.getProductLevel() == 'L2':
        get_radar_grid(nisar_product_obj, args)
    else:
        get_geolocation_grid(nisar_product_obj, args)


def get_radar_grid(nisar_product_obj, args):
    '''
    get radar grid for L2 products
    '''
    frequency_str = args.frequency

    orbit = nisar_product_obj.getOrbit()

    # Get GeoGridProduct obj and lookside
    try:
        geogrid_product_obj = nisar_product_obj.getGeoGridProduct()
    except AttributeError:
        error_message = ('ERROR get_product_geometry.py does not support'
                         f' product type "{nisar_product_obj.productType}".')
        raise NotImplementedError(error_message)

    lookside = geogrid_product_obj.lookside

    # Get Grid obj, GeoGrid obj, and wavelength
    grid_obj = nisar_product_obj.getGridMetadata(frequency_str)
    geogrid_obj = grid_obj.geogrid
    wavelength = grid_obj.wavelength

    # Get grid Doppler (zero-Doppler) and native Doppler LUTs
    grid_doppler = isce3.core.LUT2d()

    # TODO: Fix/remove try/except statements below
    # Causes for error:
    # (1) L2 products currently don't have a Doppler centroid LUT
    # (2) Once implemented, the Doppler centroid LUT will be
    # provided over map coordinates to follow the products'
    # specification. This will break the method
    # getDopplerCentroid() below
    #
    # The code below catches the case error caused by (1)
    # but it does not handle (2)
    try:
        native_doppler = nisar_product_obj.getDopplerCentroid()
        native_doppler.bounds_error = False
    except KeyError as e:
        warnings.warn(str(e))
        native_doppler = isce3.core.LUT2d()

    nbands = 1
    shape = [nbands, geogrid_obj.length, geogrid_obj.width]
    if args.output_dir and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    dem_raster = isce3.io.Raster(args.dem_file)

    output_file_list = []
    output_obj_list = []

    flag_all = (not args.flag_interpolated_dem and
                not args.flag_slant_range and
                not args.flag_azimuth_time and
                not args.flag_incidence_angle and
                not args.flag_los and
                not args.flag_along_track and
                not args.flag_elevation_angle and
                not args.flag_ground_track_velocity and
                not args.flag_local_incidence_angle and
                not args.flag_projection_angle and
                not args.flag_simulated_radar_brightness)

    interpolated_dem_raster = _get_raster(
        args.output_dir, 'interpolatedDem', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_interpolated_dem or
        flag_all)
    slant_range_raster = _get_raster(
        args.output_dir, 'slantRange', gdal.GDT_Float64, shape,
        output_file_list, output_obj_list, args.flag_slant_range or flag_all)
    azimuth_time_raster = _get_raster(
        args.output_dir, 'zeroDopplerAzimuthTime', gdal.GDT_Float64, shape,
        output_file_list, output_obj_list, args.flag_azimuth_time or flag_all)
    incidence_angle_raster = _get_raster(
        args.output_dir, 'incidenceAngle', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_incidence_angle or
        flag_all)
    los_unit_vector_x_raster = _get_raster(
        args.output_dir, 'losUnitVectorX', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_los or flag_all)
    los_unit_vector_y_raster = _get_raster(
        args.output_dir, 'losUnitVectorY', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_los or flag_all)
    along_track_unit_vector_x_raster = _get_raster(
        args.output_dir, 'alongTrackUnitVectorX', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_along_track or flag_all)
    along_track_unit_vector_y_raster = _get_raster(
        args.output_dir, 'alongTrackUnitVectorY', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_along_track or flag_all)
    elevation_angle_raster = _get_raster(
        args.output_dir, 'elevationAngle', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_elevation_angle or
        flag_all)
    ground_track_velocity_raster = _get_raster(
        args.output_dir, 'groundTrackVelocity', gdal.GDT_Float64, shape,
        output_file_list, output_obj_list, args.flag_ground_track_velocity or
        flag_all)
    local_incidence_angle_raster = _get_raster(
        args.output_dir, 'localIncidenceAngle', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_local_incidence_angle or
        flag_all)
    projection_angle_raster = _get_raster(
        args.output_dir, 'projectionAngle', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_projection_angle or
        flag_all)
    simulated_radar_brightness_raster = _get_raster(
        args.output_dir, 'simulatedRadarBrightness', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list,
        args.flag_simulated_radar_brightness or flag_all)

    dem_interp_method = get_dem_interp_method(args.dem_interp_method)

    geo2rdr_params = isce3.geometry.Geo2RdrParams()

    if args.threshold_geo2rdr is not None:
        geo2rdr_params.threshold = args.threshold_geo2rdr
    if args.num_iter_geo2rdr is not None:
        geo2rdr_params.maxiter = args.num_iter_geo2rdr
    if args.delta_range_geo2rdr is not None:
        geo2rdr_params.delta_range = args.delta_range_geo2rdr

    isce3.geogrid.get_radar_grid(lookside,
                                 wavelength,
                                 dem_raster,
                                 geogrid_obj,
                                 orbit,
                                 native_doppler,
                                 grid_doppler,
                                 dem_interp_method,
                                 geo2rdr_params,
                                 interpolated_dem_raster,
                                 slant_range_raster,
                                 azimuth_time_raster,
                                 incidence_angle_raster,
                                 los_unit_vector_x_raster,
                                 los_unit_vector_y_raster,
                                 along_track_unit_vector_x_raster,
                                 along_track_unit_vector_y_raster,
                                 elevation_angle_raster,
                                 ground_track_velocity_raster,
                                 local_incidence_angle_raster,
                                 projection_angle_raster,
                                 simulated_radar_brightness_raster)

    info_channel = journal.info("get_radar_grid")
    for f in output_file_list:
        info_channel.log(f'file saved: {f}')


def get_geolocation_grid(nisar_product_obj, args,
                         other_radar_grid=None):
    '''
    get geolocation grid for L0B and L1 products

    NOTE: other_radar_grid parameter is added in this function
    to accommodate the pixel offsets in the RIFG and ROFF product
    '''

    if other_radar_grid is not None:
        radar_grid = other_radar_grid
    else:
        radar_grid = nisar_product_obj.getRadarGrid()

    orbit = nisar_product_obj.getOrbit()
    grid_doppler = isce3.core.LUT2d()
    native_doppler = nisar_product_obj.getDopplerCentroid()
    native_doppler.bounds_error = False

    rdr2geo_params = isce3.geometry.Rdr2GeoParams()

    if args.threshold_rdr2geo is not None:
        rdr2geo_params.threshold = args.threshold_rdr2geo
    if args.num_iter_rdr2geo is not None:
        rdr2geo_params.maxiter = args.num_iter_rdr2geo
    if args.extra_iter_rdr2geo is not None:
        rdr2geo_params.extraiter = args.extra_iter_rdr2geo

    geo2rdr_params = isce3.geometry.Geo2RdrParams()

    if args.threshold_geo2rdr is not None:
        geo2rdr_params.threshold = args.threshold_geo2rdr
    if args.num_iter_geo2rdr is not None:
        geo2rdr_params.maxiter = args.num_iter_geo2rdr
    if args.delta_range_geo2rdr is not None:
        geo2rdr_params.delta_range = args.delta_range_geo2rdr

    if args.threshold_geo2rdr is None:
        args.threshold_geo2rdr = 1e-8
    if args.num_iter_geo2rdr is None:
        args.num_iter_geo2rdr = 50
    if args.delta_range_geo2rdr is None:
        args.delta_range_geo2rdr = 10.0

    nbands = 1
    shape = [nbands, radar_grid.length, radar_grid.width]
    if args.output_dir and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    dem_raster = isce3.io.Raster(args.dem_file)
    if args.epsg is None:
        args.epsg = dem_raster.get_epsg()

    output_file_list = []
    output_obj_list = []

    flag_all = (not args.flag_interpolated_dem and
                not args.flag_coordinate_x and
                not args.flag_coordinate_y and
                not args.flag_incidence_angle and
                not args.flag_los and
                not args.flag_along_track and
                not args.flag_elevation_angle and
                not args.flag_ground_track_velocity)

    interpolated_dem_raster = _get_raster(
        args.output_dir, 'interpolatedDem', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_interpolated_dem or
        flag_all)
    coordinate_x_raster = _get_raster(
        args.output_dir, 'coordinateX', gdal.GDT_Float64, shape,
        output_file_list, output_obj_list,  args.flag_coordinate_x or flag_all)
    coordinate_y_raster = _get_raster(
        args.output_dir, 'coordinateY', gdal.GDT_Float64, shape,
        output_file_list, output_obj_list, args.flag_coordinate_y or flag_all)
    incidence_angle_raster = _get_raster(
        args.output_dir, 'incidenceAngle', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_incidence_angle or
        flag_all)
    los_unit_vector_x_raster = _get_raster(
        args.output_dir, 'losUnitVectorX', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_los or flag_all)
    los_unit_vector_y_raster = _get_raster(
        args.output_dir, 'losUnitVectorY', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_los or flag_all)
    along_track_unit_vector_x_raster = _get_raster(
        args.output_dir, 'alongTrackUnitVectorX', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_along_track or flag_all)
    along_track_unit_vector_y_raster = _get_raster(
        args.output_dir, 'alongTrackUnitVectorY', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_along_track or flag_all)
    elevation_angle_raster = _get_raster(
        args.output_dir, 'elevationAngle', gdal.GDT_Float32, shape,
        output_file_list, output_obj_list, args.flag_elevation_angle or
        flag_all)
    ground_track_velocity_raster = _get_raster(
        args.output_dir, 'groundTrackVelocity', gdal.GDT_Float64, shape,
        output_file_list, output_obj_list, args.flag_ground_track_velocity or
        flag_all)

    dem_interp_method = get_dem_interp_method(args.dem_interp_method)

    isce3.geometry.get_geolocation_grid(dem_raster,
                                        radar_grid,
                                        orbit,
                                        native_doppler,
                                        grid_doppler,
                                        args.epsg,
                                        dem_interp_method,
                                        rdr2geo_params,
                                        geo2rdr_params,
                                        interpolated_dem_raster,
                                        coordinate_x_raster,
                                        coordinate_y_raster,
                                        incidence_angle_raster,
                                        los_unit_vector_x_raster,
                                        los_unit_vector_y_raster,
                                        along_track_unit_vector_x_raster,
                                        along_track_unit_vector_y_raster,
                                        elevation_angle_raster,
                                        ground_track_velocity_raster)

    # Flush data
    for obj in output_obj_list:
        del obj

    for f in output_file_list:
        print(f'file saved: {f}')


def _get_raster(output_dir, ds_name, dtype, shape, output_file_list,
                output_obj_list, flag_save_layer):
    """Create an ISCE3 raster object (GTiff) for a radar geometry layer.
       Parameters
       ----------
       output_dir: str
              Output directory
       ds_name: str
              Dataset (geometry layer) name
       dtype:: gdal.DataType
              GDAL data type
       shape: list
              Shape of the output raster
       output_file_list: list
              Mutable list of output files
       output_obj_list: list
              Mutable list of output raster objects
       flag_save_layer: bool
              Flag indicating if raster object should be created
       Returns
       -------
       raster_obj : isce3.io.Raster
              ISCE3 raster object
    """
    if not flag_save_layer:
        return

    output_file = os.path.join(output_dir, ds_name)+'.tif'
    raster_obj = isce3.io.Raster(
        output_file,
        shape[2],
        shape[1],
        shape[0],
        dtype,
        "GTiff")
    output_file_list.append(output_file)
    output_obj_list.append(raster_obj)
    return raster_obj


def get_dem_interp_method(dem_interp_method):
    if dem_interp_method is None:
        return isce3.core.DataInterpMethod.BIQUINTIC
    return isce3.core.normalize_data_interp_method(dem_interp_method)


def main(argv=None):
    argv = get_parser()
    run(argv)


if __name__ == '__main__':
    main()
