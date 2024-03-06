#!/usr/bin/env python3

import copy
import os
import time

import h5py
import isce3
import journal
import numpy as np
from osgeo import gdal
from scipy.interpolate import griddata

from nisar.products.insar.product_paths import CommonPaths
from nisar.products.readers import SLC
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.workflows import h5_prep
from nisar.workflows.baseline_runconfig import BaselineRunConfig
from nisar.workflows.compute_stats import compute_stats_real_data
from nisar.workflows.yaml_argparse import YamlArgparse


def read_geo2rdr(scratch):
    """Read geo2rdr products.

    Parameters
    ----------
    scratch: str
        path for geo2rdr directory

    Returns
    -------
    rg_off: numpy.ndarray
        The range offsets.
    az_off: numpy.ndarray
        The azimuth offsets.
    """
    off_arr = [[]] * 2
    for i, off_type in enumerate(['range', 'azimuth']):
        off_path = os.path.join(scratch, f'{off_type}.off')
        off_gdal = gdal.Open(off_path)
        off_arr[i] = off_gdal.ReadAsArray()
        del off_gdal
    # range offset, azimuth offset
    return off_arr[0], off_arr[1]


def write_xyz_data(data, output):
    """Save x/y/z data to ENVI format file.

    Parameters
    ----------
    data: numpy.ndarray
        data to be saved
    output: str
        path for output file
    """
    info_channel = journal.info("baseline.run.write_xyz_data")
    error_channel = journal.error("baseline.run.write_xyz_data")

    gdal_type = gdal.GDT_Float32
    image_size = data.shape

    if len(image_size) == 3:
        nim, ny, nx = image_size
    elif len(image_size) == 2:
        ny, nx = image_size
        nim = 1
    else:
        err_str = "Input data must be 2D or 3D."
        error_channel.log(err_str)
        raise ValueError(err_str)

    if np.isnan(data).any():
        info_channel.log("NaN values found in x/y/z data. "
                         "This will be replaced with nearest neighbors.")

        y, x = np.indices(data.shape[-2:])
        valid_mask = ~np.isnan(data)
        coords = np.array([x[valid_mask], y[valid_mask]]).T
        values = data[valid_mask]

        data[np.isnan(data)] = griddata(coords,
                                        values,
                                        (x[np.isnan(data)],
                                         y[np.isnan(data)]),
                                        method='nearest')

    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('ENVI').Create(
                    output, nx, ny, nim, gdal_type)
    if dst_ds is None:
        err_str = f"Could not create output file at {output}."
        error_channel.log(err_str)
        raise IOError(err_str)
    # Write data to file
    if nim == 1:
        dst_ds.GetRasterBand(1).WriteArray(np.squeeze(data))
    else:
        for im_ind in range(nim):
            dst_ds.GetRasterBand(im_ind + 1).WriteArray(data[im_ind])


def write_xyz(scratch_path,
              x_array,
              y_array,
              height_array,
              epsg_code):
    """Save x/y/z data to raster files for geo2rdr
    and produce VRT file.

    Parameters
    ----------
    scratch_path: str
        temp directory for baseline
    x_array: numpy.ndarray
        2 dimensional array for longitude
    y_array: numpy.ndarray
        2 dimensional array for latitude
    height_array:numpy.ndarray
        2 dimensional array for height
    epsg_code: int
        epsg code for x, y, height

    Returns
    -------
    topovrt_path: str
        vrt path for x, y, z rdr files.
    """
    path_list = []
    for xyz, data in zip('xyz', [x_array, y_array, height_array]):
        write_xyz_data(data, f'{scratch_path}/{xyz}.rdr')
        path_list.append(f'{scratch_path}/{xyz}.rdr')

    raster_list = [isce3.io.Raster(path) for path in path_list]
    topovrt_path = f'{str(scratch_path)}/topo.vrt'
    output_vrt = isce3.io.Raster(topovrt_path, raster_list)
    output_vrt.set_epsg(epsg_code)

    return topovrt_path


def compute_rng_aztime(scratch, radargrid):
    """Compute slant range distance and azimuth time
    from geo2rdr outputs.

    Parameters
    ----------
    scratch: str
        geo2rdr directory having azimuth off and range off
    radargrid: isce3.product.RadarGridParameters
        radarGridParameters object

    Returns
    -------
    rng: numpy.ndarray
        2 dimensional range distance computed from range offset
    azt: numpy.ndarray
        2 dimensional azimuth time computed from azimuth offset
    """

    rg_off, az_off = read_geo2rdr(scratch)

    az_off[az_off == -1000000] = np.nan
    rg_off[rg_off == -1000000] = np.nan

    rpixel = np.repeat(np.reshape(np.arange(rg_off.shape[1]),
                                  [1, rg_off.shape[1]]),
                                  rg_off.shape[0], axis=0)
    apixel = np.repeat(np.reshape(np.arange(az_off.shape[0]),
                                  [rg_off.shape[0], 1]),
                                  az_off.shape[1], axis=1)

    rng = radargrid.starting_range \
        + (rg_off + rpixel)\
        * radargrid.range_pixel_spacing
    azt = radargrid.sensing_start \
        + (az_off + apixel) / radargrid.prf

    return rng, azt


def _get_rgrid_dopp_orbit(slc_obj, orbit_path=None):
    """Internal helper to get radargrid, doppler, and orbit
    from SLC object to avoid repeating code

    Parameters
    ----------
    slc_obj: nisar.products.readers.SLC
        SLC object
    orbit_path: str or None, optional
        External orbit XML file path. If not provided, defaults to the
        orbit data contained within the RSLC product.
    Returns
    -------
    radargrid: isce3.product.RadarGridParameters
        radargrid of frequency A if frequency A exists
        radargrid of frequency A if not
    doppler: isce3.product.LUT2d
        doppler LUT2D
    orbit: isce3.core.Orbit
        orbit object
    """

    # if frequency A exists, use frequencyA doppler,
    # if not, use frequency B instead.
    freq = 'A' if 'A' in slc_obj.frequencies else 'B'
    radargrid = slc_obj.getRadarGrid(freq)

    # import external orbit if file exists
    if orbit_path is not None:
        orbit = load_orbit_from_xml(orbit_path, radargrid.ref_epoch)
    else:
        orbit = slc_obj.getOrbit()

    # baseline is estimated assuming native-doppler
    doppler = slc_obj.getDopplerCentroid(frequency=freq)
    doppler.bounds_error = False

    return radargrid, doppler, orbit


def _prepare_baseline_datasets(dst_h5,
                               perp_base_path,
                               para_base_path,
                               grid_path,
                               cubes_shape):
    """Internal convenience function that creates baseline datasets

    Parameters
    ----------
    dst_h5: h5py.File
       h5py File, data to be saved
    perp_base_path: str
         hdf5 path for perpendicular baseline
    para_base_path: str
        hdf5 path for parallel baseline
    grid_path: str
        hdf5 path for geolocation/radarGrid
    cubes_shape: list
        metadata cube size, [height, row, col]
    """
    # Create metadata if baselines do not exist in h5 file
    if perp_base_path not in dst_h5:
        recreate_flag = True

    else:
        # Delete and recreate baseline with new sizes
        # if baseline mode do not match the existing baselines
        old_shape = list(dst_h5[perp_base_path].shape)
        if old_shape != cubes_shape:
            del dst_h5[perp_base_path]
            del dst_h5[para_base_path]
            recreate_flag = True
        else:
            recreate_flag = False

    if not recreate_flag:
        return

    for bmode in ['perpendicular', 'parallel']:
        bmode_cap = bmode.capitalize()
        descr = f"{bmode_cap} component of the InSAR baseline"

        h5_prep._create_datasets(
            dst_h5[grid_path],
            cubes_shape,
            np.float32,
            f"{bmode}Baseline",
            descr=descr,
            units="meters",
            long_name=f'{bmode} baseline')


def _get_invalid_regions(slant_range,
                         min_slant_range,
                         max_slant_range):
    """
    Finds invalid regions using slant range,
    considering the specified minimum and maximum range,
    and also checks for NaN values.

    Parameters
    ----------
    slant_range : numpy.ndarray
        numpy array representing the slant range distances
        for reference acquisition.
    min_slant_range : float
        Minimum valid value for the slant range.
    max_slant_range : float
        Maximum valid value for the slant range.

    Returns
    -------
    invalid_region : numpy.ndarray
        A boolean array where True indicates
        invalid regions in the slant range array.
    """
    invalid_region = \
        (slant_range > max_slant_range) | \
        (slant_range < min_slant_range) | \
        (np.isnan(slant_range))
    return invalid_region


def compute_baseline(ref_rngs,
                     ref_azts,
                     sec_rngs,
                     sec_azts,
                     coord_set,
                     ref_orbit,
                     sec_orbit,
                     ellipsoid,
                     epsg_code):
    """Returns perpendicular and parallel components of
    spatial baseline between two SAR orbits.

    Parameters
    ----------
    ref_rngs : numpy.ndarray
        2 dimensional range distance for reference
        acquisition with size [N, M]
    ref_azts: numpy.ndarray
        2 dimensional azimuth time for reference
        acquisition with size [N, M]
    sec_rngs: numpy.ndarray
        2 dimensional range distance for secondary
        acquisition with size [N, M]
    sec_azts: numpy.ndarray
        2 dimensional azimuth time for secondary
        acquisition with size [N, M]
    coord_set: numpy.ndarray
        set of 2 dimensional x/y/z with size [3, N, M]
    ref_orbit: isce3.core.Orbit
        orbit object for the reference acquisition
    sec_orbit: isce3.core.Orbit
        orbit object for the secondary acquisition
    ellipsoid: isce3.core.Ellipsoid
        an instance of the Ellipsoid class
    epsg_code: int
        epsg code for coord_set

    Returns
    -------
    par_baseline_array: numpy.ndarray
        A component of the baseline parallel to the los vector
        from the reference sensor position to the target.
    perp_baseline_array: numpy.ndarray
        A component of the baseline perpendicular to the los vector
        from the reference sensor position to the target.
    """

    proj = isce3.core.make_projection(epsg_code)
    meta_rows, meta_cols = ref_rngs.shape

    # Initialize output arrays
    perp_baseline_array = np.zeros([meta_rows, meta_cols])
    par_baseline_array = np.zeros([meta_rows, meta_cols])

    for row_ind in range(meta_rows):
        for col_ind in range(meta_cols):
            ref_azt = ref_azts[row_ind, col_ind]
            ref_rng = ref_rngs[row_ind, col_ind]
            sec_azt = sec_azts[row_ind, col_ind]
            sec_rng = sec_rngs[row_ind, col_ind]
            target_proj = np.array([coord_set[0, row_ind, col_ind],
                                    coord_set[1, row_ind, col_ind],
                                    coord_set[2, row_ind, col_ind]])

            target_llh = proj.inverse(target_proj)

            target_xyz = ellipsoid.lon_lat_to_xyz(target_llh)

            if not np.isnan(ref_azt):
                ref_xyz, ref_vel = ref_orbit.interpolate(ref_azt)
                # get the sensor position at the sec_aztime
                # on the secondary orbit
                sec_xyz, _ = sec_orbit.interpolate(sec_azt)

                # compute the baseline
                baseline = np.linalg.norm(sec_xyz - ref_xyz)

                # compute the cosine of the angle between the baseline vector
                # and the reference LOS vector (refernce sensor to target)
                if baseline == 0:
                    cos_vbase_los = 1
                else:
                    cos_vbase_los = (ref_rng ** 2 + baseline ** 2
                                     - sec_rng ** 2) / (
                                    2.0 * ref_rng * baseline)

                # project the baseline to LOS to get the parallel component
                # of the baseline (i.e., parallel to the LOS direction)
                # parallel baseline in refernce LOS direction is positive
                parallel_baseline = baseline * cos_vbase_los

                # project the baseline to the normal to
                # the reference LOS direction
                perp_baseline_temp = baseline * np.sqrt(1 - cos_vbase_los ** 2)

                # get the direction sign of the perpendicular baseline.
                # positive perpendicular baseline is defined
                # at below to LOS vector
                direction = np.sign(
                    np.dot(np.cross(target_xyz - ref_xyz,
                                    sec_xyz - ref_xyz),
                           ref_vel))
                perpendicular_baseline = direction * perp_baseline_temp

                perp_baseline_array[row_ind, col_ind] = perpendicular_baseline
                par_baseline_array[row_ind, col_ind] = parallel_baseline
            else:
                perp_baseline_array[row_ind, col_ind] = np.nan
                par_baseline_array[row_ind, col_ind] = np.nan

    return par_baseline_array, perp_baseline_array


def add_baseline(output_paths,
                 ref_orbit,
                 sec_orbit,
                 ref_radargrid,
                 sec_radargrid,
                 ref_doppler,
                 sec_doppler,
                 metadata_path_dict,
                 geo2rdr_parameters,
                 use_gpu,
                 baseline_dir_path,
                 baseline_mode='top_bottom'):
    """Add perpendicular and parallel components of spatial baseline
    datasets to the metadata cubes of InSAR products.

    If the parallel and perpendicular baseline cubes are already present
    in the product, then they will be overwritten.

    Parameters
    ----------
    output_paths: dict
        a dictionary containing the the different InSAR product paths
        e.g.: output_paths={"RIFG": "/home/process/insar_rifg.h5",
                            "GUNW": "/home/process/insar_gunw.h5"}
    ref_orbit: isce3.core.Orbit
        orbit object for the reference acquisition
    sec_orbit: isce3.core.Orbit
        orbit object for the secondary acquisition
    ref_radargrid: isce3.product.RadarGridParameters
        radarGridParameters object for the reference acquisition
    sec_radargrid: isce3.product.RadarGridParameters
        radarGridParameters object for the secondary acquisition
    ref_doppler: isce3.core.LUT2d
        doppler LUT2D for the reference acquisition
    sec_doppler: isce3.core.LUT2d
        doppler LUT2D for the secondary acquisition
    metadata_path_dict: dict
        a dictionary representing the path of different metadata cubes
        e.g., metadata_path_dict =
                {"heights": {metadata_path}/heightAboveEllipsoid,
                 "azimuthTime": {metadata_path}/zeroDopplerTime,
                 "slantRange": {metadata_path}/slantRange,
                 "coordX": {metadata_path}/xCoordinates,
                 "coordY": {metadata_path}/yCoordinates,
                 "perpendicularBaseline":
                            {metadata_path}/perpendicularBaseline,
                 "parallelBaseline": {metadata_path}/parallelBaseline,
                 "epsg": {metadata_path}/epsg},
                 "projection": {metadata_path}/projection},
                 "range_start",
                 "range_end",
        where metadata_path = /science/LSAR/RIFG/metadata/geolocationGrid

    geo2rdr_parameters: dict
        A dictionary representing the parameters used in geo2rdr computation.
        The dictionary includes three keys: threshold, maxiter, delta_range
        e.g., geo2rdr_parameters = {'threshold': 1.0e-8,
                                    'maxiter': 50,
                                    'delta_range': 1.0e-8}
    use_gpu: bool
        gpu usage flag
    baseline_dir_path: str
        directory path for baseline computation
    baseline_mode: str
        'top_bottom' computes baselines at bottom and top heights
        '3D_full' computes baselines at all heights
    """
    error_channel = journal.error('baseline.run')

    # CPU or GPU geo2rdr
    if use_gpu:
        geo2rdr = isce3.cuda.geometry.Geo2Rdr
    else:
        geo2rdr = isce3.geometry.Geo2Rdr

    # check if products are 'RIFG, RUNW, ROFF' or 'GUNW, GOFF'
    first_product_id = next(iter(output_paths))
    if first_product_id.startswith('R'):
        radar_or_geo = 'radar'
        product_id = next(iter(output_paths))
    elif first_product_id.startswith('G'):
        radar_or_geo = 'geo'
        product_id = next(iter(output_paths))

    output_hdf5 = output_paths[product_id]
    dst_meta_path = f'{CommonPaths.RootPath}/{product_id}/metadata'

    # read 3d cube size from arbitary metadata
    if radar_or_geo == 'radar':
        grid_path = f"{dst_meta_path}/geolocationGrid"
        cube_ref_dataset = f'{grid_path}/coordinateX'
    else:
        grid_path = f"{dst_meta_path}/radarGrid"
        cube_ref_dataset = f'{grid_path}/slantRange'

    # Remove product_id from copy of output_paths to track
    # other products to insert baseline into.
    # Instead of computing baselines for residual outputs,
    # baselines are copied to the residual outputs.
    residual_output_paths = copy.deepcopy(output_paths)
    del residual_output_paths[product_id]

    with h5py.File(output_hdf5, "a") as dst_h5:

        height_levels = dst_h5[metadata_path_dict["heights"]][:]
        coord_x = dst_h5[metadata_path_dict["coordX"]][:]
        coord_y = dst_h5[metadata_path_dict["coordY"]][:]
        if 'projection' not in metadata_path_dict:
            # L1 products have the 'epsg' dataset
            epsg_code = dst_h5[metadata_path_dict["epsg"]][()]
        else:
            # L2 products have the 'projection' dataset that includes
            # the `epsg_code` attribute
            epsg_code = \
                dst_h5[metadata_path_dict["projection"]].attrs['epsg_code']
        slant_range = dst_h5[metadata_path_dict["slantRange"]][:]
        proj = isce3.core.make_projection(epsg_code)
        ellipsoid = proj.ellipsoid

        slant_range_min = metadata_path_dict["range_start"]
        slant_range_max = metadata_path_dict["range_end"]
        # Read row and column size from metadata.
        cube_row = dst_h5[cube_ref_dataset].shape[1]
        cube_col = dst_h5[cube_ref_dataset].shape[2]

        # if height_levels has one height, then ignore the baseline mode
        if len(height_levels) == 1:
            cubes_shape = [1, cube_row, cube_col]
            height_list = [height_levels[0]]
        else:
            # baselines are computed for all height levels
            if baseline_mode == '3D_full':
                cubes_shape = [len(height_levels), cube_row, cube_col]
                height_list = height_levels
            # produce baselines for two heights levels (bottom and top)
            elif baseline_mode == 'top_bottom':
                cubes_shape = [2, cube_row, cube_col]
                height_list = [height_levels[0], height_levels[-1]]
            else:
                err_str = f'Baseline mode {baseline_mode} is not supported.'
                error_channel.log(err_str)
                raise ValueError(err_str)

        _prepare_baseline_datasets(dst_h5,
                                   metadata_path_dict["perpendicularBaseline"],
                                   metadata_path_dict["parallelBaseline"],
                                   grid_path,
                                   cubes_shape)

        ds_bperp = dst_h5[metadata_path_dict["perpendicularBaseline"]]
        ds_bpar = dst_h5[metadata_path_dict["parallelBaseline"]]

        bperp_raster_path = f"IH5:::ID={ds_bperp.id.id}".encode("utf-8")
        bperp_raster = isce3.io.Raster(bperp_raster_path, update=True)
        bpar_raster_path = f"IH5:::ID={ds_bpar.id.id}".encode("utf-8")
        bpar_raster = isce3.io.Raster(bpar_raster_path, update=True)

        for height_ind, height in enumerate(height_list):

            if radar_or_geo == 'geo':
                # coordX and coordY have one dimension
                grid_x, grid_y = np.meshgrid(coord_x, coord_y)
            else:
                # extract the index of the current height
                # in the height level array
                height_level_ind = np.argmin(np.abs(height_levels-height))
                grid_x = np.squeeze(coord_x[height_level_ind, :, :])
                grid_y = np.squeeze(coord_y[height_level_ind, :, :])

            height_2d = height * np.ones([cube_row, cube_col],
                                         dtype='float32')
            topovrt_path = write_xyz(baseline_dir_path,
                                     grid_x,
                                     grid_y,
                                     height_2d,
                                     epsg_code)
            coord_set = np.zeros([3, cube_row, cube_col])
            coord_set[0, :, :] = grid_x
            coord_set[1, :, :] = grid_y
            coord_set[2, :, :] = height_2d

            topo_raster = isce3.io.Raster(topovrt_path)
            base_dir_set = []
            for refsec, rdrgrid, orbit, dopp in \
                zip(['ref', 'sec'], [ref_radargrid, sec_radargrid],
                    [ref_orbit, sec_orbit], [ref_doppler, sec_doppler]):
                base_dir = f'{baseline_dir_path}/{refsec}_geo2rdr'
                os.makedirs(base_dir, exist_ok=True)
                # run geo2rdr
                geo2rdr_obj = geo2rdr(rdrgrid,
                                      orbit,
                                      ellipsoid,
                                      dopp,
                                      geo2rdr_parameters['threshold'],
                                      geo2rdr_parameters['maxiter'])
                geo2rdr_obj.geo2rdr(topo_raster, base_dir)
                base_dir_set.append(base_dir)

            # read slant range and azimuth time
            ref_rngs, ref_azts = \
                compute_rng_aztime(base_dir_set[0], ref_radargrid)
            sec_rngs, sec_azts = \
                compute_rng_aztime(base_dir_set[1], sec_radargrid)

            # get invalid regions where the slant distance
            # is out of observable range.
            if radar_or_geo == 'geo':
                invalid = _get_invalid_regions(
                    np.squeeze(slant_range[height_ind, :, :]),
                    min_slant_range=slant_range_min,
                    max_slant_range=slant_range_max)
            else:
                invalid_range = _get_invalid_regions(
                    np.squeeze(slant_range),
                    min_slant_range=slant_range_min,
                    max_slant_range=slant_range_max)
                invalid = np.tile(invalid_range, (cube_row, 1))
            par_baseline, perp_baseline = compute_baseline(
                ref_rngs,
                ref_azts,
                sec_rngs,
                sec_azts,
                coord_set,
                ref_orbit,
                sec_orbit,
                ellipsoid,
                epsg_code)
            par_baseline[invalid] = np.nan
            perp_baseline[invalid] = np.nan
            ds_bpar[height_ind, :, :] = par_baseline
            ds_bperp[height_ind, :, :] = perp_baseline

        # compute statistics
        data_names = ['perpendicularBaseline', 'parallelBaseline']
        for data_name in data_names:
            dataset = dst_h5[metadata_path_dict[data_name]]
            raster_path = f"IH5:::ID={dataset.id.id}".encode("utf-8")
            baseline_raster = isce3.io.Raster(raster_path)
            compute_stats_real_data(baseline_raster, dataset)
            del baseline_raster

        # Copy baselines to the other products if more than one products
        # are requested
        for residual_key in residual_output_paths.keys():
            perp_base_path = metadata_path_dict["perpendicularBaseline"
                                                ].replace(product_id,
                                                          residual_key)
            para_base_path = metadata_path_dict["parallelBaseline"
                                                ].replace(product_id,
                                                          residual_key)
            grid_path_resi = grid_path.replace(product_id, residual_key)

            with h5py.File(output_paths[residual_key], "r+") as h5_resi:

                _prepare_baseline_datasets(h5_resi,
                                           perp_base_path,
                                           para_base_path,
                                           grid_path_resi,
                                           cubes_shape)

                residual_ds_bperp = h5_resi[perp_base_path]
                residual_ds_bpar = h5_resi[para_base_path]
                residual_ds_bperp[:] = ds_bperp[:]
                residual_ds_bpar[:] = ds_bpar[:]

                # copy attributes from RIFG to RUNW
                copy_attr(ds_bpar, residual_ds_bpar)
                copy_attr(ds_bperp, residual_ds_bperp)


def copy_attr(src_ds, dst_ds):
    """Copy statistics from one to another.

    Parameters
    ----------
    src_ds: h5py.Dataset
        HDF5 dataset for source
    dst_ds: h5py.Dataset
        h5py file for target
    """
    dst_ds.attrs.create('min_value', src_ds.attrs['min_value'])
    dst_ds.attrs.create('mean_value', src_ds.attrs['mean_value'])
    dst_ds.attrs.create('max_value', src_ds.attrs['max_value'])
    dst_ds.attrs.create('sample_stddev', src_ds.attrs['sample_stddev'])


def run(cfg: dict, output_paths):
    """Compute the parallel and perpendicular baseline cubes and add them to
    the InSAR product's metadata 3D cubes.
    The baseline cubes are baseline components computed assuming different
    heights on ground such that each layer represents
    the baseline at a given height.

    Parameters
    ----------
    cfg: dict
        A dictionary of the insar.py configuration workflow
    output_paths: dict
        A dictionary conatining the different InSAR product paths
        e.g.: output_paths={"RIFG": "/home/process/insar_rifg.h5",
                            "GUNW": "/home/process/insar_gunw.h5"}

    """
    ref_hdf5 = cfg["input_file_group"]["reference_rslc_file"]
    sec_hdf5 = cfg["input_file_group"]["secondary_rslc_file"]
    scratch_path = cfg['product_path_group']['scratch_path']
    ref_orbit_path = cfg['dynamic_ancillary_file_group']['orbit_files'][
                         'reference_orbit_file']
    sec_orbit_path = cfg['dynamic_ancillary_file_group']['orbit_files'][
                         'secondary_orbit_file']
    baseline_mode = cfg['processing']['baseline']['mode']
    baseline_dir_path = f'{scratch_path}/baseline'

    info_channel = journal.info("baseline.run")
    info_channel.log("starting baseline")
    t_all = time.time()

    # check if gpu use if required
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])
    if use_gpu:
        # set CUDA device
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)

    # create baseline directory
    os.makedirs(baseline_dir_path, exist_ok=True)

    ref_slc = SLC(hdf5file=ref_hdf5)
    sec_slc = SLC(hdf5file=sec_hdf5)

    # read radargrid, doppler, orbit
    ref_radargrid, ref_doppler, ref_orbit = \
        _get_rgrid_dopp_orbit(ref_slc, ref_orbit_path)
    sec_radargrid, sec_doppler, sec_orbit = \
        _get_rgrid_dopp_orbit(sec_slc, sec_orbit_path)
    range_start = ref_radargrid.starting_range
    range_end = range_start + \
        ref_radargrid.width * ref_radargrid.range_pixel_spacing
    geo2rdr_parameters = cfg["processing"]["geo2rdr"]
    common_path = CommonPaths.RootPath

    radar_products = {dst: output_paths[dst]
                      for dst in output_paths.keys()
                      if dst.startswith('R')}
    geo_products = {dst: output_paths[dst]
                    for dst in output_paths.keys()
                    if dst.startswith('G')}

    if geo_products:
        # only GUNW product have information requred to compute baesline.
        product_id = next(iter(geo_products))
        dst_meta_path = f'{common_path}/{product_id}/metadata'
        grid_path = f"{dst_meta_path}/radarGrid"
        metadata_path_dict = {
            "heights": f"{grid_path}/heightAboveEllipsoid",
            "azimuthTime": f"{grid_path}/zeroDopplerAzimuthTime",
            "slantRange": f"{grid_path}/slantRange",
            "coordX": f"{grid_path}/xCoordinates",
            "coordY": f"{grid_path}/yCoordinates",
            "perpendicularBaseline": f"{grid_path}/perpendicularBaseline",
            "parallelBaseline": f"{grid_path}/parallelBaseline",
            "projection": f"{grid_path}/projection",
            "range_start": range_start,
            "range_end": range_end,
            }

        add_baseline(
            geo_products,
            ref_orbit,
            sec_orbit,
            ref_radargrid,
            sec_radargrid,
            ref_doppler,
            sec_doppler,
            metadata_path_dict,
            geo2rdr_parameters,
            use_gpu,
            baseline_dir_path,
            baseline_mode)

    if radar_products:
        product_id = next(iter(radar_products))
        dst_meta_path = f'{common_path}/{product_id}/metadata'
        grid_path = f"{dst_meta_path}/geolocationGrid"

        metadata_path_dict = {
            "heights": f"{grid_path}/heightAboveEllipsoid",
            "azimuthTime": f"{grid_path}/zeroDopplerTime",
            "slantRange": f"{grid_path}/slantRange",
            "coordX": f"{grid_path}/coordinateX",
            "coordY": f"{grid_path}/coordinateY",
            "perpendicularBaseline": f"{grid_path}/perpendicularBaseline",
            "parallelBaseline": f"{grid_path}/parallelBaseline",
            "epsg": f"{grid_path}/epsg",
            "range_start": range_start,
            "range_end": range_end,
            }

        add_baseline(
            radar_products,
            ref_orbit,
            sec_orbit,
            ref_radargrid,
            sec_radargrid,
            ref_doppler,
            sec_doppler,
            metadata_path_dict,
            geo2rdr_parameters,
            use_gpu,
            baseline_dir_path,
            baseline_mode)

    t_all_elapsed = time.time() - t_all
    info_channel.log("successfully ran baseline "
                     f"in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    baseline_runcfg = BaselineRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(baseline_runcfg.cfg)
    run(baseline_runcfg.cfg, output_paths=out_paths)
