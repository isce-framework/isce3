#!/usr/bin/env python3

import os
import h5py
import numpy as np
import pathlib
import time
import journal
from osgeo import gdal, osr
import isce3
import copy

from nisar.products.readers import SLC
from nisar.workflows.baseline_runconfig import BaselineRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.compute_stats import compute_stats_real_data
from nisar.workflows import h5_prep


def read_geo2rdr(scratch):
    """Read geo2rdr products.

    Parameters
    ----------
    scratch: str
        path for geo2rdr directory
    """
    azoff_path = os.path.join(scratch, 'azimuth.off')
    rgoff_path = os.path.join(scratch, 'range.off')

    azoff_gdal = gdal.Open(azoff_path)
    azoff = azoff_gdal.ReadAsArray()
    del azoff_gdal

    rgoff_gdal = gdal.Open(rgoff_path)
    rgoff = rgoff_gdal.ReadAsArray()
    del rgoff_gdal

    return rgoff, azoff


def write_xyz_data(data, output):
    """Read geo2rdr products.

    Parameters
    ----------
    data: numpy.ndarray
        data to be saved
    output: str
        path for output file
    """
    Gdal_type = gdal.GDT_Float32
    image_size = data.shape

    if len(image_size) == 3:
        nim = image_size[0]
        ny = image_size[1]
        nx = image_size[2]
    elif len(image_size) == 2:
        ny = image_size[0]
        nx = image_size[1]
        nim = 1

    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('ENVI').Create(
                    output, nx, ny, nim, Gdal_type)
    if nim == 1:
        dst_ds.GetRasterBand(1).WriteArray(np.squeeze(data))
    else:
        for im_ind in range(0, nim):
            # write to disk
            dst_ds.GetRasterBand(im_ind+1).WriteArray(
                np.squeeze(data[im_ind, :, :]))
    dst_ds.FlushCache()
    dst_ds = None


def write_xyz(scratch_path, x_array, y_array, height_array):
    """Compute slant range distance and azimuth time 
    from geo2rdr outputs. 

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

    Returns
    -------
    topovrt_path: str
        vrt path for x, y, z rdr files. 
    """
    zpath = os.path.join(scratch_path, 'z.rdr')
    write_xyz_data(height_array, zpath)

    xpath = os.path.join(scratch_path, 'x.rdr')
    write_xyz_data(x_array, xpath)

    ypath = os.path.join(scratch_path, 'y.rdr')
    write_xyz_data(y_array, ypath)

    path_list= [xpath, ypath, zpath]
    raster_list = [isce3.io.Raster(path) for path in path_list]
    topovrt_path = f'{str(scratch_path)}/topo.vrt'
    output_vrt = isce3.io.Raster(topovrt_path, raster_list)
    output_vrt.set_epsg(4326)

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

    az_off[az_off==-1000000]=np.nan
    rg_off[rg_off==-1000000]=np.nan

    rng = radargrid.starting_range \
              + rg_off * radargrid.range_pixel_spacing
    azt = radargrid.sensing_start \
              + az_off / radargrid.prf

    return rng, azt


def compute_baseline(baseline_dir_path,
                     topovrt_path,
                     ref_orbit,
                     sec_orbit,
                     ref_doppler,
                     sec_doppler,
                     ref_radargrid,
                     sec_radargrid,
                     ellipsoid,
                     epsg_code,
                     geo2rdr_parameters,
                     use_gpu=False):
    """Returns perpendicular and parallel components of spatial baseline
    between two SAR orbits.

    Parameters
    ----------
    baseline_dir_path: str
        baseline directory for processing
    topovrt_path: str
        vrt file path for topo
    ref_orbit: isce3.core.Orbit
        orbit object for the reference acquisition
    sec_orbit: isce3.core.Orbit
        orbit object for the secondary acquisition
    ref_doppler: isce3.core.LUT2d
        doppler LUT2D for the reference acquisition
    sec_doppler: isce3.core.LUT2d
        doppler LUT2D for the secondary acquisition
    ref_radargrid: isce3.product.RadarGridParameters
        radarGridParameters object for the reference acquisition
    sec_radargrid: isce3.product.RadarGridParameters
        radarGridParameters object for the secondary acquisition
    ellipsoid: isce3.core.Ellipsoid
        an instance of the Ellipsoid class
    geo2rdr_parameters: dict
        A dictionary representing the parameters used in geo2rdr computation.
        The dictionary includes three keys: threshold, maxiter, delta_range
        e.g., geo2rdr_parameters = {'threshold': 1.0e-8,
                                    'maxiter': 50,
                                    'delta_range': 1.0e-8}
    use_gpu: bool
        bool type for gpu usage
    Returns
    -------
    par_baseline_array: numpy.ndarray
        A component of the baseline parallel to the los vector
        from the reference sensor position to the target.
    perp_baseline_array: numpy.ndarray
        A component of the baseline perpendicular to the los vector
        from the reference sensor position to the target.
    """

    # read x, y, z coordinates
    topo_obj = gdal.Open(topovrt_path)
    topo = topo_obj.ReadAsArray()
    topo_obj = None
    topo_raster = isce3.io.Raster(topovrt_path, update=True)

    proj = isce3.core.make_projection(epsg_code)

    # CPU or GPU geo2rdr
    if use_gpu:
        Geo2Rdr = isce3.cuda.geometry.Geo2Rdr
    else:
        Geo2Rdr = isce3.geometry.Geo2Rdr

    # create geo2rdr output directory for baselines
    ref_base_dir = f'{baseline_dir_path}/ref_baseline'
    sec_base_dir = f'{baseline_dir_path}/sec_baseline'
    os.makedirs(ref_base_dir, exist_ok=True)
    os.makedirs(sec_base_dir, exist_ok=True)

    # run geo2rdr
    geo2rdr_ref_obj = Geo2Rdr(ref_radargrid,
                                ref_orbit,
                                ellipsoid,
                                ref_doppler,
                                geo2rdr_parameters['threshold'],
                                geo2rdr_parameters['maxiter'])

    geo2rdr_sec_obj = Geo2Rdr(sec_radargrid,
                                sec_orbit,
                                ellipsoid,
                                sec_doppler,
                                geo2rdr_parameters['threshold'],
                                geo2rdr_parameters['maxiter'])

    geo2rdr_ref_obj.geo2rdr(topo_raster, ref_base_dir)
    geo2rdr_sec_obj.geo2rdr(topo_raster, sec_base_dir)

    # read slant range and azimuth time
    ref_rngs, ref_azts = \
        compute_rng_aztime(ref_base_dir, ref_radargrid)
    sec_rngs, sec_azts = \
        compute_rng_aztime(sec_base_dir, sec_radargrid)

    perp_baseline_array = np.zeros_like(ref_rngs)
    par_baseline_array = np.zeros_like(ref_rngs)
    meta_rows, meta_cols = ref_rngs.shape

    for row_ind in range(meta_rows):
        for col_ind in range(meta_cols):
            ref_azt = ref_azts[row_ind, col_ind]
            ref_rng = ref_rngs[row_ind, col_ind]
            sec_azt = sec_azts[row_ind, col_ind]
            sec_rng = sec_rngs[row_ind, col_ind]
            target_proj = np.array([topo[0, row_ind, col_ind],
                                    topo[1, row_ind, col_ind],
                                    topo[2, row_ind, col_ind]])

            target_llh = proj.inverse(target_proj)

            target_xyz = ellipsoid.lon_lat_to_xyz(target_llh)

            if not np.isnan(ref_azt):
                ref_xyz, ref_vel = ref_orbit.interpolate(ref_azt)
                # get the sensor position at the sec_aztime
                # on the secondary orbit
                sec_xyz, sec_vel = sec_orbit.interpolate(sec_azt)

                # compute the baseline
                baseline = np.linalg.norm(sec_xyz - ref_xyz)

                # compute the cosine of the angle between the baseline vector and the
                # reference LOS vector (refernce sensor to target)
                if baseline == 0:
                    cos_vbase_los = 1
                else:
                    cos_vbase_los = (ref_rng ** 2 + baseline ** 2 - sec_rng ** 2) / (
                        2.0 * ref_rng * baseline)

                # project the baseline to LOS to get the parallel component of the baseline
                # (i.e., parallel to the LOS direction)
                # parallel baseline in refernce LOS direction is positive
                parallel_baseline = baseline * cos_vbase_los

                # project the baseline to the normal to to the reference LOS direction
                perp_baseline_temp = baseline * np.sqrt(1 - cos_vbase_los ** 2)

                # get the direction sign of the perpendicular baseline.
                # positive perpendicular baseline is defined at below to LOS vector
                direction = np.sign(
                    np.dot(np.cross(target_xyz - ref_xyz, sec_xyz - ref_xyz), ref_vel))
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
                 ellipsoid,
                 metadata_path_dict,
                 geo2rdr_parameters,
                 baseline_mode='top_bottom'):
    """Add perpendicular and parallel components of spatial baseline
    datasets to the metadata cubes of InSAR products.

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
    ellipsoid: isce3.core.Ellipsoid
        an instance of the Ellipsoid class
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
                 "epsg": {metadata_path}/epsg}
        where metadata_path = /science/LSAR/RIFG/metadata/geolocationGrid

    geo2rdr_parameters: dict
        A dictionary representing the parameters used in geo2rdr computation.
        The dictionary includes three keys: threshold, maxiter, delta_range
        e.g., geo2rdr_parameters = {'threshold': 1.0e-8,
                                    'maxiter': 50,
                                    'delta_range': 1.0e-8}
    """
    error_channel = journal.error('baseline.run')

    common_parent_path = 'science/LSAR'
    baseline_dir_path =  metadata_path_dict['baseline_dir']
    use_gpu = metadata_path_dict['use_gpu']

    if "RIFG" in output_paths.keys() or "ROFF" in output_paths.keys() or\
        "RUNW" in output_paths.keys():
        radar_or_geo = 'radar'
        product_id = next(iter(output_paths))
        output_hdf5 = output_paths[product_id]
        dst_meta_path = f'{common_parent_path}/{product_id}/metadata'
        grid_path = f"{dst_meta_path}/geolocationGrid"
        cube_ref_dataset = f'{grid_path}/coordinateX'

    elif "GUNW" in output_paths.keys() or "GOFF" in output_paths.keys():
        product_id = "GUNW"
        radar_or_geo = 'geo'
        output_hdf5 = output_paths[product_id]
        dst_meta_path = f'{common_parent_path}/{product_id}/metadata'
        grid_path = f"{dst_meta_path}/radarGrid"
        cube_ref_dataset = f'{grid_path}/slantRange'

    # remove product_id from copy of output_paths to track
    # other products to insert baseline into
    residual_output_paths = copy.deepcopy(output_paths)
    del residual_output_paths[product_id]

    with h5py.File(output_hdf5, "a") as dst_h5:

        height_levels = dst_h5[metadata_path_dict["heights"]][:]
        ref_times = dst_h5[metadata_path_dict["azimuthTime"]][:]
        ref_rnges = dst_h5[metadata_path_dict["slantRange"]][:]
        coordX = dst_h5[metadata_path_dict["coordX"]][:]
        coordY = dst_h5[metadata_path_dict["coordY"]][:]
        geo2rdr_parameters['delta_range'] = 1e-8
        epsg_code = dst_h5[metadata_path_dict["epsg"]][()]

        proj = isce3.core.make_projection(epsg_code)
        ellipsoid = proj.ellipsoid

        cube_row = dst_h5[cube_ref_dataset].shape[1]
        cube_col = dst_h5[cube_ref_dataset].shape[2]

        # produce baselines for two heights levels
        if len(height_levels) == 1:
            cubes_shape = [1, cube_row, cube_col]
        else:
            if baseline_mode == '3D_full':
                cubes_shape = [len(height_levels), cube_row, cube_col]
            elif baseline_mode == 'top_bottom':
                cubes_shape = [2, cube_row, cube_col]
            else:
                err_str = f'{baseline_mode} is not supported.'
                error_channel.log(err_str)
                raise ValueError(err_str)

        # Create metadata if baselines do not exist in h5 file
        if (metadata_path_dict["perpendicularBaseline"] not in dst_h5):
            recreate_flag = True
        else:
            old_shape = list(dst_h5[
                metadata_path_dict["perpendicularBaseline"]].shape)
            if old_shape != cubes_shape:
                del dst_h5[metadata_path_dict["perpendicularBaseline"]]
                del dst_h5[metadata_path_dict["parallelBaseline"]]
                recreate_flag = True
            else:
                recreate_flag = False

        if recreate_flag:
            descr = "Perpendicular component of the InSAR baseline"
            h5_prep._create_datasets(dst_h5[grid_path],
                            cubes_shape, np.float32,
                            "perpendicularBaseline",
                            descr=descr, units="meters",
                            long_name='perpendicular baseline')
            h5_prep._create_datasets(dst_h5[grid_path],
                            cubes_shape, np.float32,
                            "parallelBaseline",
                            descr=descr.replace(
                                'Perpendicular', 'Parallel'),
                            units="meters",
                            long_name='parallel baseline')

        ds_bperp = dst_h5[metadata_path_dict["perpendicularBaseline"]]
        ds_bpar = dst_h5[metadata_path_dict["parallelBaseline"]]

        bperp_raster_path = f"IH5:::ID={ds_bperp.id.id}".encode("utf-8")
        bperp_raster = isce3.io.Raster(bperp_raster_path, update=True)
        bpar_raster_path = f"IH5:::ID={ds_bpar.id.id}".encode("utf-8")
        bpar_raster = isce3.io.Raster(bpar_raster_path, update=True)

        # compute baselines for 2 height levels assuming the linear variation
        # of the baselines along the heights
        if len(height_levels) > 1:
            if baseline_mode == 'top_bottom':
                height_list = [height_levels[0], height_levels[-1]]
            elif baseline_mode == '3D_full':
                height_list = height_levels
        else:
            height_list = [height_levels[0]]

        for height_ind, h in enumerate(height_list):
            # when we allow a block of geo2rdr run on an array
            # the following two 'for loops' can be eliminated
            if radar_or_geo =='geo':
                coordX2, coordY2 = np.meshgrid(coordX, coordY)
            else:
                # extract the index of the current height
                # in the height level array
                height_level_ind = np.argmin(np.abs(height_levels-h))
                coordX2 = np.squeeze(coordX[height_level_ind, :, :])
                coordY2 = np.squeeze(coordY[height_level_ind, :, :])

            height_array = h * np.ones([cube_row, cube_col],
                                       dtype='float32')
            topovrt_path = write_xyz(baseline_dir_path,
                                     coordX2,
                                     coordY2,
                                     height_array)

            par_baseline, perp_baseline = compute_baseline(
                baseline_dir_path,
                topovrt_path,
                ref_orbit,
                sec_orbit,
                ref_doppler,
                sec_doppler,
                ref_radargrid,
                sec_radargrid,
                ellipsoid,
                epsg_code,
                geo2rdr_parameters,
                use_gpu
                )

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
        if residual_output_paths:
            for residual_key in residual_output_paths.keys():
                perp_base_path = metadata_path_dict["perpendicularBaseline"
                                ].replace(product_id, residual_key)
                para_base_path = metadata_path_dict["parallelBaseline"
                                ].replace(product_id, residual_key)
                grid_path = grid_path.replace(product_id, residual_key)

                with h5py.File(output_paths[residual_key], "r+") as h5_resi:
                    if perp_base_path not in h5_resi:
                        recreate_flag = True
                    else:
                        old_shape = list(h5_resi[perp_base_path].shape)
                        if old_shape != cubes_shape:
                            del h5_resi[perp_base_path]
                            del h5_resi[para_base_path]
                            recreate_flag = True
                        else:
                            recreate_flag = False

                    if recreate_flag:
                        descr = "Perpendicular component of the InSAR baseline"
                        h5_prep._create_datasets(h5_resi[grid_path],
                                        cubes_shape,
                                        np.float32,
                                        "perpendicularBaseline",
                                        descr=descr,
                                        units="meters",
                                        long_name='perpendicular baseline')
                        h5_prep._create_datasets(h5_resi[grid_path],
                                        cubes_shape,
                                        np.float32,
                                        "parallelBaseline",
                                        descr=descr.replace('Perpendicular',
                                                            'Parallel'),
                                        units="meters",
                                        long_name='parallel baseline')

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
    src_ds: dict
        HDF5 dataset for source
    dst_ds: h5py.File
        h5py file for target
    """
    dst_ds.attrs.create('min_value', src_ds.attrs['min_value'])
    dst_ds.attrs.create('mean_value', src_ds.attrs['mean_value'])
    dst_ds.attrs.create('max_value', src_ds.attrs['max_value'])
    dst_ds.attrs.create('sample_stddev', src_ds.attrs['sample_stddev'])

def run(cfg: dict, output_paths):
    """computes the parallel and perpendicular baseline cubes
    and adds them to the InSAR product's metadata 3D cubes.
    The baseline cubes are baseline components computed assuming different
    heights on ground such that each layer represents
    the baseline at a given height.

    Parameters
    ----------
    cfg: dict
        a dictionary of the insar.py configuration workflow
    output_paths: dict
        a dictionary conatining the the different InSAR product paths
        e.g.: output_paths={"RIFG": "/home/process/insar_rifg.h5",
                            "GUNW": "/home/process/insar_gunw.h5"}

    """
    ref_hdf5 = cfg["input_file_group"]["reference_rslc_file_path"]
    sec_hdf5 = cfg["input_file_group"]["secondary_rslc_file_path"]
    scratch_path = cfg['product_path_group']['scratch_path']
    ref_orbit_path = cfg['dynamic_ancillary_file_group']['orbit'][
                         'reference_orbit_file']
    sec_orbit_path = cfg['dynamic_ancillary_file_group']['orbit'][
                         'secondary_orbit_file']
    baseline_mode = cfg['processing']['baseline']['mode']
    baseline_dir_path = f'{scratch_path}/baseline'

    # check if gpu use if required
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])
    if use_gpu:
        # set CUDA device
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)

    info_channel = journal.info("baseline.run")
    info_channel.log("starting baseline")
    t_all = time.time()

    # create baseline directory
    os.makedirs(baseline_dir_path, exist_ok=True)

    ref_slc = SLC(hdf5file=ref_hdf5)
    sec_slc = SLC(hdf5file=sec_hdf5)

    ellipsoid = isce3.core.Ellipsoid()

    # import external orbit if file exists
    if ref_orbit_path is not None:
        ref_orbit = load_orbit_from_xml(ref_orbit_path)
    else:
        ref_orbit = ref_slc.getOrbit()

    if sec_orbit_path is not None:
        sec_orbit = load_orbit_from_xml(sec_orbit_path)
    else:
        sec_orbit = sec_slc.getOrbit()

    ref_radargrid = ref_slc.getRadarGrid()
    sec_radargrid = sec_slc.getRadarGrid()

    # baseline is estimated assuming native-doppler
    # if frequency A exists, use frequencyA doppler,
    # if not, use frequency B instead.
    if 'A' in ref_slc.frequencies:
        ref_freq = 'A'
    else:
        ref_freq = 'B'
    ref_doppler = ref_slc.getDopplerCentroid(frequency=ref_freq)
    ref_doppler.bounds_error = False

    if 'A' in sec_slc.frequencies:
        sec_freq = 'A'
    else:
        sec_freq = 'B'
    sec_doppler = sec_slc.getDopplerCentroid(frequency=sec_freq)
    sec_doppler.bounds_error = False

    geo2rdr_parameters = cfg["processing"]["geo2rdr"]
    common_path = 'science/LSAR'

    radar_products = {dst: output_paths[dst]
                    for dst in output_paths.keys()
                    if dst.startswith('R')}
    geo_products = {dst: output_paths[dst]
                    for dst in output_paths.keys()
                    if dst.startswith('G')}

    if geo_products:
        # only GUNW product have information requred to compute baesline.
        product_id =  next(iter(geo_products))
        dst_meta_path = f'{common_path}/{product_id}/metadata'
        grid_path = f"{dst_meta_path}/radarGrid"
        output_paths_gunw = {"GUNW": output_paths["GUNW"]}
        metadata_path_dict = {
            "heights": f"{grid_path}/heightAboveEllipsoid",
            "azimuthTime": f"{grid_path}/zeroDopplerAzimuthTime",
            "slantRange": f"{grid_path}/slantRange",
            "coordX": f"{grid_path}/xCoordinates",
            "coordY": f"{grid_path}/yCoordinates",
            "perpendicularBaseline": f"{grid_path}/perpendicularBaseline",
            "parallelBaseline": f"{grid_path}/parallelBaseline",
            "epsg": f"{grid_path}/epsg",
            "use_gpu": use_gpu,
            "baseline_dir": baseline_dir_path
            }

        add_baseline(geo_products,
                    ref_orbit,
                    sec_orbit,
                    ref_radargrid,
                    sec_radargrid,
                    ref_doppler,
                    sec_doppler,
                    ellipsoid,
                    metadata_path_dict,
                    geo2rdr_parameters,
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
            "use_gpu": use_gpu,
            "baseline_dir": baseline_dir_path
            }

        add_baseline(radar_products,
                    ref_orbit,
                    sec_orbit,
                    ref_radargrid,
                    sec_radargrid,
                    ref_doppler,
                    sec_doppler,
                    ellipsoid,
                    metadata_path_dict,
                    geo2rdr_parameters,
                    baseline_mode)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran baseline in {t_all_elapsed:.3f} seconds")

if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    baseline_runcfg = BaselineRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(baseline_runcfg.cfg)
    run(baseline_runcfg.cfg, output_paths=out_paths)
