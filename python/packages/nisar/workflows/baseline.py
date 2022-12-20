#!/usr/bin/env python3

import h5py
import numpy as np
import pathlib
import time
import journal
import isce3
import copy
from nisar.products.readers import SLC
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.compute_stats import compute_stats_real_data
from nisar.workflows import h5_prep


def compute_baseline(target_llh,
                     ref_orbit,
                     sec_orbit,
                     ref_doppler,
                     sec_doppler,
                     ref_radargrid,
                     sec_radargrid,
                     ellipsoid,
                     geo2rdr_parameters):

    """Returns perpendicular and parallel components of spatial baseline
    between two SAR orbits.

    Parameters
    ----------
    target_llh: array
        target position as an array of [longitude, latitude, height]
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

    Returns
    -------
    parallel_baseline: float
        A component of the baseline parallel to the los vector
        from the reference sensor position to the target.
    perpendicular_baseline: float
        A component of the baseline perpendicular to the los vector
        from the reference sensor position to the target.
    """
    # convert the target lon,lat,height to cartesian XYZ
    # Interinsic assumption is that the target is in ECEF lon lat height
    target_xyz = ellipsoid.lon_lat_to_xyz(target_llh)

    # get the azimuth time and slant range to given target
    # in order to avoid the divergence of geo2rdr, use try and except
    try:
        ref_aztime, ref_rng = isce3.geometry.geo2rdr(
            target_llh,
            ellipsoid,
            ref_orbit,
            ref_doppler,
            ref_radargrid.wavelength,
            ref_radargrid.lookside,
            threshold=geo2rdr_parameters["threshold"],
            maxiter=geo2rdr_parameters["maxiter"],
            delta_range=geo2rdr_parameters["delta_range"])
    except RuntimeError:
        # geo2rdr may fail in areas outside swath.
        ref_aztime = np.nan
        ref_rng = np.nan

    # get sensor position and velocity at ref_aztime
    ref_xyz, ref_velocity = ref_orbit.interpolate(ref_aztime)

    # call geo2rdr to compute the slant range and azimuth time
    # of the sensor on second orbit
    try:
        sec_aztime, sec_rng = isce3.geometry.geo2rdr(
            target_llh,
            ellipsoid,
            sec_orbit,
            sec_doppler,
            sec_radargrid.wavelength,
            sec_radargrid.lookside,
            threshold=geo2rdr_parameters["threshold"],
            maxiter=geo2rdr_parameters["maxiter"],
            delta_range=geo2rdr_parameters["delta_range"])
    except RuntimeError:
        sec_aztime = np.nan
        sec_rng = np.nan

    # get the sensor position at the sec_aztime on the secondary orbit
    sec_xyz, sec_vel = sec_orbit.interpolate(sec_aztime)

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
        np.dot(np.cross(target_xyz - ref_xyz, sec_xyz - ref_xyz), ref_velocity))
    perpendicular_baseline = direction * perp_baseline_temp

    return parallel_baseline, perpendicular_baseline

def add_baseline(output_paths,
                 ref_orbit,
                 sec_orbit,
                 ref_radargrid,
                 sec_radargrid,
                 ref_doppler,
                 sec_doppler,
                 ellipsoid,
                 metadata_path_dict,
                 geo2rdr_parameters):
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
                 "perpendicularBaseline": {metadata_path}/perpendicularBaseline,
                 "parallelBaseline": {metadata_path}/parallelBaseline,
                 "epsg": {metadata_path}/epsg}
        where metadata_path = /science/LSAR/RIFG/metadata/geolocationGrid

    geo2rdr_parameters: dict
        A dictionary representing the parameters used in geo2rdr computation.
        The dictionary includes three keys: threshold, maxiter, delta_range
        e.g., geo2rdr_parameters = {'threshold': 1.0e-8, 'maxiter': 50, delta_range: 1.0e-8}
    """

    common_parent_path = 'science/LSAR'
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
            cubes_shape = [2, cube_row, cube_col]

        # Create metadata if baselines do not exist in h5 file
        if metadata_path_dict["perpendicularBaseline"] not in dst_h5:

            descr = "Perpendicular component of the InSAR baseline"
            h5_prep._create_datasets(dst_h5[grid_path], cubes_shape, np.float32,
                            "perpendicularBaseline",
                            descr=descr, units="meters",
                            long_name='perpendicular baseline')
            h5_prep._create_datasets(dst_h5[grid_path], cubes_shape, np.float32,
                            "parallelBaseline",
                            descr=descr.replace('Perpendicular', 'Parallel'),
                            units="meters",
                            long_name='parallel baseline')

        ds_bperp = dst_h5[metadata_path_dict["perpendicularBaseline"]]
        ds_bpar = dst_h5[metadata_path_dict["parallelBaseline"]]

        if radar_or_geo =='geo':
            _, meta_row, meta_width = np.shape(ref_times)
        else:
            meta_row = len(ref_times)
            meta_width = len(ref_rnges)

        bperp_raster_path = f"IH5:::ID={ds_bperp.id.id}".encode("utf-8")
        bperp_raster = isce3.io.Raster(bperp_raster_path, update=True)
        bpar_raster_path = f"IH5:::ID={ds_bpar.id.id}".encode("utf-8")
        bpar_raster = isce3.io.Raster(bpar_raster_path, update=True)

        par_baselines = np.zeros([meta_row, meta_width], dtype=np.float32)
        perp_baselines = np.zeros([meta_row, meta_width], dtype=np.float32)

        # compute baselines for 2 height levels assuming the linear variation
        # of the baselines along the heights
        if len(height_levels) > 1:
            height_list = [height_levels[0], height_levels[-1]]
        else:
            height_list = [height_levels[0]]

        for height_ind, h in enumerate(height_list):
            # when we allow a block of geo2rdr run on an array
            # the following two 'for loops' can be eliminated
            for row_ind in range(meta_row):
                for col_ind in range(meta_width):

                    if radar_or_geo =='geo':
                        target_proj = np.array([coordX[col_ind],
                                                coordY[row_ind],
                                                h])
                    else:
                        # extract the index of the current height
                        # in the height level array
                        height_level_ind = np.argmin(np.abs(height_levels-h))

                        # sample UAVSAR datasets have some large NOVALUE data
                        if (coordX[height_level_ind, row_ind, col_ind] \
                                == -1.00e12) or \
                           (coordY[height_level_ind, row_ind, col_ind] \
                                == -1.00e12):
                            continue
                        target_proj = np.array([coordX[height_level_ind,
                                                       row_ind,
                                                       col_ind], \
                                                coordY[height_level_ind,
                                                       row_ind,
                                                       col_ind],
                                                h])

                    target_llh = proj.inverse(target_proj)
                    par_baseline, perp_baseline = compute_baseline(
                        target_llh,
                        ref_orbit,
                        sec_orbit,
                        ref_doppler,
                        sec_doppler,
                        ref_radargrid,
                        sec_radargrid,
                        ellipsoid,
                        geo2rdr_parameters)

                    par_baselines[row_ind, col_ind] = par_baseline
                    perp_baselines[row_ind, col_ind] = perp_baseline

                ds_bpar[height_ind, :, :] = par_baselines
                ds_bperp[height_ind, :, :] = perp_baselines

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
    Returns
    -------
    None

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
    ref_orbit_path = cfg['dynamic_ancillary_file_group']['orbit'][
                         'reference_orbit_file']
    sec_orbit_path = cfg['dynamic_ancillary_file_group']['orbit'][
                         'secondary_orbit_file']

    info_channel = journal.info("baseline.run")
    info_channel.log("starting baseline")
    t_all = time.time()

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
        print(product_id)
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
            "epsg": f"{grid_path}/epsg",
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
                    geo2rdr_parameters)

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
                    geo2rdr_parameters)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran baseline in {t_all_elapsed:.3f} seconds")
