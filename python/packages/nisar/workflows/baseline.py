#!/usr/bin/env python3

import h5py
import numpy as np
import pathlib
import time
import journal
import isce3
from nisar.products.readers import SLC
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
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
    ref_orbit: object
        orbit object for the reference acquisition
    sec_orbit: object
        orbit object for the secondary acquisition
    ref_doppler: object
        doppler LUT2D for the reference acquisition 
    sec_doppler: object
        doppler LUT2D for the secondary acquisition 
    sec_radargrid: object
        radarGridParameters object for the secondary acquisition
    ellipsoid: object
        an instance of the Ellipsoid class
    geo2rdr_parameters: dict
        A dictionary representing the parameters used in geo2rdr computation.
        The dictionary includes three keys: threshold, maxiter, delta_range
        e.g., geo2rdr_parameters = {'threshold': 1.0e-8, 'maxiter': 50, delta_range: 1.0e-8}

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
    except:
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
    except:
            sec_aztime = np.nan
            sec_rng = np.nan

    # get the sensor position at the sec_aztime on the secondary orbit
    sec_xyz, sec_vel = sec_orbit.interpolate(sec_aztime)

    # compute the baseline
    baseline = np.linalg.norm(sec_xyz - ref_xyz)

    # compute the cosine of the angle between the baseline vector and the
    # reference LOS vector (refernce sensor to target)
    costheta = (ref_rng ** 2 + baseline ** 2 - sec_rng ** 2) / (
        2.0 * ref_rng * baseline)

    # project the baseline to LOS to get the parallel component of the baseline
    # (i.e., parallel to the LOS direction)
    # parallel baseline in refernce LOS direction is positive
    parallel_baseline = baseline * costheta
    if costheta**2 > 1:
        print(ref_rng, baseline, sec_rng)
        print(sec_xyz, ref_xyz)
    # project the baseline to the normal to to the reference LOS direction
    perp_baseline_temp = baseline * np.sqrt(1 - costheta ** 2)

    # get the direction sign of the perpendicular baseline. 
    direction = np.sign(
        np.dot(np.cross(target_xyz - ref_xyz, sec_xyz - ref_xyz), ref_velocity)
    )

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
        a dictionary conatining the the different InSAR product paths
        e.g.: output_paths={"RIFG": "/home/process/insar_rifg.h5",
                            "GUNW": "/home/process/insar_gunw.h5"}
    ref_orbit: object
        orbit object for the reference acquisition
    sec_orbit: object
        orbit object for the secondary acquisition
    sec_radargrid: object
        radarGridParameters object for the secondary acquisition
    doppler: object
        doppler LUT2D
    ellipsoid: object
        an instance of the Ellipsoid class
    metadata_path_dict: dict
        a dictionary representing the path of different metadat cubes
    geo2rdr_parameters: dict
        A dictionary representing the parameters used in geo2rdr computation.
        The dictionary includes three keys: threshold, maxiter, delta_range
        e.g., geo2rdr_parameters = {'threshold': 1.0e-8, 'maxiter': 50, delta_range: 1.0e-8}

    Returns
    -------
    None

    """

    info_channel = journal.info("baseline.run")
    info_channel.log("starting baseline computation")
    t_start = time.time()

    common_parent_path = 'science/LSAR'

    if "RIFG" in output_paths.keys():
        output_hdf5 = output_paths["RIFG"]
        dst_meta_path = f'{common_parent_path}/RIFG/metadata'
        grid_path = f"{dst_meta_path}/geolocationGrid"
        cube_ref_dataset = f'{grid_path}/coordinateX'
    else:
        output_hdf5 = output_paths["GUNW"]
        dst_meta_path = f'{common_parent_path}/GUNW/metadata'
        grid_path = f"{dst_meta_path}/radarGrid"
        cube_ref_dataset = f'{grid_path}/slantRange'

    with h5py.File(output_hdf5, "r+") as src_h5:
        # Create metadata if baselines do not exist in h5 file
        if metadata_path_dict["perpendicularBaseline"] not in src_h5:
            cubes_shape = src_h5[cube_ref_dataset].shape
            descr = "Perpendicular component of the InSAR baseline"
            h5_prep._create_datasets(src_h5[grid_path], cubes_shape, np.float32,
                            "perpendicularBaseline",
                            descr=descr, units="meters",
                            long_name='perpendicular baseline')
            h5_prep._create_datasets(src_h5[grid_path], cubes_shape, np.float32,
                            "parallelBaseline",
                            descr=descr.replace('Perpendicular', 'Parallel'),
                            units="meters",
                            long_name='parallel baseline')

        height_levels = src_h5[metadata_path_dict["heights"]][:]
        ref_times = src_h5[metadata_path_dict["azimuthTime"]][:]
        ref_rnges = src_h5[metadata_path_dict["slantRange"]][:]
        coordX = src_h5[metadata_path_dict["coordX"]][:]
        coordY = src_h5[metadata_path_dict["coordY"]][:]
        ds_bperp = src_h5[metadata_path_dict["perpendicularBaseline"]]
        ds_bpar = src_h5[metadata_path_dict["parallelBaseline"]]
        geo2rdr_parameters['delta_range'] = 1e-8 
        epsg_code = src_h5[metadata_path_dict["epsg"]][()]
        proj = isce3.core.make_projection(epsg_code)
        ellipsoid = proj.ellipsoid

        if "GUNW" in output_paths.keys():
            _, meta_height, meta_width = np.shape(ref_times)
        else:
            meta_height = len(ref_times)
            meta_width = len(ref_rnges)

        par_baseline = np.zeros([meta_height, meta_width], dtype=np.float32)
        perp_baseline = np.zeros([meta_height, meta_width], dtype=np.float32)

        for kk, h in enumerate(height_levels):

            # when we allow a block of geo2rdr run on an array
            # the following two 'for loops' can be eliminated
            for height_ind in range(meta_height):
                for width_ind in range(meta_width):

                    if "GUNW" in output_paths.keys():
                        # sample UAVSAR datasets have some large NOVALUE data
                        if (coordX[width_ind] == -1.00e12) or (coordY[height_ind] == -1.00e12):
                            continue
                        target_proj = np.array([coordX[width_ind], coordY[height_ind], h])
                    else:
                        if (coordX[kk, height_ind, width_ind] == -1.00e12) or \
                            (coordY[kk, height_ind, width_ind] == -1.00e12):
                            continue
                        target_proj = np.array([coordX[kk, height_ind, width_ind], \
                            coordY[kk, height_ind, width_ind], h])
                    
                    target_llh = proj.inverse(target_proj)
                    parallel_baseline, perpendicular_baseline = compute_baseline(
                        target_llh,
                        ref_orbit,
                        sec_orbit,
                        ref_doppler,
                        sec_doppler, 
                        ref_radargrid,
                        sec_radargrid,
                        ellipsoid,
                        geo2rdr_parameters)

                    par_baseline[height_ind, width_ind] = parallel_baseline
                    perp_baseline[height_ind, width_ind] = perpendicular_baseline

            ds_bpar[kk, :, :] = par_baseline
            ds_bperp[kk, :, :] = perp_baseline

        if "RUNW" in output_paths:
            info_channel.log(
                f"copy the perpendicular and parallel baselines to RUNW product"
            )
            with h5py.File(output_paths["RUNW"], "r+") as h5_runw:
                runw_ds_bperp = h5_runw[
                    "science/LSAR/RUNW/metadata/geolocationGrid/perpendicularBaseline"
                ]
                runw_ds_bpar = h5_runw[
                    "science/LSAR/RUNW/metadata/geolocationGrid/parallelBaseline"
                ]
                runw_ds_bperp[:] = ds_bperp[:]
                runw_ds_bpar[:] = ds_bpar[:]

    time_elapsed = time.time() - t_start
    info_channel.log(f"successfully ran baseline in {time_elapsed:.3f} seconds")

    return None

def run(cfg: dict, output_paths):
    """computes the parallel and perpendicular baseline cubes
    and adds them to the InSAR product's metadata 3D cubes.
    The baseline cubes are baseline components computed assuming different
    heights on ground such that each layer represents the baseline at a given height.

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

    ref_slc = SLC(hdf5file=ref_hdf5)
    sec_slc = SLC(hdf5file=sec_hdf5)

    ellipsoid = isce3.core.Ellipsoid()

    ref_orbit = ref_slc.getOrbit()
    sec_orbit = sec_slc.getOrbit()

    ref_radargrid = ref_slc.getRadarGrid()
    sec_radargrid = sec_slc.getRadarGrid()
    
    # native-doppler
    ref_doppler = ref_slc.getDopplerCentroid(frequency='A')
    ref_doppler.bounds_error = False
    sec_doppler = sec_slc.getDopplerCentroid(frequency='A')
    sec_doppler.bounds_error = False

    geo2rdr_parameters = cfg["processing"]["geo2rdr"]
    common_path = 'science/LSAR'
    
    if "GUNW" in output_paths:
        
        dst_meta_path = f'{common_path}/GUNW/metadata'
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
            }

        add_baseline(output_paths_gunw,
                    ref_orbit,
                    sec_orbit,
                    ref_radargrid,
                    sec_radargrid,
                    ref_doppler,
                    sec_doppler, 
                    ellipsoid,
                    metadata_path_dict,
                    geo2rdr_parameters)

    else:
        dst_meta_path = f'{common_path}/RIFG/metadata'
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

        add_baseline(output_paths,
                    ref_orbit,
                    sec_orbit,
                    ref_radargrid,
                    sec_radargrid,
                    ref_doppler,
                    sec_doppler, 
                    ellipsoid,
                    metadata_path_dict,
                    geo2rdr_parameters)

if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    insar_runcfg = InsarRunConfig(args)
    _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)
    run(insar_runcfg.cfg, output_paths=out_paths)
