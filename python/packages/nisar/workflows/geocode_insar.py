#!/usr/bin/env python3

"""
collection of functions for NISAR geocode workflow
"""

import numpy as np
import pathlib
import shutil
import time

import h5py
import journal
import isce3
from osgeo import gdal
from nisar.products.readers import SLC
from nisar.workflows import h5_prep
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from nisar.workflows.geocode_insar_runconfig import \
    GeocodeInsarRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.compute_stats import compute_stats_real_data


def run(cfg, runw_hdf5, output_hdf5):
    """ Run geocode insar on user specified hardware

    Parameters
    ----------
    cfg : dict
        Dictionary containing run configuration
    runw_hdf5 : str
        Path input RUNW HDF5
    output_hdf5 : str
        Path to output GUNW HDF5
    """
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)
        gpu_run(cfg, runw_hdf5, output_hdf5)
    else:
        cpu_run(cfg, runw_hdf5, output_hdf5)


def get_shadow_input_output(scratch_path, freq, dst_freq_path):
    """ Create input raster object and output dataset path for shadow layover

    Parameters
    ----------
    scratch_path : pathlib.Path
        Scratch path to shadow layover mask rasters
    freq : str
        Frequency, A or B, of shadow layover mask raster
    dst_freq_path : str
        HDF5 path to destination frequency group of geocoded shadow layover

    Returns
    -------
    input_raster : isce3.io.Raster
        Shadow layover input raster object
    dataset_path : str
        HDF5 path to geocoded shadow layover dataset
    """
    raster_ref = scratch_path / 'rdr2geo' / f'freq{freq}' / 'layoverShadowMask.rdr'
    input_raster = isce3.io.Raster(str(raster_ref))

    # access the HDF5 dataset for layover shadow mask
    dataset_path = f"{dst_freq_path}/interferogram/layoverShadowMask"

    return input_raster, dataset_path


def get_input_output(src_freq_path, dst_freq_path, pol, runw_hdf5, dataset_name):
    """ Create input raster object and output dataset path for datasets outside

    Parameters
    ----------
    src_freq_path : str
        HDF5 path to input frequency group of input dataset
    dst_freq_path : str
        HDF5 path to input frequency group of output dataset
    pol : str
        Polarity of input dataset
    runw_hdf5 : str
        Path to input RUNW HDF5
    dataset_name : str
        Name of dataset to be geocoded

    Returns
    -------
    input_raster : isce3.io.Raster
        Shadow layover input raster object
    dataset_path : str
        HDF5 path to geocoded shadow layover dataset
    """
    if dataset_name in ['alongTrackOffset', 'slantRangeOffset']:
        src_group_path = f'{src_freq_path}/pixelOffsets/{pol}'
        dst_group_path = f'{dst_freq_path}/pixelOffsets/{pol}'
    else:
        src_group_path = f'{src_freq_path}/interferogram/{pol}'
        dst_group_path = f'{dst_freq_path}/interferogram/{pol}'

    # prepare input raster
    input_raster_str = (f"HDF5:{runw_hdf5}:/{src_group_path}/{dataset_name}")
    input_raster = isce3.io.Raster(input_raster_str)

    # access the HDF5 dataset for a given frequency and pol
    dataset_path = f"{dst_group_path}/{dataset_name}"

    return input_raster, dataset_path


def get_offset_radar_grid(offset_cfg, radar_grid_slc):
    ''' Create radar grid object for offset datasets

    Parameters
    ----------
    offset_cfg : dict
        Dictionary containing offset run configuration
    radar_grid_slc : SLC
        Object containing SLC properties
    '''
    # Define margin used during dense offsets execution
    margin = max(offset_cfg['margin'],
                 offset_cfg['gross_offset_range'],
                 offset_cfg['gross_offset_azimuth'])

    # If not allocated, determine shape of the offsets
    if offset_cfg['offset_length'] is None:
        length_margin = 2 * margin + 2 * offset_cfg[
            'half_search_azimuth'] + \
                        offset_cfg['window_azimuth']
        offset_cfg['offset_length'] = (radar_grid_slc.length - length_margin
                                       ) // offset_cfg['skip_azimuth']
    if offset_cfg['offset_width'] is None:
        width_margin = 2 * margin + 2 * offset_cfg[
            'half_search_range'] + \
                       offset_cfg['window_range']
        offset_cfg['offset_width'] = (radar_grid_slc.width -
                                      width_margin) // offset_cfg['skip_azimuth']
    # Determine the starting range and sensing start for the offset radar grid
    offset_starting_range = radar_grid_slc.starting_range + \
                            (offset_cfg['start_pixel_range'] + offset_cfg['window_range']//2)\
                            * radar_grid_slc.range_pixel_spacing
    offset_sensing_start = radar_grid_slc.sensing_start + \
                           (offset_cfg['start_pixel_azimuth'] + offset_cfg['window_azimuth']//2)\
                           / radar_grid_slc.prf
    # Range spacing for offsets
    offset_range_spacing = radar_grid_slc.range_pixel_spacing * offset_cfg['skip_range']
    offset_prf = radar_grid_slc.prf / offset_cfg['skip_azimuth']

    # Create offset radar grid
    radar_grid = isce3.product.RadarGridParameters(offset_sensing_start,
                                                   radar_grid_slc.wavelength,
                                                   offset_prf,
                                                   offset_starting_range,
                                                   offset_range_spacing,
                                                   radar_grid_slc.lookside,
                                                   offset_cfg['offset_length'],
                                                   offset_cfg['offset_width'],
                                                   radar_grid_slc.ref_epoch)
    return radar_grid


def add_radar_grid_cube(cfg, freq, radar_grid, orbit, dst_h5):
    ''' Write radar grid cube to HDF5

    Parameters
    ----------
    cfg : dict
        Dictionary containing run configuration
    freq : str
        Frequency in HDF5 to add cube to
    radar_grid : isce3.product.radar_grid
        Radar grid of current frequency of datasets other than offset and shadow
        layover datasets
    orbit : isce3.core.orbit
        Orbit object of current SLC
    dst_h5: str
        Path to output GUNW HDF5
    '''
    radar_grid_cubes_geogrid = cfg['processing']['radar_grid_cubes']['geogrid']
    radar_grid_cubes_heights = cfg['processing']['radar_grid_cubes']['heights']
    threshold_geo2rdr = cfg["processing"]["geo2rdr"]["threshold"]
    iteration_geo2rdr = cfg["processing"]["geo2rdr"]["maxiter"]

    ref_hdf5 = cfg["InputFileGroup"]["InputFilePath"]
    slc = SLC(hdf5file=ref_hdf5)

    # get doppler centroid
    cube_geogrid_param = isce3.product.GeoGridParameters(
        start_x=radar_grid_cubes_geogrid.start_x,
        start_y=radar_grid_cubes_geogrid.start_y,
        spacing_x=radar_grid_cubes_geogrid.spacing_x,
        spacing_y=radar_grid_cubes_geogrid.spacing_y,
        width=int(radar_grid_cubes_geogrid.width),
        length=int(radar_grid_cubes_geogrid.length),
        epsg=radar_grid_cubes_geogrid.epsg)

    cube_group_path = '/science/LSAR/GUNW/metadata/radarGrid'

    native_doppler = slc.getDopplerCentroid(frequency=freq)
    grid_zero_doppler = isce3.core.LUT2d()
    '''
    The native-Doppler LUT bounds error is turned off to
    computer cubes values outside radar-grid boundaries
    '''
    native_doppler.bounds_error = False
    add_radar_grid_cubes_to_hdf5(dst_h5, cube_group_path,
                                 cube_geogrid_param, radar_grid_cubes_heights,
                                 radar_grid, orbit, native_doppler,
                                 grid_zero_doppler, threshold_geo2rdr,
                                 iteration_geo2rdr)


def cpu_run(cfg, runw_hdf5, output_hdf5):
    """ Geocode RUNW products on CPU

    Parameters
    ----------
    cfg : dict
        Dictionary containing run configuration
    runw_hdf5 : str
        Path input RUNW HDF5
    output_hdf5 : str
        Path to output GUNW HDF5
    """
    # pull parameters from cfg
    ref_hdf5 = cfg["InputFileGroup"]["InputFilePath"]
    freq_pols = cfg["processing"]["input_subset"]["list_of_frequencies"]
    geogrids = cfg["processing"]["geocode"]["geogrids"]
    dem_file = cfg["DynamicAncillaryFileGroup"]["DEMFile"]
    threshold_geo2rdr = cfg["processing"]["geo2rdr"]["threshold"]
    iteration_geo2rdr = cfg["processing"]["geo2rdr"]["maxiter"]
    lines_per_block = cfg["processing"]["geocode"]["lines_per_block"]
    dem_block_margin = cfg["processing"]["dem_margin"]
    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]
    interp_method = cfg["processing"]["geocode"]["interp_method"]
    gunw_datasets = cfg["processing"]["geocode"]["datasets"]
    scratch_path = pathlib.Path(cfg['ProductPathGroup']['ScratchPath'])
    offset_cfg = cfg["processing"]["dense_offsets"]

    slc = SLC(hdf5file=ref_hdf5)

    info_channel = journal.info("geocode.run")
    info_channel.log("starting geocode")

    # NISAR products are always zero doppler
    grid_zero_doppler = isce3.core.LUT2d()

    # set defaults shared by both frequencies
    dem_raster = isce3.io.Raster(dem_file)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # init geocode object
    geo = isce3.geocode.GeocodeFloat32()

    # init geocode members
    orbit = slc.getOrbit()
    geo.orbit = orbit
    geo.ellipsoid = ellipsoid
    geo.doppler = grid_zero_doppler
    geo.threshold_geo2rdr = threshold_geo2rdr
    geo.numiter_geo2rdr = iteration_geo2rdr
    geo.dem_block_margin = dem_block_margin
    geo.lines_per_block = lines_per_block
    geo.data_interpolator = interp_method

    t_all = time.time()
    with h5py.File(output_hdf5, "a") as dst_h5:
        for freq in freq_pols.keys():
            pol_list = freq_pols[freq]

            radar_grid_slc = slc.getRadarGrid(freq)
            if az_looks > 1 or rg_looks > 1:
                radar_grid_mlook = radar_grid_slc.multilook(az_looks, rg_looks)

            geo_grid = geogrids[freq]
            geo.geogrid(
                geo_grid.start_x,
                geo_grid.start_y,
                geo_grid.spacing_x,
                geo_grid.spacing_y,
                geo_grid.width,
                geo_grid.length,
                geo_grid.epsg,
            )
            src_freq_path = f"/science/LSAR/RUNW/swaths/frequency{freq}"
            dst_freq_path = f"/science/LSAR/GUNW/grids/frequency{freq}"

            # flag to ensure layover shadown only geocoded once per freq
            # layover shadow has no polarization
            skip_layover_shadow = False
            for pol in pol_list:
                # iterate over key: dataset name value: bool flag to perform geocode
                for dataset_name, geocode_this_dataset in gunw_datasets.items():
                    if not geocode_this_dataset:
                        continue

                    if dataset_name == "layoverShadowMask" and skip_layover_shadow:
                        continue

                    # Create radar grid for the offsets (and dataset path)
                    if (dataset_name == "layoverShadowMask"):
                        input_raster, dataset_path = get_shadow_input_output(
                            scratch_path, freq, dst_freq_path)
                        skip_layover_shadow = True
                    else:
                        input_raster, dataset_path = get_input_output(src_freq_path,
                                                                      dst_freq_path,
                                                                      pol,
                                                                      runw_hdf5,
                                                                      dataset_name)

                    if dataset_name in ['alongTrackOffset', 'slantRangeOffset']:
                        radar_grid = get_offset_radar_grid(offset_cfg,
                                                           radar_grid_slc)
                        geo.data_interpolator = interp_method
                    # prepare input raster
                    elif (dataset_name == "layoverShadowMask"):
                        geo.data_interpolator = 'NEAREST'
                        radar_grid = radar_grid_slc
                    else:
                        geo.data_interpolator = interp_method
                        # Assign correct radar grid
                        if az_looks > 1 or rg_looks > 1:
                            radar_grid = radar_grid_mlook
                        else:
                            radar_grid = radar_grid_slc

                    geocoded_dataset = dst_h5[dataset_path]

                    # Construct the output ratster directly from HDF5 dataset
                    geocoded_raster = isce3.io.Raster(
                        f"IH5:::ID={geocoded_dataset.id.id}".encode("utf-8"),
                        update=True,
                    )

                    geo.geocode(
                        radar_grid=radar_grid,
                        input_raster=input_raster,
                        output_raster=geocoded_raster,
                        dem_raster=dem_raster,
                        output_mode=isce3.geocode.GeocodeOutputMode.INTERP)

                    del geocoded_raster

                    # Construct the output ratster directly from HDF5 dataset
                    geocoded_raster = isce3.io.Raster(
                        f"IH5:::ID={geocoded_dataset.id.id}".encode("utf-8"))

                    if (dataset_name != "layoverShadowMask"):
                        # Layover/shadow masks dont't have min/max/mean/stddev
                        # stats attributes
                        compute_stats_real_data(geocoded_raster, geocoded_dataset)

            # spec for NISAR GUNW does not require freq B so skip radar cube
            if freq.upper() == 'B':
                continue

            add_radar_grid_cube(cfg, freq, radar_grid, orbit, dst_h5)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran geocode in {t_all_elapsed:.3f} seconds")


def gpu_run(cfg, runw_hdf5, output_hdf5):
    """ Geocode RUNW products on GPU

    Parameters
    ----------
    cfg : dict
        Dictionary containing run configuration
    runw_hdf5 : str
        Path input RUNW HDF5
    output_hdf5 : str
        Path to output GUNW HDF5
    """
    t_all = time.time()

    # Extract parameters from cfg dictionary
    dem_block_margin = cfg["processing"]["dem_margin"]
    ref_hdf5 = cfg["InputFileGroup"]["InputFilePath"]
    dem_file = cfg["DynamicAncillaryFileGroup"]["DEMFile"]
    freq_pols = cfg["processing"]["input_subset"]["list_of_frequencies"]
    geogrids = cfg["processing"]["geocode"]["geogrids"]
    lines_per_block = cfg["processing"]["geocode"]["lines_per_block"]
    interp_method = cfg["processing"]["geocode"]["interp_method"]
    gunw_datasets = cfg["processing"]["geocode"]["datasets"]
    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]
    offset_cfg = cfg["processing"]["dense_offsets"]
    scratch_path = pathlib.Path(cfg['ProductPathGroup']['ScratchPath'])

    if interp_method == 'BILINEAR':
        interp_method = isce3.core.DataInterpMethod.BILINEAR
    if interp_method == 'BICUBIC':
        interp_method = isce3.core.DataInterpMethod.BICUBIC
    if interp_method == 'NEAREST':
        interp_method = isce3.core.DataInterpMethod.NEAREST
    if interp_method == 'BIQUINTIC':
        interp_method = isce3.core.DataInterpMethod.BIQUINTIC

    info_channel = journal.info("geocode.run")
    info_channel.log("starting geocode")

    # Init frequency independent objects
    slc = SLC(hdf5file=ref_hdf5)
    grid_zero_doppler = isce3.core.LUT2d()
    dem_raster = isce3.io.Raster(dem_file)

    with h5py.File(output_hdf5, "a", libver='latest', swmr=True) as dst_h5:
        # Loop over frequencies
        for freq in freq_pols.keys():

            geogrid = geogrids[freq]

            # Create frequency based radar grid
            radar_grid = slc.getRadarGrid(freq)
            if az_looks > 1 or rg_looks > 1:
                # Multilook radar grid if needed
                radar_grid = radar_grid.multilook(az_looks, rg_looks)

            # Create radar grid geometry used by most datasets
            rdr_geometry = isce3.container.RadarGeometry(radar_grid,
                                                         slc.getOrbit(),
                                                         grid_zero_doppler)

            # Create geocode object other than offset and shadow layover datasets
            geocode_obj = isce3.cuda.geocode.Geocode(geogrid, rdr_geometry,
                                                     dem_raster,
                                                     dem_block_margin,
                                                     lines_per_block,
                                                     interp_method,
                                                     invalid_value=np.nan)

            '''
            connectedComponents raster has type unsigned char and an invalid
            value of NaN becomes 0 which conflicts with 0 being used to indicate
            an unmasked value/pixel. 255 is chosen as it is the most distant
            value from components assigned in ascending order [0, 1, ...)
            '''
            geocode_conn_comp_obj = isce3.cuda.geocode.Geocode(geogrid, rdr_geometry,
                                                     dem_raster,
                                                     dem_block_margin,
                                                     lines_per_block,
                                                     isce3.core.DataInterpMethod.NEAREST,
                                                     invalid_value=0)

            # If needed create geocode object for offset datasets
            if gunw_datasets['alongTrackOffset'] or gunw_datasets['slantRangeOffset']:
                # Create offset unique radar grid
                radar_grid = get_offset_radar_grid(offset_cfg,
                                                   slc.getRadarGrid(freq))

                # Create radar grid geometry required by offset datasets
                rdr_geometry = isce3.container.RadarGeometry(radar_grid,
                                                             slc.getOrbit(),
                                                             grid_zero_doppler)

                geocode_offset_obj = isce3.cuda.geocode.Geocode(geogrid,
                                                                rdr_geometry,
                                                                dem_raster,
                                                                dem_block_margin,
                                                                lines_per_block,
                                                                interp_method,
                                                                invalid_value=np.nan)

            # If needed create geocode object for shadow layover dataset
            if gunw_datasets['layoverShadowMask']:
                # Create radar grid geometry required by layover shadow
                rdr_geometry = isce3.container.RadarGeometry(slc.getRadarGrid(freq),
                                                             slc.getOrbit(),
                                                             grid_zero_doppler)

                '''
                layover shadow raster has type char and an invalid
                value of NaN becomes 0 which conflicts with 0 being used
                to indicate an unmasked value/pixel. 127 is chosen as it is
                the most distant value from the allowed set of [0, 1, 2, 3].
                '''
                geocode_shadow_obj = isce3.cuda.geocode.Geocode(geogrid,
                                                                rdr_geometry,
                                                                dem_raster,
                                                                dem_block_margin,
                                                                lines_per_block,
                                                                isce3.core.DataInterpMethod.NEAREST,
                                                                invalid_value=127)

            pol_list = freq_pols[freq]
            src_freq_path = f"/science/LSAR/RUNW/swaths/frequency{freq}"
            dst_freq_path = f"/science/LSAR/GUNW/grids/frequency{freq}"
            # flag to ensure layover shadown only geocoded once per freq
            # layover shadow has no polarization
            skip_layover_shadow = False
            # Loop over polarizations
            for pol in pol_list:

                # Loop over number blocks
                for i_block in range(geocode_obj.n_blocks):
                    # Set interpolation grid for current block
                    geocode_obj.set_block_radar_coord_grid(i_block)
                    geocode_conn_comp_obj.set_block_radar_coord_grid(i_block)

                    if gunw_datasets['alongTrackOffset'] or gunw_datasets['slantRangeOffset']:
                        geocode_offset_obj.set_block_radar_coord_grid(i_block)

                    if gunw_datasets['layoverShadowMask']:
                        geocode_shadow_obj.set_block_radar_coord_grid(i_block)

                    # Iterate over/input output raster pairs and geocode
                    for dataset_name, geocode_this_dataset in gunw_datasets.items():
                        if dataset_name == "layoverShadowMask" and skip_layover_shadow:
                            continue

                        # Prepare input raster
                        if (dataset_name == "layoverShadowMask"):
                            input_raster, dataset_path = get_shadow_input_output(
                                scratch_path, freq, dst_freq_path)
                        else:
                            input_raster, dataset_path = get_input_output(src_freq_path,
                                                                          dst_freq_path,
                                                                          pol,
                                                                          runw_hdf5,
                                                                          dataset_name)

                        # Prepare output raster
                        # Access the HDF5 dataset for a given frequency and pol
                        geocoded_dataset = dst_h5[dataset_path]

                        # Construct the output ratster directly from HDF5 dataset
                        geocoded_raster = isce3.io.Raster(
                            f"IH5:::ID={geocoded_dataset.id.id}".encode("utf-8"),
                            update=True)

                        # Choose geocode object based on dataset
                        if dataset_name in ['alongTrackOffset', 'slantRangeOffset']:
                            block_geocode_obj = geocode_offset_obj
                        elif dataset_name == "layoverShadowMask":
                            block_geocode_obj = geocode_shadow_obj
                        elif dataset_name == "connectedComponents":
                            block_geocode_obj = geocode_conn_comp_obj
                        else:
                            block_geocode_obj = geocode_obj

                        block_geocode_obj.geocode_raster_block(geocoded_raster,
                                                               input_raster)

                        geocoded_raster.set_geotransform([geogrid.start_x,
                                                          geogrid.spacing_x, 0.0,
                                                          geogrid.start_y, 0.0,
                                                          geogrid.spacing_y])
                        del input_raster
                        del geocoded_raster

                        # Construct the output ratster directly from HDF5 dataset
                        geocoded_raster = isce3.io.Raster(
                            f"IH5:::ID={geocoded_dataset.id.id}".encode("utf-8"))

                        if (dataset_name != "layoverShadowMask"):
                            # Layover/shadow masks dont't have min/max/mean/stddev
                            # stats attributes
                            compute_stats_real_data(geocoded_raster, geocoded_dataset)

                if gunw_datasets['layoverShadowMask']:
                    skip_layover_shadow = True

            # spec for NISAR GUNW does not require freq B so skip radar cube
            if freq.upper() == 'B':
                continue

            add_radar_grid_cube(cfg, freq, radar_grid, slc.getOrbit(), dst_h5)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"Successfully ran geocode in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    """
    run geocode from command line
    """

    # load command line args
    geocode_insar_parser = YamlArgparse()
    args = geocode_insar_parser.parse()

    # Get a runconfig dictionary from command line args
    geocode_insar_runconfig = GeocodeInsarRunConfig(args)

    # prepare RIFG HDF5
    geocode_insar_runconfig.cfg['PrimaryExecutable']['ProductType'] = 'GUNW_STANDALONE'
    out_paths = h5_prep.run(geocode_insar_runconfig.cfg)
    runw_path = geocode_insar_runconfig.cfg['processing']['geocode'][
        'runw_path']
    if runw_path is not None:
        out_paths['RUNW'] = runw_path

    # Run geocode
    run(geocode_insar_runconfig.cfg, out_paths["RUNW"], out_paths["GUNW"])
