#!/usr/bin/env python3

"""
collection of functions for NISAR geocode workflow
"""

import pathlib
import time

import h5py
import journal
import pybind_isce3 as isce3
from pybind_nisar.products.readers import SLC
from pybind_nisar.workflows import h5_prep
from pybind_nisar.workflows.geocode_insar_runconfig import \
    GeocodeInsarRunConfig
from pybind_nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from pybind_nisar.workflows.yaml_argparse import YamlArgparse


def run(cfg, runw_hdf5, output_hdf5):
    """
    geocode RUNW products
    """

    # pull parameters from cfg
    ref_hdf5 = cfg["InputFileGroup"]["InputFilePath"]
    freq_pols = cfg["processing"]["input_subset"]["list_of_frequencies"]
    geogrids = cfg["processing"]["geocode"]["geogrids"]
    radar_grid_cubes_geogrid = cfg['processing']['radar_grid_cubes']['geogrid']
    radar_grid_cubes_heights = cfg['processing']['radar_grid_cubes']['heights']
    dem_file = cfg["DynamicAncillaryFileGroup"]["DEMFile"]
    threshold_geo2rdr = cfg["processing"]["geo2rdr"]["threshold"]
    iteration_geo2rdr = cfg["processing"]["geo2rdr"]["maxiter"]
    lines_per_block = cfg["processing"]["blocksize"]["y"]
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

            for pol in pol_list:
                # iterate over key: dataset name value: bool flag to perform geocode
                for dataset_name, geocode_this_dataset in gunw_datasets.items():
                    if not geocode_this_dataset:
                        continue

                    # Create radar grid for the offsets (and dataset path)
                    if dataset_name in ['alongTrackOffset', 'slantRangeOffset']:
                        src_group_path = f'{src_freq_path}/pixelOffsets/{pol}'
                        dst_group_path = f'{dst_freq_path}/pixelOffsets/{pol}'

                        # Define margin used during dense offsets execution
                        margin = max(offset_cfg['margin'],
                                     offset_cfg['gross_offset_range'],
                                     offset_cfg['gross_offset_azimuth'])

                        # If not allocated, determine shape of the offsets
                        if offset_cfg['offset_length'] is None:
                            length_margin = 2 * margin + 2 * offset_cfg[
                                'half_search_azimuth'] + \
                                            offset_cfg['window_azimuth']
                            offset_cfg['offset_length'] = (radar_grid_slc.length -
                                                           length_margin) // offset_cfg['skip_azimuth']
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
                        # prepare input raster
                        input_raster_str = (
                            f"HDF5:{runw_hdf5}:/{src_group_path}/{dataset_name}"
                            )
                        input_raster = isce3.io.Raster(input_raster_str)

                        # access the HDF5 dataset for a given frequency and pol
                        geo.data_interpolator = interp_method
                        dataset_path = f"{dst_group_path}/{dataset_name}"

                    # prepare input raster
                    elif (dataset_name == "layoverShadowMask"):
                        # prepare input raster
                        raster_ref = scratch_path / 'rdr2geo' / f'freq{freq}' / 'mask.rdr'
                        input_raster = isce3.io.Raster(str(raster_ref))

                        # access the HDF5 dataset for layover shadow mask
                        dataset_path = f"{dst_freq_path}/interferogram/{dataset_name}"
                        geo.data_interpolator = 'NEAREST'
                        radar_grid = radar_grid_slc
                    else:
                        # Assign correct radar grid
                        if az_looks > 1 or rg_looks > 1:
                            radar_grid = radar_grid_mlook
                        else:
                            radar_grid = radar_grid_slc

                        # Prepare input path
                        src_group_path = f'{src_freq_path}/interferogram/{pol}'
                        dst_group_path = f'{dst_freq_path}/interferogram/{pol}'

                        # prepare input raster
                        input_raster_str = (
                            f"HDF5:{runw_hdf5}:/{src_group_path}/{dataset_name}"
                        )
                        input_raster = isce3.io.Raster(input_raster_str)

                        # access the HDF5 dataset for a given frequency and pol
                        geo.data_interpolator = interp_method
                        dataset_path = f"{dst_group_path}/{dataset_name}"

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
                        output_mode=isce3.geocode.GeocodeOutputMode.INTERP
                    )

                    del geocoded_raster

            if freq.upper() == 'B':
                continue

            # get doppler centroid
            cube_geogrid = isce3.product.GeoGridParameters(
                start_x=radar_grid_cubes_geogrid.start_x,
                start_y=radar_grid_cubes_geogrid.start_y,
                spacing_x=radar_grid_cubes_geogrid.spacing_x,
                spacing_y=radar_grid_cubes_geogrid.spacing_y,
                width=int(radar_grid_cubes_geogrid.width),
                length=int(radar_grid_cubes_geogrid.length),
                epsg=radar_grid_cubes_geogrid.epsg)

            cube_group_name = '/science/LSAR/GUNW/metadata/radarGrid'

            native_doppler = slc.getDopplerCentroid(frequency=freq)
            '''
            The native-Doppler LUT bounds error is turned off to
            computer cubes values outside radar-grid boundaries
            '''
            native_doppler.bounds_error = False
            add_radar_grid_cubes_to_hdf5(dst_h5, cube_group_name,
                                         cube_geogrid, radar_grid_cubes_heights,
                                         radar_grid, orbit, native_doppler,
                                         grid_zero_doppler, threshold_geo2rdr,
                                         iteration_geo2rdr)

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
    out_paths = h5_prep.run(geocode_insar_runconfig.cfg)
    runw_path = geocode_insar_runconfig.cfg['processing']['geocode'][
        'runw_path']
    if runw_path is not None:
        out_paths['RUNW'] = runw_path

    # Run geocode
    run(geocode_insar_runconfig.cfg, out_paths["RUNW"], out_paths["GUNW"])
