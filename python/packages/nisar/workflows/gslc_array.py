#!/usr/bin/env python3

import os
import time

import h5py
import journal
import numpy as np

import isce3
import nisar
from nisar.products.readers import SLC
from nisar.workflows import h5_prep
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.gslc_runconfig import GSLCRunConfig
from nisar.workflows.compute_stats import compute_stats_complex_data
from nisar.products.readers.orbit import load_orbit_from_xml


def _block_generator(geo_grid, radar_grid, orbit, dem_interp,
                     geo_lines_per_block, geo_cols_per_block,
                     geogrid_expansion_threshold=1):
    """
    Compute radar/geo slices, dimensions and geo grid for each geo block. If no
    radar data is found for a geo block, nothing is returned for that geo block.

    Parameters:
    -----------
    geo_grid: isce3.product.GeoGridParameters
        Geo grid whose radar grid bounding box indices are to be computed
    radar_grid: isce3.product.RadarGridParameters
        Radar grid that computed indices are computed with respect to
    orbit: Orbit
        Orbit object
    dem_interp: isce3.geometry.DEMInterpolator
        DEM to be interpolated over geo grid
    geo_lines_per_block: int
        Line per geo block
    geo_cols_per_block: int
        Columns per geo block
    geogrid_expansion_threshold: int
        Number of geogrid expansions if geo2rdr fails (default: 100)

    Yields:
    -------
    radar_slice: tuple[slice]
        Slice of current radar block. Defined as:
        [azimuth_time_start:azimuth_time_stop, slant_range_start:slant_range_stop]
    geo_slice: tuple[slice]
        Slice of current geo block. Defined as:
        [y_start:y_stop, x_start:x_stop]
    geo_block_shape: tuple[int]
        Shape of current geo block as (block_length, block_width)
    blk_geo_grid: isce3.product.GeoGridParameters
        Geo grid parameters of current geo block
    """
    info_channel = journal.info("gslc_array._block_generator")

    # compute number of geo blocks in x and y directions
    n_geo_block_y = int(np.ceil(geo_grid.length / geo_lines_per_block))
    n_geo_block_x = int(np.ceil(geo_grid.width / geo_cols_per_block))
    n_blocks = n_geo_block_x * n_geo_block_y

    # compute length and width of geo block
    geo_block_length = geo_lines_per_block * geo_grid.spacing_y
    geo_block_width = geo_cols_per_block * geo_grid.spacing_x

    # iterate over number of y geo blocks
    # *_end is open
    for i_blk_y in range(n_geo_block_y):

        # compute start index and end index for current geo y block
        # use min to account for last block
        y_start_index = i_blk_y * geo_lines_per_block
        y_end_index = min(y_start_index + geo_lines_per_block, geo_grid.length)
        blk_geogrid_length = y_end_index - y_start_index

        # compute start and length along x for current geo block
        y_start = geo_grid.start_y + i_blk_y * geo_block_length

        # iterate over number of x geo blocks
        for i_blk_x in range(n_geo_block_x):

            # log current block info
            i_blk = i_blk_x * n_geo_block_y + i_blk_x + 1
            info_channel.log(f"running geocode SLC array block {i_blk} of {n_blocks}")

            # compute start index and end index for current geo x block
            # use min to catch last block
            x_start_index = i_blk_x * geo_cols_per_block
            x_end_index = min(x_start_index + geo_cols_per_block, geo_grid.width)
            blk_geogrid_width = x_end_index - x_start_index

            # compute start and width along y for current geo block
            x_start = geo_grid.start_x + i_blk_x * geo_block_width

            # create geogrid for current geo block
            blk_geo_grid = isce3.product.GeoGridParameters(x_start, y_start,
                                                           geo_grid.spacing_x,
                                                           geo_grid.spacing_y,
                                                           blk_geogrid_width,
                                                           blk_geogrid_length,
                                                           geo_grid.epsg)

            # compute radar bounding box for current geo block
            try:
                bbox = isce3.geometry.get_radar_bbox(blk_geo_grid, radar_grid,
                                                     orbit, dem_interp,
                                                     geogrid_expansion_threshold=geogrid_expansion_threshold)
            except RuntimeError:
                info_channel.log(f"no radar data found for block {i_blk} of {n_blocks}")
                # skip this geo block if no radar data is found
                continue

            # return radar block bounding box/geo block indices pair
            radar_slice = np.s_[bbox.first_azimuth_line:bbox.last_azimuth_line,
                                bbox.first_range_sample:bbox.last_range_sample]
            geo_slice = np.s_[y_start_index:y_end_index,
                              x_start_index:x_end_index]
            geo_block_shape = (blk_geogrid_length, blk_geogrid_width)
            yield (radar_slice, geo_slice, geo_block_shape, blk_geo_grid)


def run(cfg):
    '''
    run geocodeSlc according to parameters in cfg dict
    '''
    # pull parameters from cfg
    input_hdf5 = cfg['input_file_group']['input_file_path']
    output_hdf5 = cfg['product_path_group']['sas_output_file']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    geogrids = cfg['processing']['geocode']['geogrids']
    radar_grid_cubes_geogrid = cfg['processing']['radar_grid_cubes']['geogrid']
    radar_grid_cubes_heights = cfg['processing']['radar_grid_cubes']['heights']
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']
    orbit_file = cfg["dynamic_ancillary_file_group"]['orbit_file']
    threshold_geo2rdr = cfg['processing']['geo2rdr']['threshold']
    iteration_geo2rdr = cfg['processing']['geo2rdr']['maxiter']
    columns_per_block = cfg['processing']['blocksize']['x']
    lines_per_block = cfg['processing']['blocksize']['y']
    flatten = cfg['processing']['flatten']
    geogrid_expansion_threshold = 100

    output_dir = os.path.dirname(os.path.abspath(output_hdf5))
    os.makedirs(output_dir, exist_ok=True)

    # init parameters shared by frequency A and B
    slc = SLC(hdf5file=input_hdf5)

    # if provided, load an external orbit from the runconfig file;
    # othewise, load the orbit from the RSLC metadata
    if orbit_file is not None:
        orbit = load_orbit_from_xml(orbit_file)
    else:
        orbit = slc.getOrbit()
    dem_raster = isce3.io.Raster(dem_file)
    dem_interp = isce3.geometry.DEMInterpolator(dem_raster)
    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    # Doppler of the image grid (Zero for NISAR)
    image_grid_doppler = isce3.core.LUT2d()

    info_channel = journal.info("gslc_array.run")
    info_channel.log("starting geocode SLC")

    t_all = time.time()
    with h5py.File(output_hdf5, 'a') as dst_h5, \
            h5py.File(input_hdf5, 'r', libver='latest', swmr=True) as src_h5:
        for freq, pol_list in freq_pols.items():
            frequency = f"frequency{freq}"
            radar_grid = slc.getRadarGrid(freq)
            geo_grid = geogrids[freq]

            # get doppler centroid
            native_doppler = slc.getDopplerCentroid(frequency=freq)

            # loop over polarizations
            for polarization in pol_list:
                t_pol = time.time()

                # path and dataset to rdr SLC data in HDF5
                rslc_ds_path = slc.slcPath(freq, polarization)
                rslc_ds = src_h5[rslc_ds_path]

                # path and dataset to geo SLC data in HDF5
                dataset_path = f'/science/LSAR/GSLC/grids/{frequency}/{polarization}'
                gslc_dataset = dst_h5[dataset_path]

                # loop over blocks
                for (rdr_blk_slice, geo_blk_slice, geo_blk_shape, blk_geo_grid) in \
                     _block_generator(geo_grid, radar_grid, orbit,
                                      dem_interp,  lines_per_block,
                                      columns_per_block,
                                      geogrid_expansion_threshold):

                    # unpack block parameters
                    az_first = rdr_blk_slice[0].start
                    rg_first = rdr_blk_slice[1].start

                    # prepare zero'd GSLC data block/array
                    gslc_data_blk = np.zeros(geo_blk_shape, dtype=np.complex64)

                    # extract RSLC data block/array
                    rslc_data_blk = nisar.types.read_c4_dataset_as_c8(rslc_ds,
                                                                      rdr_blk_slice)

                    # run geocodeSlc
                    isce3.geocode.geocode_slc(gslc_data_blk, rslc_data_blk,
                                              dem_raster, radar_grid, blk_geo_grid,
                                              orbit, native_doppler,
                                              image_grid_doppler, ellipsoid,
                                              threshold_geo2rdr,
                                              iteration_geo2rdr,
                                              az_first, rg_first, flatten)

                    # write to GSLC block HDF5
                    gslc_dataset.write_direct(gslc_data_blk,
                                              dest_sel=geo_blk_slice)

                gslc_raster = isce3.io.Raster(f"IH5:::ID={gslc_dataset.id.id}".encode("utf-8"), update=True)
                compute_stats_complex_data(gslc_raster, gslc_dataset)

                t_pol_elapsed = time.time() - t_pol
                info_channel.log(f'polarization {polarization} ran in {t_pol_elapsed:.3f} seconds')

        cube_geogrid = isce3.product.GeoGridParameters(
            start_x=radar_grid_cubes_geogrid.start_x,
            start_y=radar_grid_cubes_geogrid.start_y,
            spacing_x=radar_grid_cubes_geogrid.spacing_x,
            spacing_y=radar_grid_cubes_geogrid.spacing_y,
            width=int(radar_grid_cubes_geogrid.width),
            length=int(radar_grid_cubes_geogrid.length),
            epsg=radar_grid_cubes_geogrid.epsg)

        cube_group_name = '/science/LSAR/GSLC/metadata/radarGrid'

        # if available use frequency A to get radar grid and native doppler
        # else use frequency B
        cube_freq = "A" if "A" in freq_pols else "B"
        cube_rdr_grid = slc.getRadarGrid(cube_freq)
        cube_native_doppler = slc.getDopplerCentroid(frequency=cube_freq)
        cube_native_doppler.bounds_error = False
        # The native-Doppler LUT bounds error is turned off to
        # computer cubes values outside radar-grid boundaries
        add_radar_grid_cubes_to_hdf5(dst_h5, cube_group_name,
                                     cube_geogrid, radar_grid_cubes_heights,
                                     cube_rdr_grid, orbit, cube_native_doppler,
                                     image_grid_doppler, threshold_geo2rdr,
                                     iteration_geo2rdr)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran geocode SLC in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    gslc_runcfg = GSLCRunConfig(args)
    h5_prep.run(gslc_runcfg.cfg)
    run(gslc_runcfg.cfg)
