import journal
import numpy as np
import os

import isce3

def block_generator(geo_grid, radar_grid, orbit, dem_raster,
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
    min_height: float
        Min height of DEM
    max_height: float
        Max height of DEM
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
    info_channel = journal.info("rdr_geo_block_generator.block_generator")

    # compute number of geo blocks in x and y directions
    n_geo_block_y = int(np.ceil(geo_grid.length / geo_lines_per_block))
    n_geo_block_x = int(np.ceil(geo_grid.width / geo_cols_per_block))
    n_blocks = n_geo_block_x * n_geo_block_y

    # compute length and width of geo block
    geo_block_length = geo_lines_per_block * geo_grid.spacing_y
    geo_block_width = geo_cols_per_block * geo_grid.spacing_x

    # compute max and min of DEM to use as extreme starting guesses for geo2rdr
    # within get_radar_bbox
    dem_interp = isce3.geometry.DEMInterpolator(dem_raster)
    dem_interp.compute_min_max_mean_height()

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
            i_blk = i_blk_x * n_geo_block_y + i_blk_y + 1
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
                                                     orbit,
                                                     dem_interp.min_height,
                                                     dem_interp.max_height,
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
