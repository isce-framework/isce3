import isce3
import numpy as np
from osgeo import gdal


def compute_stats_complex_data(raster, h5_ds):
    """
    Compute raster statistics for complex datasets

    raster: isce3.io.Raster
        ISCE3 Raster object
    h5_ds: h5py.File
        h5py file
    """
    stats_obj = isce3.math.compute_raster_stats_real_imag(raster)[0]
    write_stats_complex_data(h5_ds, stats_obj)


def write_stats_complex_data(h5_ds, stats_obj):
    h5_ds.attrs.create('min_real_value', data=stats_obj.real.min)
    h5_ds.attrs.create('mean_real_value', data=stats_obj.real.mean)
    h5_ds.attrs.create('max_real_value', data=stats_obj.real.max)
    h5_ds.attrs.create('sample_stddev_real',
                       data=stats_obj.real.sample_stddev)
    h5_ds.attrs.create('min_imag_value', data=stats_obj.imag.min)
    h5_ds.attrs.create('mean_imag_value', data=stats_obj.imag.mean)
    h5_ds.attrs.create('max_imag_value', data=stats_obj.imag.max)
    h5_ds.attrs.create('sample_stddev_imag',
                       data=stats_obj.imag.sample_stddev)


def compute_stats_real_data(raster, h5_ds):
    """
    Compute raster statistics for real datasets

    raster: isce3.io.Raster
       ISCE3 Raster object
    h5_ds: h5py.File
        h5py file
    """
    if raster.datatype() == gdal.GDT_Float64:
        stats_obj = isce3.math.compute_raster_stats_float64(raster)[0]
    else:
        stats_obj = isce3.math.compute_raster_stats_float32(raster)[0]
    h5_ds.attrs.create('min_value', data=stats_obj.min)
    h5_ds.attrs.create('mean_value', data=stats_obj.mean)
    h5_ds.attrs.create('max_value', data=stats_obj.max)
    h5_ds.attrs.create('sample_stddev',
                       data=stats_obj.sample_stddev)


def compute_water_mask_stats(h5_ds, lines_per_block=1000):
    '''
    Compute statistics for water mask layer in GUNW
    Statistics correspond to
    1. Percentage of pixels with water (labelled with 1)
    2. Percentage of pixels on land (labelled with 0)

    Parameters
    ----------
    h5_ds: h5py.Dataset
        water mask HDF5 dataset
    lines_per_block: int
        Number of lines to process in batch
    '''

    length, width = h5_ds.shape
    land_pixs = 0
    water_pixs = 0
    valid_pixs = 0

    # Use block processing to count pixels on land (0) and water (1)
    lines_per_block = min(length, lines_per_block)
    num_blocks = int(np.ceil(length / lines_per_block))

    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = length - line_start
        else:
            block_length = lines_per_block

        data_block = np.empty((block_length, width), dtype=h5_ds.dtype)
        h5_ds.read_direct(data_block,
                          np.s_[line_start:line_start + block_length])
        water = data_block == 1
        land = data_block == 0
        water_pixs += np.count_nonzero(water)
        land_pixs += np.count_nonzero(land)
        # We are counting valid pixels (outside the
        # geocoded area we have np.nan)
        valid_pixs += water_pixs + land_pixs

    percent_water_pixs = (water_pixs / (valid_pixs)) * 100
    percent_land_pixs = (land_pixs / (valid_pixs)) * 100
    h5_ds.attrs.create('percentage_water_pixels',
                       data=percent_water_pixs)
    h5_ds.attrs.create('percentage_land_pixels',
                       data=percent_land_pixs)


def compute_layover_shadow_stats(h5_ds, lines_per_block=1000):
    '''
    Compute statistics for layover/shadow dataset
    1. Percentage of pixels in layover (labelled with 1)
    2. Percentage of pixels in shadow (labelled with 2)
    3. Percentage of pixels in layover and shadow (labelled with 3)

    Parameters
    ----------
    h5_ds: h5py.File
        layover shadow mask HDF5 dataset
    lines_per_block: int
        Number of lines to process in batch
    '''

    length, width = h5_ds.shape
    layover_pixs, shadow_pixs, layover_shadow_pixs, valid_pixs = 0, 0, 0, 0

    # Use block processing to count pixels on land (0) and water (1)
    lines_per_block = min(length, lines_per_block)
    num_blocks = int(np.ceil(length / lines_per_block))

    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = length - line_start
        else:
            block_length = lines_per_block

        data_block = np.empty((block_length, width), dtype=h5_ds.dtype)
        h5_ds.read_direct(data_block,
                          np.s_[line_start:line_start + block_length])
        no_layover_shadow = data_block == 0
        layover = data_block == 1
        shadow = data_block == 2
        layover_shadow = data_block == 3
        layover_pixs += np.count_nonzero(layover)
        shadow_pixs += np.count_nonzero(shadow)
        layover_shadow_pixs += np.count_nonzero(layover_shadow)
        valid_pixs = np.count_nonzero(
            no_layover_shadow) + layover_pixs + shadow_pixs + layover_shadow_pixs

    percent_layover_pixs = (layover_pixs / (length * width)) * 100
    percent_shadow_pixs = (shadow_pixs / (length * width)) * 100
    percent_layover_shadow_pixs = (layover_shadow_pixs / (length * width)) * 100

    h5_ds.attrs.create('percentage_layover_pixels',
                       data=percent_layover_pixs)
    h5_ds.attrs.create('percentage_shadow_pixels',
                       data=percent_shadow_pixs)
    h5_ds.attrs.create('percentage_layover_shadow_pixels',
                       data=percent_layover_shadow_pixs)
