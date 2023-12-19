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

def compute_stats_real_hdf5_dataset(h5_ds):
    """
    Compute raster statistics for hdf5 real datasets

    Parameters
    ----------
    h5_ds: h5py.Dataset
        hdf5 dataset object
    """
    h5_ds.attrs.create('min_value', data=np.nanmin(h5_ds))
    h5_ds.attrs.create('mean_value', data=np.nanmean(h5_ds))
    h5_ds.attrs.create('max_value', data=np.nanmax(h5_ds))
    h5_ds.attrs.create('sample_stddev', data=np.nanstd(h5_ds))

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

def compute_layover_shadow_water_stats(h5_ds, lines_per_block=1000):
    '''
    Compute statistics for masks containing layover/shadow and water
    1. Percentage of pixels in shadow (labeled with 1 or 5)
    2. Percentage of pixels in layover (labeled with 2 or 6)
    3. Percentage of pixels in layover and shadow (labeled with 3 or 7)
    4. Percentage of pixels in water (labeled with 4, 5, 6, or 7)
    Parameters
    ----------
    h5_ds: h5py.File
        mask HDF5 dataset
    lines_per_block: int
        Number of lines to process in batch
    '''
    shadow_value = 1
    layover_value = 2
    layover_shadow_value = 3
    water_value = 4
    shadow_water_value = water_value + shadow_value
    layover_water_value = water_value + layover_value
    layover_shadow_water_value = water_value + layover_shadow_value

    length, width = h5_ds.shape
    layover_pixs, shadow_pixs, layover_shadow_pixs, water_pixs, valid_pixs = \
        0, 0, 0, 0, 0

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

        layover = (data_block == layover_value) | (data_block == layover_water_value)
        shadow = (data_block == shadow_value) | (data_block == shadow_water_value)
        layover_shadow = (data_block == layover_shadow_value) | \
                         (data_block == layover_shadow_water_value)
        water = (data_block == water_value) | \
                (data_block == shadow_water_value)| \
                (data_block == layover_water_value) | \
                (data_block == layover_shadow_water_value)
        valid = (data_block >= 0) & (data_block <= 7)

        valid_pixs += np.count_nonzero(valid)
        layover_pixs += np.count_nonzero(layover)
        shadow_pixs += np.count_nonzero(shadow)
        layover_shadow_pixs += np.count_nonzero(layover_shadow)
        water_pixs += np.count_nonzero(water)

    # If there are no valid pixels, the percentage could be represented as "inf."
    # Additionally, when the number of samples is only one,
    # the percentage becomes meaningless.
    # Therefore, in such cases, we assign "nan" to the valid_pixs variable.
    if (valid_pixs == 0) or (length * width == 1):
        valid_pixs = np.nan

    percent_layover_pixs = round((layover_pixs / valid_pixs) * 100, 2)
    percent_shadow_pixs = round((shadow_pixs / valid_pixs) * 100, 2)
    percent_layover_shadow_pixs = \
        round((layover_shadow_pixs / valid_pixs) * 100, 2)
    percent_water_pixs = round((water_pixs / valid_pixs) * 100, 2)

    h5_ds.attrs.create('percentage_layover_pixels',
                       data=percent_layover_pixs)
    h5_ds.attrs.create('percentage_shadow_pixels',
                       data=percent_shadow_pixs)
    h5_ds.attrs.create('percentage_layover_shadow_pixels',
                       data=percent_layover_shadow_pixs)
    h5_ds.attrs.create('percentage_water_pixels',
                       data=percent_water_pixs)