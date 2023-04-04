import os
import isce3
import h5py
import numpy as np
from osgeo import gdal

from isce3.core.block_param_generator import (block_param_generator,
                                              get_raster_block,
                                              write_raster_block)

def np2gdal_dtype(np_dtype):
    dict_np2gdal = {
        np.byte: gdal.GDT_Byte,
        np.ushort: gdal.GDT_UInt16,
        np.short: gdal.GDT_Int16,
        np.uintc: gdal.GDT_UInt32,
        np.intc: gdal.GDT_Int32,
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64,
        np.complex64: gdal.GDT_CFloat32,
        np.complex128: gdal.GDT_CFloat64}
    if np_dtype not in dict_np2gdal:
        # throw unsupported error
        pass
    else:
        return dict_np2gdal[int_dtype]


def get_raster_info(raster):
    ''' Determine raster shape based on raster
        type (h5py.Dataset or GDAL-friendly raster).

    Parameters
    ----------
    raster: h5py.Dataset or str
        Raster whose size is to be determined. String value represents
        filepath for GDAL rasters.

    Returns
    -------
        data_width: int
            Width of raster.
        data_length: int
            Length of raster.
    '''
    if isinstance(raster, h5py.Dataset):
        return raster.shape, raster.dtype
    else:
        # Open input data using GDAL to get raster length
        ds = gdal.Open(raster, gdal.GA_ReadOnly)
        data_length = ds.RasterYSize
        data_width = ds.RasterXSize
        data_type = ds.GetRasterBand(1).DataType
        return (data_length, data_width), data_type


def filter_data(input_data, lines_per_block,
           kernel_rows, kernel_cols, output_data=None, mask_path=None):
    ''' Filter data using two separable 1D kernels.

    Parameters
    ----------
    input_data: str
        File path to input data raster (GDAL-friendly)
    lines_per_block: int
        Number of lines to process in batch
    kernel_rows: float array
        1D kernel along rows direction
    kernel_cols: float array
        1D kernel along columns direction
    output_data: h5py.Dataset or str
        Raster where a block needs to be written to. String value represents
        file path for GDAL rasters. If not provided, input_data is overwritten
        with the output filtered data
    mask_path: str
        Filepath to the mask to use during filtering

    Returns
    -------
    '''

    data_shape, data_type = get_raster_info(input_data)
    data_length, data_width = data_shape

    # Determine the amount of padding
    pad_length = 2 * (len(kernel_rows) // 2)
    pad_width = 2 * (kernel_cols.shape[1] // 2)
    pad_shape = (pad_length, pad_width)

    # Determine number of blocks to process
    lines_per_block = min(data_length,
                          lines_per_block)

    # Start block processing
    block_params = block_param_generator(lines_per_block, data_shape, pad_shape)
    for block_param in block_params:
        # Read a block of data. If hdf5_dset is set, read a block of data
        # directly from the hdf5 file. Otherwise, use gdal to read block of data
        data_block = get_raster_block(input_data, block_param)

        # Get if filtering needs to be performed with or without a mask
        if mask_path is not None:
            # Use gdal to extract a mask block, pad the mask (mask need to be same shape as input)
            ds_mask = gdal.Open(mask_path,
                                gdal.GA_ReadOnly)
            mask_block = ds_mask.GetRasterBand(1).ReadAsArray(0,
                                                              block_param.read_start_line,
                                                              block_param.data_width,
                                                              block_param.read_length)
            mask_block = np.pad(mask_block, block_param.block_pad,
                                mode='constant', constant_values=0)
            filt_data_block = isce3.signal.convolve2D(data_block,
                                                      mask_block,
                                                      kernel_cols,
                                                      kernel_rows,
                                                      False)
        else:
            filt_data_block = isce3.signal.convolve2D(data_block,
                                                      kernel_cols,
                                                      kernel_rows,
                                                      False)
        # If no value provided for output_data, then overwrite existing
        # input with filtered output
        # Otherwise write filtered output to output_data
        out_raster = input_data if output_data is None else output_data

        # If writing to GDAL raster, prepare file
        if not isinstance(out_raster, h5py.Dataset) and not os.path.isfile(out_raster):
            raster = isce3.io.Raster(path=out_raster, width=data_width,
                                     length=data_length, num_bands=1,
                                     dtype=data_type, driver_name='GTiff')
            del raster

        write_raster_block(out_raster, filt_data_block, block_param)


def create_gaussian_kernel(size, sigma):
    '''
    Create 1D gaussian kernel given kernel size
    and standard deviation
    '''
    array = np.arange(-int(size / 2), int(size / 2) + 1)
    return np.asarray([1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -float(x) ** 2 / (2 * sigma ** 2))
                       for x in array])
