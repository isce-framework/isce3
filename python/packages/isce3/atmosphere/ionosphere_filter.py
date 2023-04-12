import os

import h5py
import isce3
import numpy as np
from osgeo import gdal
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt

from isce3.core.block_param_generator import block_param_generator
from isce3.signal.filter_data import create_gaussian_kernel, get_raster_info

class IonosphereFilter:
    '''
    Filter ionospheric phase screen
    '''
    def __init__(self,
                 x_kernel,
                 y_kernel,
                 sig_x,
                 sig_y,
                 iteration=1,
                 filling_method='nearest',
                 outputdir='.'):
        """Initialized IonosphereFilter with filter options

        Parameters
        ----------
        x_kernel : int
            x kernel size for gaussian filtering
        y_kernel : int
            y kernel size for gaussian filtering
        sig_x : int
            x standard deviation for gaussian window
        sig_y : int
            y standard deviation for gaussian window
        iteration : int
            number of iterations for filtering
        filling_method : str {'nearest', 'smoothed'}
            filling gap method for masked area
        outputdir : str
            output directory for filtered dispersive
        """
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel
        self.sig_x = sig_x
        self.sig_y = sig_y
        self.iteration = iteration
        self.filling_method = filling_method
        self.outputdir = outputdir

    def low_pass_filter(self,
            input_data,
            input_std_dev,
            mask_path,
            filtered_output,
            filtered_std_dev,
            lines_per_block):
        """Apply low_pass_filtering for dispersive and nondispersive
        with standard deviation. Before filtering, fill the gaps with
        smoothed or nearest values.

        Parameters
        ----------
        input_data : str
            file path for data to be filtered.
        input_std_dev : str
            file path for stardard deviation
            or nondispersive array
        mask_path : str
            file path for mask raster
            1: valid pixels,
            0: invalid pixels.
        filtered_output : str
            output file path or h5py dataset to write the filtered data
        filtered_std_dev : str
            output file path or h5py dataset to write filtered standard deviation.

        Returns
        -------
        """
        data_shape, _ = get_raster_info(input_data)
        data_length, data_width = data_shape
        # Determine number of blocks to process
        lines_per_block = min(data_length,
                            lines_per_block)
        # Determine the amount of padding
        pad_length = 2 * (self.y_kernel // 2)
        pad_width = 2 * (self.x_kernel// 2)
        pad_shape = (pad_length, pad_width)

        # Prepare to write output to files
        for output in [filtered_output, filtered_std_dev]:
            if not isinstance(output, h5py.Dataset) and \
                not os.path.isfile(output):
                raster = isce3.io.Raster(path=output,
                    width=data_width,
                    length=data_length,
                    num_bands=1,
                    dtype=gdal.GDT_Float32,
                    driver_name='ENVI')
                del raster

        for iter_cnt in range(self.iteration):

            block_params = block_param_generator(
                lines_per_block, data_shape, pad_shape)
            # Start block processing
            for block_param in block_params:
                # Prepare to write temp_files
                filtered_iono_temp_input_path = f'{self.outputdir}/filtered_iono_temp{iter_cnt-1}'
                filtered_std_temp_input_path = f'{self.outputdir}/filtered_iono_std_temp{iter_cnt-1}'

                block_data_path = filtered_iono_temp_input_path if iter_cnt > 0 else input_data
                data_block = read_block_array(block_data_path, block_param)
                block_sig_path = filtered_std_temp_input_path if iter_cnt > 0 else input_std_dev
                data_sig_block = read_block_array(block_sig_path, block_param)
                mask_block = read_block_array(mask_path, block_param, fill_value=1)
                mask0 = mask_block == 0
                mask1 = mask_block == 1
                data_block[mask0] = np.nan
                data_sig_block[mask0] = np.nan

                # filling gaps with smoothed or nearest values
                fill_method = fill_with_smoothed \
                    if self.filling_method == "smoothed" else fill_nearest
                filled_data = fill_method(data_block)
                filled_data_sig = fill_method(data_sig_block)

                if iter_cnt > 0 :
                    # Replace the valid pixels with original unfiltered data
                    # to avoid too much smoothed signal
                    unfilt_data_block = read_block_array(input_data, block_param)
                    filled_data[mask1] = unfilt_data_block[mask1]
                    unfilt_data_block = read_block_array(input_std_dev, block_param)
                    filled_data_sig[mask1] = unfilt_data_block[mask1]

                # after filling gaps, filter the data
                filt_data, filt_data_sig = filter_data_with_sig(
                    input_array=filled_data,
                    sig_array=filled_data_sig,
                    kernel_width=self.x_kernel,
                    kernel_length=self.y_kernel,
                    sig_kernel_x=self.sig_x,
                    sig_kernel_y=self.sig_y)

                # set output to HDF5 for final iteration
                # otherwise write to temp file
                if iter_cnt == self.iteration - 1 :
                    output_iono = filtered_output
                    output_std = filtered_std_dev
                else:
                    output_iono = f'{self.outputdir}/filtered_iono_temp{iter_cnt}'
                    output_std = f'{self.outputdir}/filtered_iono_std_temp{iter_cnt}'

                write_array(output_iono, filt_data,
                    block_row=block_param.write_start_line,
                    data_shape=data_shape)

                write_array(output_std, filt_data_sig,
                    block_row=block_param.write_start_line,
                    data_shape=data_shape)

def fill_with_smoothed(data):
    """Replace the value of nan 'data' cells
    by the value of the linear interpolated data cell.
    The values, not covered by interpolation, are filled
    with nearest values.

    Parameters
    ----------
    data : numpy.ndarray
        array containing holes to be filled.
        nan values are considered as holes.

    Returns
    -------
    numpy.ndarray
        array with no data values filled with data values
        from numpy.griddata
    """
    rows, cols = data.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    xx, yy = np.meshgrid(x, y)

    xx = xx.ravel()
    yy = yy.ravel()
    data = data.ravel()

    is_nan_mask = np.isnan(data)
    not_nan_mask = np.invert(is_nan_mask)

    if np.all(not_nan_mask):
        return data.reshape([rows, cols])

    # find x and y where valid values are located.
    xx_wo_nan = xx[not_nan_mask]
    yy_wo_nan = yy[not_nan_mask]
    data_wo_nan = data[not_nan_mask]

    xnew = xx[np.isnan(data)]
    ynew = yy[np.isnan(data)]

    # linear interpolation with griddata
    znew = griddata((xx_wo_nan, yy_wo_nan),
                    data_wo_nan,
                    (xnew, ynew),
                    method='linear')
    data_filt = data.copy()
    data_filt[np.isnan(data)] = znew
    n_nonzero = np.sum(np.count_nonzero(np.isnan(data_filt)))

    if n_nonzero > 0:
        idx2= np.isnan(data_filt)

        xx_wo_nan = xx[np.invert(idx2)]
        yy_wo_nan = yy[np.invert(idx2)]
        data_wo_nan = data_filt[np.invert(idx2)]
        xnew = xx[idx2]
        ynew = yy[idx2]

        # extrapolation using nearest values
        znew_ext = griddata((xx_wo_nan, yy_wo_nan),
            data_wo_nan, (xnew, ynew), method='nearest')
        data_filt[np.isnan(data_filt)] = znew_ext
    return data_filt.reshape([rows, cols])

def filter_data_with_sig(
        input_array,
        sig_array,
        kernel_width,
        kernel_length,
        sig_kernel_x,
        sig_kernel_y,
        mask_array=None):
    """ Filter input array by applying weighting
    based on the statndard deviations
    Parameters
    ----------
    input_array : numpy.ndarray
        2D dispersive or nondispersive array
    sig_array : numpy.ndarray
        2D standard deviation array of dispersive
        or nondispersive array
    kernel_width : int
        x kernel size for gaussian filtering
    kernel_length : int
        y kernel size for gaussian filtering
    sig_kernel_x : int
        x standard deviation for gaussian window
    sig_kernel_y : int
        y standard deviation for gaussian window

    Returns
    -------
    filt_data : numpy.ndarray
        2D filtered image
    filt_data_sig : numpy.ndarray
        2D filtered standard deviation image
    """
    # Create Gaussian kernel for filtering
    kernel_rows = create_gaussian_kernel(kernel_length, sig_kernel_y)
    kernel_rows = np.reshape(kernel_rows, (len(kernel_rows), 1))
    kernel_cols = create_gaussian_kernel(kernel_width, sig_kernel_x)
    kernel_cols = np.reshape(kernel_cols, (1, len(kernel_cols)))

    sig_array_sqr = sig_array**2
    input_div_sig = np.divide(input_array,
        sig_array_sqr,
        out=np.zeros_like(input_array),
        where=sig_array_sqr!=0)

    inv_sig = np.divide(1,
        sig_array_sqr,
        out=np.zeros_like(sig_array_sqr),
        where=sig_array_sqr!=0)

    if mask_array is not None:
        filt_input_div_sig = isce3.signal.convolve2D(
            input_div_sig,
            mask_array,
            kernel_cols,
            kernel_rows,
            False)

        filt_inv_sig = isce3.signal.convolve2D(
            inv_sig,
            mask_array,
            kernel_cols,
            kernel_rows,
            False)

        filt_inv_sig_kernel2 = isce3.signal.convolve2D(
            inv_sig,
            mask_array,
            kernel_cols**2,
            kernel_rows**2,
            False)

    else:
        filt_input_div_sig = isce3.signal.convolve2D(
            input_div_sig,
            kernel_cols,
            kernel_rows,
            False)

        filt_inv_sig = isce3.signal.convolve2D(
            inv_sig,
            kernel_cols,
            kernel_rows,
            False)

        filt_inv_sig_kernel2 = isce3.signal.convolve2D(
            inv_sig,
            kernel_cols**2,
            kernel_rows**2,
            False)

    filt_data = np.divide(filt_input_div_sig, filt_inv_sig,
        out=np.zeros_like(filt_input_div_sig),
        where=filt_inv_sig!=0)

    filt_data_sig = np.divide(filt_inv_sig_kernel2, filt_inv_sig**2,
        out=np.zeros_like(filt_inv_sig_kernel2),
        where=filt_inv_sig!=0)

    filt_data_sig = np.sqrt(filt_data_sig)

    return filt_data, filt_data_sig

def read_block_array(raster, block_param, fill_value=0):
    ''' Get a block of data from raster.
        Raster can be a HDF5 file or a GDAL-friendly raster

    Parameters
    ----------
    raster: h5py.Dataset or str
        Raster where a block is to be read from. String value represents a
        filepath for GDAL rasters.
    block_param: BlockParam
        Object specifying size of block and where to read from raster,
        and amount of padding for the read array
    fill_value: float
        Pads with a fill value.
    Returns
    -------
    data_block: np.ndarray
        Block read from raster with shape specified in block_param.
    '''
    if isinstance(raster, h5py.Dataset):
        data_block = np.empty((block_param.read_length, block_param.data_width),
                           dtype=raster.dtype)
        raster.read_direct(data_block,
            np.s_[block_param.read_start_line:
                block_param.read_start_line + block_param.read_length,
                0: block_param.data_width])
    else:
        # Open input data using GDAL to get raster length
        ds_data = gdal.Open(raster, gdal.GA_Update)
        data_block = ds_data.GetRasterBand(1).ReadAsArray(0,
                                                block_param.read_start_line,
                                                block_param.data_width,
                                                block_param.read_length)

    # Pad igram_block with zeros according to pad_length/pad_width
    data_block = np.pad(data_block, block_param.block_pad,
                         mode='constant', constant_values=fill_value)

    return data_block

def write_array(output_str,
        input_array,
        data_type=gdal.GDT_Float32,
        data_shape=None,
        block_row=0,
        file_type='ENVI'):
    """write block array to file with gdal

    Parameters
    ----------
    output_str : str
        output file name with path
    input_array : numpy.ndarray
        2D array to be written to file
    data_type : str
        gdal raster type
    data_shape : list
        raster shape, [rows, cols]
    block_row : int
        block index
    """
    rows, cols = input_array.shape
    if data_shape is not None:
        data_rows, data_cols = data_shape

    if isinstance(output_str, h5py.Dataset):
        output_str.write_direct(input_array,
                      dest_sel=np.s_[
                               block_row:block_row+rows,
                               0:cols])
    else:
        if block_row == 0:
            driver = gdal.GetDriverByName(file_type)
            ds_data = driver.Create(output_str, data_cols, data_rows, 1, data_type)
            ds_data.WriteArray(input_array, xoff=0, yoff=block_row)

        else:
            ds_data = gdal.Open(output_str, gdal.GA_Update)
            ds_data.WriteArray(input_array, xoff=0, yoff=block_row)

        ds_data = None
        del ds_data


def fill_nearest(data, invalid=None):
    """Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Parameters
    ----------
    data : numpy.ndarray
        array containing holes to be filled.
    invalid:
        a binary array of same shape as 'data'.
        data value are replaced where invalid is True
        If None (default), use: invalid  = np.isnan(data)

    Returns
    -------
    data[tuple(ind)]: numpy.ndarray
        array with no data values filled with data values
        from nearest neighborhood
    """
    if invalid is None:
        invalid = np.isnan(data)

    ind = distance_transform_edt(invalid,
                                return_distances=False,
                                return_indices=True)
    return data[tuple(ind)]
