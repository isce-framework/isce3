import os

import h5py
import journal
from multiprocessing import Pool
import numpy as np
from osgeo import gdal
from scipy.interpolate import griddata
from scipy.ndimage import (distance_transform_edt,
                           convolve,
                           label,
                           find_objects,
                           generic_filter)

import isce3
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

    def low_pass_filter(
            self,
            input_data,
            input_std_dev,
            mask_path,
            filtered_output,
            filtered_std_dev,
            lines_per_block,
            min_cluster_pixels):
        """Apply low_pass_filtering for dispersive and nondispersive
        with standard deviation. Before filtering, fill the gaps with
        smoothed or nearest values.

        Parameters
        ----------
        input_data : str
            file path for data to be filtered.
        input_std_dev : str
            file path for standard deviation
            or nondispersive array
        mask_path : str
            file path for mask raster
            1: valid pixels,
            0: invalid pixels.
        filtered_output : str
            output file path or h5py dataset to write the filtered data
        filtered_std_dev : str
            output file path or h5py dataset to write filtered standard
            deviation.

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
        pad_width = 2 * (self.x_kernel // 2)
        pad_shape = (pad_length, pad_width)

        # Prepare to write output to files
        for output in [filtered_output, filtered_std_dev]:
            if not isinstance(output, h5py.Dataset) and \
               not os.path.isfile(output):
                raster = isce3.io.Raster(
                    path=output,
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
                width_offset = pad_width // 2
                length_offset = pad_length // 2

                # Prepare to write temp_files
                filtered_iono_temp_input_path = \
                    f'{self.outputdir}/filtered_iono_temp{iter_cnt-1}'
                filtered_std_temp_input_path = \
                    f'{self.outputdir}/filtered_iono_std_temp{iter_cnt-1}'

                block_data_path = filtered_iono_temp_input_path \
                    if iter_cnt > 0 else input_data
                data_block = read_block_array(block_data_path, block_param)
                block_sig_path = filtered_std_temp_input_path \
                    if iter_cnt > 0 else input_std_dev
                data_sig_block = read_block_array(block_sig_path, block_param)
                mask_block = read_block_array(mask_path, block_param,
                                              fill_value=0)

                # buffer region represents the additional buffer areas for
                # the block.
                buffer_region = np.ones_like(mask_block, dtype=bool)
                buffer_region[
                    length_offset:length_offset+block_param.block_length,
                    width_offset:width_offset+block_param.data_width] = 0
                mask0 = mask_block == 0
                mask1 = mask0 == 0

                data_block[mask0] = np.nan

                # remove small areas from images to avoid
                # the possibility of unwrap errors
                data_block = remove_small_components(
                    data_block,
                    min_cluster_pixels=min_cluster_pixels)

                if self.filling_method == "smoothed":
                    fill_method = fill_with_smoothed
                elif self.filling_method == "nearest":
                    fill_method = fill_nearest
                elif self.filling_method == "distance":
                    fill_method = fill_with_distance

                if self.filling_method in ["distance"]:
                    weight = mask_block.astype('float')
                    filled_data = fill_method(data_block, weight)
                    filled_data_sig = fill_method(data_sig_block, weight)

                else:
                    filled_data_sig = fill_method(data_sig_block)
                    filled_data = fill_method(data_block)

                if iter_cnt > 0:
                    # Replace the valid pixels with original unfiltered data
                    # to avoid too much smoothed signal
                    unfilt_data_block = read_block_array(input_data,
                                                         block_param)
                    filled_data[mask1] = unfilt_data_block[mask1]
                    unfilt_data_block_sig = read_block_array(input_std_dev,
                                                             block_param)
                    filled_data_sig[mask1] = unfilt_data_block_sig[mask1]

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
                if iter_cnt == self.iteration - 1:
                    output_iono = filtered_output
                    output_std = filtered_std_dev
                else:
                    output_iono = \
                        f'{self.outputdir}/filtered_iono_temp{iter_cnt}'
                    output_std = \
                        f'{self.outputdir}/filtered_iono_std_temp{iter_cnt}'

                write_array(
                    output_iono,
                    filt_data,
                    block_row=block_param.write_start_line,
                    data_shape=data_shape)

                write_array(
                    output_std,
                    filt_data_sig,
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
        idx2 = np.isnan(data_filt)

        xx_wo_nan = xx[np.invert(idx2)]
        yy_wo_nan = yy[np.invert(idx2)]
        data_wo_nan = data_filt[np.invert(idx2)]
        xnew = xx[idx2]
        ynew = yy[idx2]

        # extrapolation using nearest values
        znew_ext = griddata((xx_wo_nan, yy_wo_nan),
                            data_wo_nan,
                            (xnew, ynew),
                            method='nearest')
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

    sig_array_sqr = sig_array ** 2
    input_div_sig = np.divide(
        input_array,
        sig_array_sqr,
        out=np.zeros_like(input_array),
        where=sig_array_sqr != 0)

    inv_sig = np.divide(
        1,
        sig_array_sqr,
        out=np.zeros_like(sig_array_sqr),
        where=sig_array_sqr != 0)

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

    filt_data = np.divide(filt_input_div_sig,
                          filt_inv_sig,
                          out=np.zeros_like(filt_input_div_sig),
                          where=filt_inv_sig != 0)

    filt_data_sig = np.divide(filt_inv_sig_kernel2,
                              filt_inv_sig**2,
                              out=np.zeros_like(filt_inv_sig_kernel2),
                              where=filt_inv_sig != 0)

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
        data_block = np.empty((block_param.read_length,
                               block_param.data_width),
                              dtype=raster.dtype)
        raster.read_direct(
            data_block,
            np.s_[block_param.read_start_line:
                  block_param.read_start_line + block_param.read_length,
                  0: block_param.data_width])
    else:
        # Open input data using GDAL to get raster length
        ds_data = gdal.Open(raster, gdal.GA_Update)
        data_block = ds_data.GetRasterBand(1).ReadAsArray(
            0,
            block_param.read_start_line,
            block_param.data_width,
            block_param.read_length)

    # Pad igram_block with zeros according to pad_length/pad_width
    data_block = np.pad(data_block, block_param.block_pad,
                        mode='constant', constant_values=fill_value)

    return data_block


def write_array(
        output_str,
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
        output_str.write_direct(
            input_array,
            dest_sel=np.s_[
                    block_row:block_row+rows,
                    0:cols])
    else:
        if block_row == 0:
            driver = gdal.GetDriverByName(file_type)
            ds_data = driver.Create(output_str,
                                    data_cols,
                                    data_rows,
                                    1,
                                    data_type)
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


def get_circle_idxs(
    max_radius: int, min_radius: int = 0, sort_output: bool = True
) -> np.ndarray:
    """Get the relative indices of neighboring pixels in a circular pattern.

    This function calculates the relative indices of pixels within a circle
    defined by a maximum radius. The indices can be sorted for better data
    access patterns or returned in the order they are found.

    Parameters
    ----------
    max_radius : int
        The maximum radius of the circle.
    min_radius : int, optional
        The minimum radius to start calculating indices, default is 0.
    sort_output : bool, optional
        Whether to sort the output indices, default is True.

    Returns
    -------
    np.ndarray
        An array of relative pixel indices within the specified circular
        pattern.

    Notes
    -----
    - This function uses a variation of the mid-point circle drawing algorithm
      to calculate the indices.
    - If `sort_output` is True, the indices are sorted to improve data access
      patterns, which can lead to faster runtime.
    - Adapted from the C++ version of the `psps` package:
      https://github.com/UT-Radar-Interferometry-Group/psps/blob/a15d458817fe7d06a6edaa0b3208ea78bc4782e7/src/cpp/similarity.cpp#L16
    """
    visited = np.zeros((max_radius, max_radius), dtype=bool)
    visited[0][0] = True

    indices = []
    for r in range(1, max_radius):
        x = r
        y = 0
        p = 1 - r
        if r > min_radius:
            indices.append([r, 0])
            indices.append([-r, 0])
            indices.append([0, r])
            indices.append([0, -r])

        visited[r][0] = True
        visited[0][r] = True
        # flag > 0 means there are holes between concentric circles
        flag = 0
        while x > y:
            # do not need to fill holes
            if flag == 0:
                y += 1
                if p <= 0:
                    # Mid-point is inside or on the perimeter
                    p += 2 * y + 1
                else:
                    # Mid-point is outside the perimeter
                    x -= 1
                    p += 2 * y - 2 * x + 1
            else:
                flag -= 1

            # All the perimeter points have already been visited
            if x < y:
                break

            while not visited[x - 1][y]:
                x -= 1
                flag += 1

            visited[x][y] = True
            visited[y][x] = True
            if r > min_radius:
                indices.append([x, y])
                indices.append([-x, -y])
                indices.append([x, -y])
                indices.append([-x, y])

                if x != y:
                    indices.append([y, x])
                    indices.append([-y, -x])
                    indices.append([y, -x])
                    indices.append([-y, x])

            if flag > 0:
                x += 1

    if sort_output:
        # Sorting makes it run faster, better data access patterns
        return np.array(sorted(indices))
    else:
        # Indices run from middle outward
        return np.array(indices)


def interpolate_row(row_data):
    """
    Interpolate missing or low-weight values in a row of an interferogram.

    This function interpolates values in a row of an interferogram based on the
    surrounding pixels and their weights. It uses a distance-based weighting
    scheme to estimate the value of a pixel if its weight is below a specified
    cutoff. The function is designed to handle complex interferometric data,
    but currently returns interpolated values as floats.

    Parameters
    ----------
    row_data : tuple
        A tuple containing the following elements:
        - r0 : int
            The row index being interpolated.
        - ifg : numpy.ndarray
            The 2D array representing the interferogram.
        - weights : numpy.ndarray
            The 2D array representing the weights for each pixel.
        - weight_cutoff : float
            The cutoff value for weights. Pixels with weights below this value
            will be interpolated.
        - num_neighbors : int
            The maximum number of neighboring pixels to consider for
            interpolation.
        - alpha : float
            The exponent used in the distance weighting function.
        - indices : list of tuple
            The list of relative indices for neighboring pixels.
        - ncol : int
            The number of columns in the interferogram.
        - nrow : int
            The number of rows in the interferogram.

    Returns
    -------
    r0 : int
        The row index that was interpolated.
    interpolated_row : numpy.ndarray
        The interpolated row as a 1D array of float32 values.
    """
    (r0, ifg, weights, weight_cutoff,
     num_neighbors, alpha, indices, ncol, nrow) = row_data

    interpolated_row = np.zeros(ncol, dtype=np.float32)
    for c0 in range(ncol):
        if weights[r0, c0] < weight_cutoff:
            csum = 0.0
            wsum = 0.0

            counter = 0
            r2 = np.zeros(num_neighbors, dtype=np.float64)
            phase = np.zeros(num_neighbors, dtype=np.float64)

            for idx in indices:
                r = r0 + idx[0]
                c = c0 + idx[1]

                if (0 <= r < nrow) and (0 <= c < ncol) and \
                   (weights[r, c] >= weight_cutoff):
                    # Calculate the square distance to the center pixel
                    r2[counter] = idx[0] ** 2 + idx[1] ** 2
                    phase[counter] = ifg[r, c]
                    counter += 1
                    if counter >= num_neighbors:
                        break

            if counter > 0:
                # Normalize and interpolate based on distance and phase
                r2_norm = (r2[counter - 1] ** alpha) / 2
                for i in range(counter):
                    csum += np.exp(-r2[i] / r2_norm) * phase[i]
                    wsum += np.exp(-r2[i] / r2_norm)

                interpolated_row[c0] = csum / wsum
            else:
                interpolated_row[c0] = ifg[r0, c0]
        else:
            interpolated_row[c0] = ifg[r0, c0]

    return r0, interpolated_row


def fill_with_distance(ifg,
                       weights,
                       weight_cutoff=0.5,
                       num_neighbors=20,
                       max_radius=51,
                       min_radius=0,
                       alpha=0.75):
    """Fill missing or low-weight values in an interferogram using
    distance-weighted interpolation.

    This function interpolates missing or low-weight values in an interferogram
    by using a distance-weighted interpolation method. It utilizes a circular
    neighborhood to find valid pixels and estimates the values for pixels with
    weights below a specified cutoff. The interpolation is performed in
    parallel using multiprocessing.

    Parameters
    ----------
    ifg : numpy.ndarray
        A 2D array representing the interferogram with potentially missing or
        low-weight values.
    weights : numpy.ndarray
        A 2D array representing the weights for each pixel in the
        interferogram.
    weight_cutoff : float, optional
        The cutoff value for weights. Pixels with weights below this value will
        be interpolated. Default is 0.5.
    num_neighbors : int, optional
        The maximum number of neighboring pixels to consider for interpolation.
        Default is 20.
    max_radius : int, optional
        The maximum radius for considering neighboring pixels in the circular
        neighborhood. Default is 51.
    min_radius : int, optional
        The minimum radius for starting to consider neighboring pixels.
        Default is 0.
    alpha : float, optional
        The exponent used in the distance weighting function. Default is 0.75.

    Returns
    -------
    interpolated_ifg : numpy.ndarray
        The interpolated interferogram, with missing or low-weight values
        filled using the distance-weighted interpolation method.
    """
    nrow, ncol = weights.shape
    indices = get_circle_idxs(max_radius, min_radius, sort_output=False)
    # interpolated_ifg = np.zeros((nrow, ncol), dtype=np.complex64)
    interpolated_ifg = np.zeros((nrow, ncol), dtype=np.float32)

    # Setup data for multiprocessing
    pool_data = [(r, ifg, weights, weight_cutoff,
                  num_neighbors, alpha, indices,
                  ncol, nrow) for r in range(nrow)]

    num_processes = os.cpu_count()  # Use all available CPU cores
    with Pool(num_processes) as pool:
        results = pool.map(interpolate_row, pool_data)

    # Combine results
    for r, interpolated_row in results:
        interpolated_ifg[r, :] = interpolated_row
    interpolated_ifg[interpolated_ifg == 0] = np.nan
    interpolated_ifg = fill_nearest(interpolated_ifg)

    return interpolated_ifg


def convolve_preserve_nan(image,
                          kernel,
                          mode: str = "constant",
                          cval: float = 0.0):
    """
    Convolve an image with a kernel while preserving NaN values.

    This function performs a convolution on an image, ensuring that NaN
    values in the original image are preserved in the output. The NaN values
    are temporarily replaced by zero during the convolution, and the result
    is normalized based on the coverage of valid (non-NaN) values.

    Parameters
    ----------
    image : numpy.ndarray
        A 2D array representing the input image, where some pixels may be NaN.
    kernel : numpy.ndarray
        A 2D array representing the convolution kernel.

    Returns
    -------
    convolved_image : numpy.ndarray
        A 2D array of the same shape as the input image, representing the
        result of the convolution with NaN values preserved.
    """
    # Step 1: Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(image)

    # Step 2: Replace NaN values with zero (or another placeholder)
    image_filled = np.where(valid_mask, image, 0)

    # Step 3: Apply convolution to the valid mask
    valid_mask_convolved = convolve(valid_mask.astype(float), kernel,
                                    mode=mode, cval=cval)
    # valid_mask_convolved = np.ones_like(image)
    # Step 4: Apply convolution to the filled image
    convolved_image = convolve(image_filled,
                               kernel,
                               mode=mode,
                               cval=0.0)

    # Step 5: Normalize the convolution result
    with np.errstate(invalid='ignore'):
        convolved_image = convolved_image / valid_mask_convolved

    # Step 6: Restore NaN values in the convolved output
    convolved_image[~valid_mask] = np.nan

    return convolved_image


def nan_median_filter(image, size):
    """
    Apply a median filter to an image, ignoring NaN values.

    This function uses a median filter to smooth an image while ignoring NaN
    values. The filter computes the median of the non-NaN values within a
    given neighborhood around each pixel.

    Parameters
    ----------
    image : numpy.ndarray
        A 2D array representing the input image, where some pixels may be NaN.
    size : int or sequence of int
        The size of the neighborhood used for the filter. It can be a single
        integer or a sequence of integers defining the filter size for each
        dimension.

    Returns
    -------
    filtered_image : numpy.ndarray
        A 2D array of the same shape as the input image, representing the
        filtered image with NaN values preserved.
    """
    # Create an output array to hold the filtered image
    filtered_image = np.empty_like(image)

    # Use the ndimage.generic_filter function to apply a custom function
    # to the image
    def nan_median(values):
        return np.nanmedian(values)  # Ignore NaNs when computing the median

    filtered_image = generic_filter(image,
                                    nan_median,
                                    size=size,
                                    mode='constant',
                                    cval=np.nan)

    return filtered_image


def unwrapping_correction_with_filter(unw,
                                      kernel_width,
                                      kernel_length,
                                      sig_kernel_x,
                                      sig_kernel_y,
                                      iterations,
                                      filter_method):
    """
    Correct phase unwrapping errors in an image using iterative filtering.

    This function iteratively applies either a Gaussian convolution or a
    median filter to correct phase unwrapping errors in an image. It adjusts
    values modulo 2π during each iteration to correct phase errors.

    Parameters
    ----------
    unw : numpy.ndarray
        A 2D array representing the unwrapped phase image.
    kernel_width : int
        The width of the Gaussian kernel.
    kernel_length : int
        The length of the Gaussian kernel.
    sig_kernel_x : float
        The standard deviation of the Gaussian kernel in the x direction.
    sig_kernel_y : float
        The standard deviation of the Gaussian kernel in the y direction.
    iterations : int
        The number of iterations for filtering and correction.
    filter_method : str
        The method of filtering to use,
        either 'convolution' or 'median_filter'.

    Returns
    -------
    unw : numpy.ndarray
        The corrected unwrapped phase image.
    """
    # Create a mask for zero values in the input
    mask = unw == 0
    unw[mask] = np.nan

    # Create and normalize the Gaussian kernel
    kernel_rows = create_gaussian_kernel(
        kernel_length, sig_kernel_y).reshape(-1, 1)
    kernel_cols = create_gaussian_kernel(
        kernel_width, sig_kernel_x).reshape(1, -1)
    kernel_norm = (kernel_rows * kernel_cols) / np.sum(
        kernel_rows * kernel_cols)

    # Iteratively apply the selected filter and correct phase unwrapping errors
    for _ in range(iterations):
        if filter_method == 'convolution':
            filtered_img = convolve_preserve_nan(unw, kernel_norm)
        elif filter_method == 'median_filter':
            filtered_img = nan_median_filter(unw, size=kernel_width)
        else:
            raise ValueError("Invalid filter_method. Choose 'convolution' or 'median_filter'.")

        # Restore NaN values in the filtered image and correct phase errors
        filtered_img[mask] = np.nan
        diff = unw - filtered_img
        unw_err_ind = np.round(diff / (2 * np.pi))
        unw -= unw_err_ind * 2 * np.pi

    # Restore original zero values in the output image
    unw[mask] = 0
    return unw


def remove_small_components(image, min_cluster_pixels):
    """
    Remove small clusters of valid (non-NaN) pixels from an image.

    This function identifies and labels connected regions of valid pixels in
    the input image. It then removes clusters that are smaller than the
    specified minimum size by setting their corresponding pixels to NaN in
    the output image.

    Parameters
    ----------
    image : numpy.ndarray
        A 2D array representing the input image, where some pixels may be NaN.
    min_cluster_pixels : int
        The minimum size (number of pixels) for a cluster to be retained.
        Clusters smaller than this size will be removed.

    Returns
    -------
    cleaned_image : numpy.ndarray
        A 2D array of the same shape as the input image, where small clusters
        have been removed and replaced with NaN.
    """
    warning_channel = journal.warning("ionosphere_filter.remove_small_components")
    info_channel = journal.info("ionosphere_filter.remove_small_components")

    if not isinstance(min_cluster_pixels, int):
        raise TypeError("min_cluster_pixels must be an int")
    if min_cluster_pixels < 0:
        raise ValueError("min_cluster_pixels cannot be negative")

    valid_mask = ~np.isnan(image)
    n_valid = int(valid_mask.sum())

    if min_cluster_pixels == 0:
        info_channel.log("min_cluster_pixels == 0 → "
                         "no components will be removed.")
        return image.copy()

    if min_cluster_pixels >= n_valid:
        warning_channel.log(
            f"min_cluster_pixels ({min_cluster_pixels}) are larger than "
            f"the total number of valid pixels ({n_valid})."
        )
    # 4-connected by default
    labeled_image, _ = label(valid_mask)

    object_slices = find_objects(labeled_image)

    cleaned_image = image.copy()

    for i, slice_tuple in enumerate(object_slices):
        current_object = (labeled_image[slice_tuple] == (i + 1))
        if current_object.sum() < min_cluster_pixels:
            cleaned_image[slice_tuple][current_object] = np.nan

    return cleaned_image
