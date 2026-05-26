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
                           gaussian_filter,
                           generic_filter,
                           maximum_filter,
                           uniform_filter,
                           )

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
                 guide_filter_method="median_gaussian",
                 guide_median_size=3,
                 outlier_threshold=3.5,
                 outlier_min_scale=0.0,
                 mad_scale_factor=1.4826,
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
        guide_filter_method : {'median_gaussian', 'gaussian', 'none'}, optional
            Method used to build the guide image before filling.
        guide_median_size : int, optional
            Median filter size used when guide_filter_method is
            'median_gaussian'.
        outlier_threshold : float, optional
            Threshold multiplier for robust outlier detection.
        outlier_min_scale : float, optional
            Minimum scale for robust outlier detection.
        mad_scale_factor : float, optional
            Minimum scale for robust outlier detection.
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
        self.guide_filter_method = guide_filter_method
        self.guide_median_size = guide_median_size
        self.outlier_threshold = outlier_threshold
        self.outlier_min_scale = outlier_min_scale
        self.mad_scale_factor = mad_scale_factor

    def low_pass_filter(
            self,
            input_data,
            input_std_dev,
            mask_path,
            filtered_output,
            filtered_std_dev,
            lines_per_block,
            min_cluster_pixels,
            ):
        """Apply low-pass filtering to dispersive and nondispersive phase
        with standard deviation.

        This version keeps the overall flow of the old function, but adds:
        1. guide-image smoothing before gap filling,
        2. robust outlier rejection,
        3. restoration of original valid pixels before guide construction,
        4. restoration of original valid pixels again after filling,
        5. cropped block writing for padded reads.

        Parameters
        ----------
        input_data : str or h5py.Dataset
            File path for data to be filtered.
        input_std_dev : str or h5py.Dataset
            File path for standard deviation.
        mask_path : str or h5py.Dataset
            File path for mask raster.
            1: valid pixels, 0: invalid pixels.
        filtered_output : str or h5py.Dataset
            Output file path or dataset for filtered data.
        filtered_std_dev : str or h5py.Dataset
            Output file path or dataset for filtered standard deviation.
        lines_per_block : int
            Number of lines to process in each block.
        min_cluster_pixels : int
            Minimum connected valid-pixel cluster size to keep.

        Returns
        -------
        None
        """
        data_shape, _ = get_raster_info(input_data)
        data_length, data_width = data_shape

        lines_per_block = min(data_length, lines_per_block)

        # Determine padding
        pad_length = 2 * (self.y_kernel // 2)
        pad_width = 2 * (self.x_kernel // 2)
        pad_shape = (pad_length, pad_width)

        # Prepare outputs
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

        guide_sigma = max(self.sig_x, self.sig_y, 1)
        median_size = max(3, self.guide_median_size)
        if median_size % 2 == 0:
            median_size += 1
        eps = 1e-6

        for iter_cnt in range(self.iteration):

            block_params = block_param_generator(
                lines_per_block, data_shape, pad_shape)

            for block_param in block_params:
                width_offset = pad_width // 2
                length_offset = pad_length // 2

                row_start = length_offset
                row_end = length_offset + block_param.block_length
                col_start = width_offset
                col_end = width_offset + block_param.data_width

                # Previous iteration temp paths
                filtered_iono_temp_input_path = \
                    f'{self.outputdir}/filtered_iono_temp{iter_cnt-1}'
                filtered_std_temp_input_path = \
                    f'{self.outputdir}/filtered_iono_std_temp{iter_cnt-1}'

                # Read current working image:
                #   iter 0 -> original input
                #   iter >0 -> previous filtered result
                block_data_path = filtered_iono_temp_input_path \
                    if iter_cnt > 0 else input_data
                block_sig_path = filtered_std_temp_input_path \
                    if iter_cnt > 0 else input_std_dev

                data_block = read_block_array(block_data_path, block_param)
                data_sig_block = read_block_array(block_sig_path, block_param)
                mask_block = read_block_array(mask_path, block_param, fill_value=0)

                # Always read original data to preserve valid pixels
                orig_data_block = read_block_array(input_data, block_param)
                orig_sig_block = read_block_array(input_std_dev, block_param)

                data_block = data_block.astype(np.float32, copy=False)
                data_sig_block = data_sig_block.astype(np.float32, copy=False)
                orig_data_block = orig_data_block.astype(np.float32, copy=False)
                orig_sig_block = orig_sig_block.astype(np.float32, copy=False)

                invalid_mask = (mask_block == 0)

                # Mask current working arrays
                data_block[invalid_mask] = np.nan
                data_sig_block[invalid_mask] = np.nan
                data_sig_block[data_sig_block <= 0] = np.nan

                # Build original valid support from original image
                orig_data_block[invalid_mask] = np.nan
                orig_sig_block[invalid_mask] = np.nan
                orig_sig_block[orig_sig_block <= 0] = np.nan

                orig_data_block = remove_small_components(
                    orig_data_block,
                    min_cluster_pixels=min_cluster_pixels)

                orig_valid_mask = np.isfinite(orig_data_block)
                orig_sig_block[~orig_valid_mask] = np.nan

                # Critical fix:
                # restore original valid pixels BEFORE guide generation,
                # so valid support does not drift with iteration count.
                data_block[orig_valid_mask] = orig_data_block[orig_valid_mask]
                data_sig_block[orig_valid_mask] = orig_sig_block[orig_valid_mask]

                # Guide image for filling
                guide_data = data_block.copy()

                if self.guide_filter_method == "median_gaussian":
                    guide_data = nan_median_filter(
                        guide_data, size=median_size)
                    guide_data = nan_aware_gaussian(
                        guide_data, sigma=guide_sigma)
                elif self.guide_filter_method == "gaussian":
                    guide_data = nan_aware_gaussian(
                        guide_data, sigma=guide_sigma)
                elif self.guide_filter_method == "none":
                    pass
                else:
                    raise ValueError(
                        "self.guide_filter_method must be one of "
                        "{'median_gaussian', 'gaussian', 'none'}"
                    )

                # Robust outlier rejection
                outlier_enabled = (
                    self.outlier_threshold is not None and
                    self.outlier_threshold > 0 and
                    self.guide_filter_method != "none"
                )

                if outlier_enabled:
                    residual = data_block - guide_data
                    finite_res = np.isfinite(residual) & orig_valid_mask

                    if np.any(finite_res):
                        residual_valid = residual[finite_res]
                        med_res = np.nanmedian(residual_valid)
                        mad_res = np.nanmedian(np.abs(residual_valid - med_res))
                        robust_scale = max(self.mad_scale_factor * mad_res,
                                           self.outlier_min_scale)

                        if np.isfinite(robust_scale) and robust_scale > 0:
                            outlier_mask = np.zeros_like(data_block, dtype=bool)
                            outlier_mask[finite_res] = (
                                np.abs(residual_valid - med_res) >
                                self.outlier_threshold * robust_scale
                            )

                            # Treat strong outliers as fill targets
                            data_block[outlier_mask] = np.nan
                            data_sig_block[outlier_mask] = np.nan

                            # Rebuild guide after removing outliers
                            guide_data = data_block.copy()
                            if self.guide_filter_method == "median_gaussian":
                                guide_data = nan_median_filter(
                                    guide_data, size=median_size)
                                guide_data = nan_aware_gaussian(
                                    guide_data, sigma=guide_sigma)
                            elif self.guide_filter_method == "gaussian":
                                guide_data = nan_aware_gaussian(
                                    guide_data, sigma=guide_sigma)

                # Select fill method
                if self.filling_method == "smoothed":
                    fill_method = fill_with_smoothed
                elif self.filling_method == "nearest":
                    fill_method = fill_nearest
                elif self.filling_method == "distance":
                    fill_method = fill_with_distance
                else:
                    raise ValueError(
                        f"Unsupported filling_method: {self.filling_method}"
                    )

                # Fill gaps
                if self.filling_method == "distance":
                    sigma_for_weight = np.where(
                        np.isfinite(data_sig_block), data_sig_block, np.nan)

                    weight = np.zeros_like(mask_block, dtype=np.float32)
                    good_w = np.isfinite(sigma_for_weight) & np.isfinite(data_block)
                    weight[good_w] = 1.0 / (sigma_for_weight[good_w]**2 + eps)

                    # Fallback if all weights are zero
                    if not np.any(weight > 0):
                        weight = np.isfinite(data_block).astype(np.float32)

                    filled_data = fill_method(guide_data, weight)

                    sigma_fill_source = data_sig_block.copy()
                    sigma_fill_source[~np.isfinite(sigma_fill_source)] = np.nan
                    filled_data_sig = fill_method(sigma_fill_source, weight)
                else:
                    filled_data = fill_method(guide_data)

                    sigma_fill_source = data_sig_block.copy()
                    sigma_fill_source[~np.isfinite(sigma_fill_source)] = np.nan
                    filled_data_sig = fill_method(sigma_fill_source)

                # Restore original valid pixels again after filling
                filled_data[orig_valid_mask] = orig_data_block[orig_valid_mask]
                filled_data_sig[orig_valid_mask] = orig_sig_block[orig_valid_mask]

                # Sigma fallback
                bad_sig = ~np.isfinite(filled_data_sig) | (filled_data_sig <= 0)
                good_sig = (~bad_sig) & np.isfinite(filled_data_sig)

                if np.any(good_sig):
                    replacement_sig = np.nanmedian(filled_data_sig[good_sig])
                    if not np.isfinite(replacement_sig) or replacement_sig <= 0:
                        replacement_sig = 1.0
                    filled_data_sig[bad_sig] = replacement_sig
                else:
                    filled_data_sig[:] = 1.0

                # Final filter for this iteration
                filt_data, filt_data_sig = filter_data_with_sig(
                    input_array=filled_data,
                    sig_array=filled_data_sig,
                    kernel_width=self.x_kernel,
                    kernel_length=self.y_kernel,
                    sig_kernel_x=self.sig_x,
                    sig_kernel_y=self.sig_y)

                # Last iteration -> final output, otherwise temp output
                if iter_cnt == self.iteration - 1:
                    output_iono = filtered_output
                    output_std = filtered_std_dev
                else:
                    output_iono = f'{self.outputdir}/filtered_iono_temp{iter_cnt}'
                    output_std = f'{self.outputdir}/filtered_iono_std_temp{iter_cnt}'

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


def nan_aware_gaussian(image, sigma=1.5):
    """
    Apply Gaussian smoothing to an image while properly handling NaN values.

    This function performs a normalized Gaussian convolution that ignores
    NaN pixels. It replaces NaNs with zero during convolution and then
    normalizes the result by the effective support (i.e., number of valid
    contributing pixels), ensuring that missing data does not bias the
    smoothed output.

    Parameters
    ----------
    image : numpy.ndarray
        2D input array to be smoothed. NaN values indicate invalid or
        missing pixels and are excluded from the smoothing operation.
    sigma : float or sequence of float, optional
        Standard deviation of the Gaussian kernel. Controls the amount
        of smoothing.

        - Larger sigma → stronger smoothing (larger effective kernel)
        - Smaller sigma → weaker smoothing (preserves more detail)
        The effective kernel size is approximately:
            kernel_size ≈ 2 * ceil(3 * sigma) + 1
        If a sequence is provided, anisotropic smoothing is applied
        (e.g., sigma=(sigma_y, sigma_x)).

    Returns
    -------
    out : numpy.ndarray
        Smoothed array of the same shape as `image`.
    """
    # valid is 1 where the input is finite, 0 where it is NaN
    valid = np.isfinite(image).astype(float)

    # image0 is the input image with NaNs replaced by 0
    image0 = np.nan_to_num(image, nan=0.0)

    # smooth_num is the Gaussian-smoothed sum of nearby pixel values,
    # but missing pixels contribute 0
    smooth_num = gaussian_filter(image0, sigma=sigma)
    # smooth_den is the Gaussian-smoothed sum of nearby validity weights,
    # effectively telling you how much real data contributed
    smooth_den = gaussian_filter(valid, sigma=sigma)

    out = np.full_like(image, np.nan, dtype=float)
    good = smooth_den > 1e-6

    # renormalizes the result so NaN gaps do not
    # artificially pull the average down.
    out[good] = smooth_num[good] / smooth_den[good]
    return out


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
    # Step 5: Safe normalization
    with np.errstate(divide='ignore', invalid='ignore'):
        convolved_image = np.divide(
            convolved_image,
            valid_mask_convolved,
            out=np.zeros_like(convolved_image),
            where=valid_mask_convolved > 1e-8
        )

    # Step 6: Mark regions with no valid support as NaN
    no_support = valid_mask_convolved <= 1e-8
    convolved_image[no_support] = np.nan

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



def build_mask(
    phase_image,
    *,
    coherence_image=None,
    conncomp_image=None,
    subswath_mask_image=None,
    coh_threshold=None,
    gradient_threshold=None,
    gradient_window_size=25,
    gradient_method='mean',
    gradient_mask_percentile=None,
    water_mask_image=None,
    mask_type=None,
    fill_value=0.0,
):
    """
    Apply a mask to one native-grid phase image.
    """
    if phase_image is None:
        return None

    if mask_type is None:
        mask_type = []

    mask = np.isfinite(phase_image) & (phase_image != 0)

    if "coherence" in mask_type and coherence_image is not None:
        mask &= np.isfinite(coherence_image)
        mask &= (coherence_image >= coh_threshold)

    if "connected_components" in mask_type and conncomp_image is not None:
        mask &= np.isfinite(conncomp_image)
        mask &= (conncomp_image > 0)

    if "subswath_mask" in mask_type and subswath_mask_image is not None:
        mask &= np.isfinite(subswath_mask_image)
        mask &= (subswath_mask_image > 0)

    if "water" in mask_type and water_mask_image is not None:
        mask &= water_mask_image

    if "gradient" in mask_type:
        _, _, grad_mask = detect_high_phase_gradient_local(
            phase_image,
            gradient_threshold,
            window_size=gradient_window_size,
            mask=mask,
            method=gradient_method,
            percentile=gradient_mask_percentile)
        mask &= grad_mask

    out = phase_image.copy()
    out[~mask] = fill_value
    return out, mask


def detect_high_phase_gradient_local(
    unwrapped_phase,
    threshold,
    window_size=5,
    mask=None,
    method="mean",
    percentile=90,
    return_gradient_components=False,
):
    """
    Detect high-gradient areas from an unwrapped phase image using
    a local windowed gradient score.
    Parameters
    ----------
    unwrapped_phase : np.ndarray
        2D unwrapped phase array in radians.
    threshold : float
        Threshold applied to the local gradient score.
        Unit: radians/pixel.
    window_size : int, optional
        Size of the moving window used to summarize gradient magnitude.
        Must be >= 1. Typical values: 3, 5, 7.
    mask : np.ndarray or None, optional
        Boolean valid-data mask with same shape as input.
        True means valid pixel. If None, finite values are treated as valid.
    method : str, optional
        Local summary method:
        - "mean": local mean of gradient magnitude
        - "max": local max of gradient magnitude
        - "percentile": local percentile of gradient magnitude
    percentile : float, optional
        Percentile used when method="percentile".
    return_gradient_components : bool, optional
        If True, also return gx and gy.
    Returns
    -------
    grad_mag : np.ndarray
        Pixelwise gradient magnitude in radians/pixel.
    local_grad_score : np.ndarray
        Local windowed summary of grad_mag in radians/pixel.
    high_gradient_mask : np.ndarray
        Boolean mask where local_grad_score > threshold.
    gx, gy : np.ndarray, optional
        Gradient components if return_gradient_components=True.
    Notes
    -----
    For unwrapped phase, gradients are computed using central differences
    for interior pixels and first differences at the boundaries.
    """
    phase = np.asarray(unwrapped_phase, dtype=np.float64)

    if phase.ndim != 2:
        raise ValueError("unwrapped_phase must be a 2D array")

    if window_size < 1 or int(window_size) != window_size:
        raise ValueError("window_size must be a positive integer")

    if method not in {"mean", "max", "percentile"}:
        raise ValueError("method must be 'mean', 'max', or 'percentile'")

    if mask is None:
        valid = np.isfinite(phase)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(phase)

    # Require valid neighbors for safe gradient estimation
    # We will compute gradients first on a filled array, then invalidate
    # locations influenced by invalid pixels.
    phase_filled = phase.copy()
    phase_filled[~valid] = 0.0

    # Gradient in y (rows) and x (cols)
    gy = np.full_like(phase, np.nan, dtype=np.float64)
    gx = np.full_like(phase, np.nan, dtype=np.float64)

    # Central differences for interior
    gy[1:-1, :] = (phase_filled[2:, :] - phase_filled[:-2, :]) / 2.0
    gx[:, 1:-1] = (phase_filled[:, 2:] - phase_filled[:, :-2]) / 2.0

    # Forward/backward difference at borders
    gy[0, :] = phase_filled[1, :] - phase_filled[0, :]
    gy[-1, :] = phase_filled[-1, :] - phase_filled[-2, :]

    gx[:, 0] = phase_filled[:, 1] - phase_filled[:, 0]
    gx[:, -1] = phase_filled[:, -1] - phase_filled[:, -2]

    # Build validity masks for gradients
    valid_gx = np.zeros_like(valid, dtype=bool)
    valid_gy = np.zeros_like(valid, dtype=bool)

    # gx validity
    valid_gx[:, 1:-1] = valid[:, 2:] & valid[:, :-2] & valid[:, 1:-1]
    valid_gx[:, 0] = valid[:, 0] & valid[:, 1]
    valid_gx[:, -1] = valid[:, -1] & valid[:, -2]

    # gy validity
    valid_gy[1:-1, :] = valid[2:, :] & valid[:-2, :] & valid[1:-1, :]
    valid_gy[0, :] = valid[0, :] & valid[1, :]
    valid_gy[-1, :] = valid[-1, :] & valid[-2, :]

    gx[~valid_gx] = np.nan
    gy[~valid_gy] = np.nan

    # Gradient magnitude
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Valid where both phase and grad_mag are valid
    valid_grad = np.isfinite(grad_mag) & valid

    # Local summary of gradient magnitude
    if method == "mean":
        data = np.where(valid_grad, grad_mag, 0.0)
        weights = valid_grad.astype(np.float64)

        local_sum = uniform_filter(data, size=window_size,
                                   mode="constant", cval=0.0) * (window_size ** 2)
        local_count = uniform_filter(weights, size=window_size, mode="constant", cval=0.0) * (window_size ** 2)

        with np.errstate(invalid="ignore", divide="ignore"):
            local_grad_score = local_sum / local_count

        local_grad_score[local_count == 0] = np.nan

    elif method == "max":
        data = np.where(valid_grad, grad_mag, -np.inf)
        local_grad_score = maximum_filter(data, size=window_size, mode="constant", cval=-np.inf)
        local_grad_score[~valid] = np.nan
        local_grad_score[~np.isfinite(local_grad_score)] = np.nan

    elif method == "percentile":
        def nanpercentile_func(values):
            vals = values[np.isfinite(values)]
            if vals.size == 0:
                return np.nan
            return np.percentile(vals, percentile)

        data = np.where(valid_grad, grad_mag, np.nan)
        local_grad_score = generic_filter(
            data,
            function=nanpercentile_func,
            size=window_size,
            mode="constant",
            cval=np.nan,
        )

    # Final mask
    high_gradient_mask = np.isfinite(local_grad_score) & (local_grad_score < threshold)
    high_gradient_mask[~valid] = False

    if return_gradient_components:
        return grad_mag, local_grad_score, high_gradient_mask, gx, gy

    return grad_mag, local_grad_score, high_gradient_mask
