'''
Wrapper for rubbersheet
'''
from __future__ import annotations

import pathlib
import time

import journal
import numpy as np
from isce3.io import HDF5OptimizedReader
from nisar.products.insar.product_paths import RIFGGroupsPaths
from nisar.products.readers import SLC
from nisar.workflows import prepare_insar_hdf5
from nisar.workflows.helpers import (get_cfg_freq_pols,
                                     get_ground_track_velocity_product,
                                     sum_gdal_rasters)
from nisar.workflows.rubbersheet_runconfig import RubbersheetRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse
from osgeo import gdal
from scipy import interpolate, ndimage, signal


def run(cfg: dict, output_hdf5: str = None):
    '''
    Run rubbersheet
    '''

    # Pull parameters from cfg dictionary
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    rubbersheet_params = cfg['processing']['rubbersheet']
    geo2rdr_offsets_path = pathlib.Path(rubbersheet_params['geo2rdr_offsets_path'])
    off_product_enabled = cfg['processing']['offsets_product']['enabled']
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']

    # If not set, set output HDF5 file
    if output_hdf5 is None:
        output_hdf5 = cfg['product_path_group']['sas_output_file']

    info_channel = journal.info('rubbersheet.run')
    info_channel.log('Start rubbersheet')
    t_all = time.time()

    # Initialize parameters share by frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)
    ref_radar_grid = ref_slc.getRadarGrid()

    # Get the slant range and zero doppler time spacing
    ref_slant_range_spacing = ref_radar_grid.range_pixel_spacing
    ref_zero_doppler_time_spacing = ref_radar_grid.az_time_interval

    # Pull the slant range and zero doppler time of the pixel offsets product
    # at frequencyA
    with HDF5OptimizedReader(name=output_hdf5, mode='r+', libver='latest', swmr=True) as dst_h5:

        for freq, _, pol_list in get_cfg_freq_pols(cfg):
            freq_group_path = f'{RIFGGroupsPaths().SwathsPath}/frequency{freq}'
            pixel_offsets_path = f'{freq_group_path}/pixelOffsets'
            geo_offset_dir = geo2rdr_offsets_path / 'geo2rdr' / f'freq{freq}'
            rubbersheet_dir = scratch_path / 'rubbersheet_offsets' / f'freq{freq}'
            slant_range = dst_h5[f'{pixel_offsets_path}/slantRange'][()]
            zero_doppler_time = dst_h5[f'{pixel_offsets_path}/zeroDopplerTime'][()]


            # Produce ground track velocity for the frequency under processing
            ground_track_velocity_file = get_ground_track_velocity_product(ref_slc,
                                                                           slant_range,
                                                                           zero_doppler_time,
                                                                           dem_file,
                                                                           rubbersheet_dir)
            for pol in pol_list:
                # Create input and output directories for pol under processing
                pol_group_path = f'{pixel_offsets_path}/{pol}'
                off_prod_dir = scratch_path / 'offsets_product' / f'freq{freq}' / pol
                out_dir = rubbersheet_dir / pol
                out_dir.mkdir(parents=True, exist_ok=True)

                if not off_product_enabled:
                    # Dense offset is enabled and offset products are disabled
                    dense_offsets_path = pathlib.Path(rubbersheet_params['dense_offsets_path'])
                    dense_offsets_dir = dense_offsets_path / 'dense_offsets' / f'freq{freq}' / pol
                    # Identify outliers
                    offset_az_culled, offset_rg_culled = identify_outliers(
                        str(dense_offsets_dir),
                        rubbersheet_params)
                    # Fill outliers holes
                    offset_az = fill_outliers_holes(offset_az_culled,
                                                    rubbersheet_params)
                    offset_rg = fill_outliers_holes(offset_rg_culled,
                                                    rubbersheet_params)
                    # Get correlation peak path
                    corr_peak_path = str(f'{dense_offsets_dir}/correlation_peak')
                else:
                    # Offset product is enabled, perform blending (pyramidal filling)
                    off_product_path = pathlib.Path(
                        rubbersheet_params['offsets_product_path'])
                    off_product_dir = off_product_path / 'offsets_product' / f'freq{freq}' / pol

                    # Get layer keys
                    layer_keys = [key for key in
                                  cfg['processing']['offsets_product'].keys() if
                                  key.startswith('layer')]
                    # Apply offset blending
                    offset_az, offset_rg = _offset_blending(off_product_dir,
                                                            rubbersheet_params, layer_keys)

                    # Get correlation peak path for the first offset layer
                    corr_peak_path = str(f'{off_prod_dir}/{layer_keys[0]}/correlation_peak')

                # Form a list with azimuth and slant range offset
                offsets = [offset_az, offset_rg]
                for k, offset in enumerate(offsets):
                    # If there are residual NaNs, use interpolation to fill residual holes
                    nan_count = np.count_nonzero(np.isnan(offset))
                    if nan_count > 0:
                        offsets[k] = _interpolate_offsets(offset,
                                                          rubbersheet_params['interpolation_method'])
                    # If required, filter offsets
                    offsets[k] = _filter_offsets(offsets[k], rubbersheet_params)
                    # Save offsets on disk for resampling
                    off_type = 'culled_az_offsets' if k == 0 else 'culled_rg_offsets'
                    _write_to_disk(str(f'{out_dir}/{off_type}'), offsets[k])

                # Get ground velocity and correlation peak
                ground_track_velocity = _open_raster(ground_track_velocity_file)
                corr_peak = _open_raster(corr_peak_path)

                # Get datasets from HDF5 file and update datasets
                offset_az_prod = dst_h5[f'{pol_group_path}/alongTrackOffset']
                offset_rg_prod = dst_h5[f'{pol_group_path}/slantRangeOffset']
                offset_peak_prod = dst_h5[f'{pol_group_path}/correlationSurfacePeak']

                # Assign cross-correlation peak
                offset_peak_prod[...] = corr_peak
                # Convert the along track and slant range pixel offsets to meters
                offset_az_prod[...] = \
                    offsets[0] * ground_track_velocity \
                    * ref_zero_doppler_time_spacing
                offset_rg_prod[...] = offsets[1] * ref_slant_range_spacing

                rubber_offs = ['culled_az_offsets', 'culled_rg_offsets']
                geo_offs = ['azimuth.off', 'range.off']
                for rubber_off, geo_off in zip(rubber_offs, geo_offs):
                    # Resample offsets to the size of the reference RSLC
                    culled_off_path = str(out_dir / rubber_off)
                    resamp_off_path = culled_off_path.replace('culled', 'resampled')
                    ds = gdal.Open(culled_off_path, gdal.GA_ReadOnly)
                    gdal.Translate(resamp_off_path, ds,
                                   width=ref_radar_grid.width,
                                   height=ref_radar_grid.length, format='ENVI')
                    # Sum resampled offsets to geometry offsets
                    sum_off_path = str(out_dir / geo_off)
                    sum_gdal_rasters(str(geo_offset_dir / geo_off),
                                     resamp_off_path, sum_off_path,
                                     invalid_value=-1e6)

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"Successfully ran rubbersheet in {t_all_elapsed:.3f} seconds")


def _open_raster(filepath, band=1):
    '''
    Open GDAL raster

    Parameters
    ----------
    filepath: str
        File path to the raster to open
    band: int
        Band number to extract from GDAL raster. Defaults to 1.

    Returns
    -------
    data: np.ndarray
        Array containing the "band" extracted from
        raster in "filepath"
    '''
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(band).ReadAsArray()
    return data


def _write_to_disk(outpath, array, format='ENVI',
                   datatype=gdal.GDT_Float64):
    '''
    Write numpy array to disk as a GDAl raster

    Parameters
    ----------
    outpath: str
        Path to save array on disk
    array: numpy.ndarray
        Numpy array to save locally
    format: str
        GDAL-friendly format for output raster
    datatype: str
        GDAL data type for output raster
    '''

    length, width = array.shape
    driver = gdal.GetDriverByName(format)
    ds = driver.Create(outpath, width, length, 1, datatype)
    ds.GetRasterBand(1).WriteArray(array)
    ds.FlushCache()

def identify_outliers(offsets_dir, rubbersheet_params):
    '''
    Identify outliers in the offset fields.
    Outliers are identified by a thresholding
    metric (SNR, offset covariance, offset median
    absolute deviation) suggested by the user

    Parameters
    ----------
    offsets_dir: str
        Path to the dense offset or offsets products directory
        where pixel offsets are located
    rubbersheet_params: cfg
        Dictionary containing rubbersheet parameters

    Returns
    -------
    offset_az: array, float
        2D array of culled/outlier filled azimuth offset
    offset_rg: array, float
        2D array of culled/outlier filled range offset
    '''
    # Extract parameters
    threshold = rubbersheet_params['threshold']
    window_rg, window_az = rubbersheet_params['median_filter_size_range'], rubbersheet_params['median_filter_size_azimuth']
    metric = rubbersheet_params['culling_metric']
    error_channel = journal.error('rubbersheet.run.identify_outliers')
    
    # Load data based on metric
    if metric == 'snr':
        mask_data = _open_raster(f'{offsets_dir}/snr', 1) < threshold
    elif metric == 'median_filter':
        offset_az = _open_raster(f'{offsets_dir}/dense_offsets', 1)
        offset_rg = _open_raster(f'{offsets_dir}/dense_offsets', 2)
        mask_data = compute_mad_mask(offset_az, window_az, window_rg, threshold) | \
                    compute_mad_mask(offset_rg, window_az, window_rg, threshold)
    elif metric == 'covariance':
        cov_az, cov_rg = _open_raster(f'{offsets_dir}/covariance', 1), _open_raster(f'{offsets_dir}/covariance', 2)
        mask_data = (cov_az > threshold) | (cov_rg > threshold)
    else:
        error_channel.log(f"{metric} is an invalid metric to filter outliers")
        raise ValueError(f"Invalid metric: {metric}")
    
    # Apply mask
    offset_az[mask_data], offset_rg[mask_data] = np.nan, np.nan
    
    # Optional refinement
    if rubbersheet_params.get('mask_refine_enabled', False):
        filter_size = rubbersheet_params['mask_refine_filter_size']
        max_nan_neighbors = rubbersheet_params['mask_refine_min_neighbors']
        mask_final = compute_mad_mask(offset_az, filter_size, filter_size, threshold) | \
                     compute_mad_mask(offset_rg, filter_size, filter_size, threshold)
        offset_az[mask_final], offset_rg[mask_final] = np.nan, np.nan
        offset_az = remove_pixels_with_many_nans(offset_az, filter_size, max_nan_neighbors)
        offset_rg = remove_pixels_with_many_nans(offset_rg, filter_size, max_nan_neighbors)
    
    return offset_az, offset_rg


def remove_pixels_with_many_nans(offset, kernel_size=3, max_nan_neighbors=4):
    """
    Replace pixels with NaN if they have too many NaN values in their neighborhood.

    Parameters
    ----------
    offset: np.ndarray
        2D array of offset values, where NaN represents missing data.
    kernel_size: int
        Size of the square neighborhood kernel (default is 3).
    max_nan_neighbors: int
        Maximum number of NaN neighbors allowed for a pixel to remain unchanged (default is 4).

    Returns
    -------
    offset: np.ndarray
        The input array with pixels removed if they have too many NaN neighbors.
    """
    mask = np.isnan(offset)  # Binary mask of NaN pixels (1 for NaN, 0 for valid)

    # Create a kernel of ones, excluding the center pixel
    kernel = np.ones((kernel_size, kernel_size), dtype=int)
    center = kernel_size // 2
    kernel[center, center] = 0  # Exclude center pixel from count

    # Count the number of NaN neighbors
    nan_neighbor_count = ndimage.convolve(mask.astype(int), kernel, mode='constant', cval=0)

    # Identify pixels with too many NaN neighbors
    remove_pixels = nan_neighbor_count > max_nan_neighbors

    # Remove these pixels by setting them to NaN
    offset[remove_pixels] = np.nan

    return offset


def compute_mad_mask(offset, window_az, window_rg, threshold):
    '''
    Compute a mask of outliers in a 2D offset array using the Median
    Absolute Deviation (MAD). This function identifies pixels whose
    absolute deviation from the local median exceeds a given threshold.
    The median is computed over a sliding window of specified size

    Parameters
    ----------
    offset: np.ndarray
        2D array of offset values from image matching to 
        be analyzed for outliers
    window_az: int
        Size of the filtering window along the row direction
    window_rg: int
        Size of the filtering window along the column direction
    threshold: float
        Threshold for outliers identification. Offsets with MAD
        greater than this threshold are marked as outliers

    Returns
    -------
    outliers_mask: np.ndarray of booleans
        Boolean mask of the same shape as 'offset', where 'True'
        indicates a detected offset outlier
    '''
    # Mask the NaN values in the input array
    masked_offset = np.ma.masked_invalid(offset)
    
    # Apply median filter, ignoring NaNs
    median_off = ndimage.median_filter(masked_offset, [window_az, window_rg])
    
    # Compute the absolute deviation from the median
    mad = np.abs(offset - median_off) 
    
    # Create a mask for pixels where MAD exceeds the threshold
    outliers_mask = mad > threshold
    
    return outliers_mask  


def fill_outliers_holes(offset, rubbersheet_params):
    '''
    Fill no data values according to user-preference.
    No data values are filled using one of the following:
       - fill_smoothed: replace no data values with smoothed value
         in a neighborhood.
       - nearest_neighbor: replace no data with nearest neighbor
         interpolation
       - hybrid: Use one iteration of fill smoothed followed by
         linear interpolation

    Parameters
    ----------
    offset: np.ndarray, float
        2D array with no data values (NaNs) to be filled
    rubbersheet_params: dict
        Dictionary containing rubbersheet parameters from runconfig

    Returns
    -------
    offset_filled: np.ndarray, float
        2D array with no data values filled with data values
        from one of the algorithms (fill_smoothed, nearest_neighbor,hybrid)
    '''
    # Pull parameters from rubbersheet cfg
    method = rubbersheet_params['outlier_filling_method']
    error_channel = journal.error('rubbersheet.run.fill_outliers_holes')
    info_channel = journal.info('rubbersheet.run.fill_outliers_holes')
    if method == 'nearest_neighbor':
        # Use nearest neighbor interpolation from scipy.ndimage
        invalid = np.isnan(offset)
        indices = ndimage.distance_transform_edt(invalid,
                                                 return_distances=True,
                                                 return_indices=True)
        offset_filled = offset[tuple(indices)]
    elif method == 'fill_smoothed':
        filter_size = rubbersheet_params['fill_smoothed']['kernel_size']
        iterations = rubbersheet_params['fill_smoothed']['iterations']
        offset_filled = _fill_nan_with_mean(offset, offset, filter_size)
        nan_count = np.count_nonzero(np.isnan(offset_filled))
        info_channel.log(f'Number of outliers at first filling iteration {nan_count}')
        while nan_count != 0 and iterations != 0:
              iterations -= 1
              nan_count = np.count_nonzero(np.isnan(offset_filled))
              info_channel.log(f'Number of outliers: {nan_count} at iteration: {iterations}')
              offset_filled = _fill_nan_with_mean(offset_filled, offset_filled, filter_size)
    else:
        err_str = f"{method} invalid method to fill outliers holes"
        error_channel.log(err_str)
        raise ValueError(err_str)

    return offset_filled


def _fill_nan_with_mean(arr_in, arr_out, neighborhood_size):
    '''
    Fill NaN locations in 'arr_in' with the mean of 'arr_out'
    pixels centered in a neighborhood of size 'neighborhood_size'
    around the NaN location. If the neighborhood contains only NaNs,
    then a NaN gets assigned in arr_in for that location.

    Parameters
    ----------
    arr_in: np.ndarray
        Array with outliers to fill
    arr_out: np.ndarray
        Array to use to compute value to replace
        NaNs in 'arr_in'
    neighborhood_size: int
        Size of the square neighborhood to compute replacement
        for NaNs in arr_in
    '''
    filled_arr = arr_in.copy()
    nan_mask = np.isnan(arr_in)

    # Create a kernel for computing the local mean
    kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.float32)

    # Replace NaNs in arr_out with zeros temporarily
    masked_arr_out = np.where(np.isnan(arr_out), 0, arr_out)

    # Use convolve to compute the local sum
    local_sum = ndimage.convolve(masked_arr_out, kernel, mode='constant', cval=0.0)

    # Count non-NaN contributions in the neighborhood
    valid_mask = np.isfinite(arr_out).astype(np.float32)  # Use np.isfinite for valid values
    valid_counts = ndimage.convolve(valid_mask, kernel, mode='constant', cval=0.0)

    # Avoid division by zero by replacing zero counts with NaN
    valid_counts[valid_counts == 0] = np.nan

    # Compute local mean
    local_means = local_sum / valid_counts

    # Fill NaNs in the input array with the computed local means
    filled_arr[nan_mask] = local_means[nan_mask]

    return filled_arr

    
def _offset_blending(off_product_dir, rubbersheet_params, layer_keys):
    '''
    Blends offsets layers at different resolution. Implements a
    pyramidal filling algorithm using the offset layer at higher
    resolution (i.e., layer1). NaN locations in this layer are
    filled with the mean of pixel in the subsequent layer at
    coarser resolution (i.e., layer2) computed in a neighborhood
    of a predefined size.

    Parameters
    ---------
    off_product_dir: str
        Path to the directory containing the unfiltered
        pixel offsets layers
    rubbersheet_params: dict
        Dictionary containing the user-defined rubbersheet options
    layer_keys: list
        List of layers within the offset product

    Returns
    -------
    offset_az, offset_rg: [np.ndarray, np.ndarray]
        Blended pixel offsets layers in azimuth and slant range
    '''
    # Get neighborhood size
    filter_size = rubbersheet_params['fill_smoothed']['kernel_size']

    # Filter outliers from layer one
    offset_az, offset_rg = identify_outliers(str(off_product_dir / layer_keys[0]),
                                             rubbersheet_params)

    # Replace the NaN locations in layer1 with the mean of pixels in layers
    # at lower resolution computed in a neighborhood centered at the NaN location
    # and with a size equal to 'filter_size'
    for layer_key in layer_keys[1:]:
        nan_count_az = np.count_nonzero(np.isnan(offset_az))
        nan_count_rg = np.count_nonzero(np.isnan(offset_rg))

        if nan_count_az > 0:
            offset_az_culled, _ = identify_outliers(str(off_product_dir / layer_key),
                                                    rubbersheet_params)
            offset_az = _fill_nan_with_mean(offset_az, offset_az_culled, filter_size)

        if nan_count_rg > 0:
            _, offset_rg_culled = identify_outliers(str(off_product_dir / layer_key),
                                                    rubbersheet_params)
            offset_rg = _fill_nan_with_mean(offset_rg, offset_rg_culled, filter_size)
    
    # Fill remaining holes by iteratively filling the output offset layer
    offset_az = fill_outliers_holes(offset_az,
                                    rubbersheet_params)
    offset_rg = fill_outliers_holes(offset_rg,
                                    rubbersheet_params)

    return offset_az, offset_rg


import numpy as np
from scipy import interpolate
import journal

def _interpolate_offsets(offset, interp_method):
    '''
    Replace NaN in offset with interpolated values

    Parameters
    ----------
    offset : np.ndarray
        Numpy array containing residual outliers (NaN)
    interp_method : str
        Interpolation method ('linear', 'nearest', 'cubic', or 'no_interpolation')

    Returns
    -------
    offset_interp : np.ndarray
        Interpolated numpy array
    '''
    info_channel = journal.info('rubbersheet._interpolate_offsets')

    # Create mask of valid (non-NaN) data
    valid_mask = ~np.isnan(offset)

    # Skip interpolation if all values are NaN
    if not np.any(valid_mask):
        info_channel.log('Warning: No valid data points. Skipping interpolation')
        return np.full_like(offset, 0)

    # If user chose to skip interpolation
    if interp_method == 'no_interpolation':
        info_channel.log('Interpolation skipped by user. Offsets may contain fill value equal to 0')
        offset_filled = np.copy(offset)
        offset_filled[~valid_mask] = 0
        return offset_filled

    # Extract coordinates and values of valid data points
    new_y, new_x = np.nonzero(valid_mask)
    new_values = offset[valid_mask]

    # Prepare interpolation grid
    x = np.arange(offset.shape[1])
    y = np.arange(offset.shape[0])
    xx, yy = np.meshgrid(x, y)

    # Use fast interpolators where possible
    if interp_method == 'nearest':
        interpolator = interpolate.NearestNDInterpolator((new_x, new_y), new_values)
        offset_interp = interpolator(xx, yy)
    elif interp_method == 'linear':
        interpolator = interpolate.LinearNDInterpolator((new_x, new_y), new_values, fill_value=0)
        offset_interp = interpolator(xx, yy)
    elif interp_method == 'cubic':
        offset_interp = interpolate.griddata(
            (new_x, new_y), new_values.ravel(),
            (xx, yy),
            method='cubic',
            fill_value=0)
    else:
        raise ValueError(f"Unsupported interpolation method: '{interp_method}'")

    return offset_interp


def _filter_offsets(offset, rubbersheet_params):
    '''
    Apply low-pass filter on 'offset'

    Parameters
    ---------
    offset: np.ndarray
        Numpy array to filter
    rubbersheet_params: dict
        Dictionary containing rubbersheet options
    '''
    error_channel = journal.error('rubbersheet._filter_offsets')
    filter_type = rubbersheet_params['offsets_filter']
    if filter_type == 'none':
        return offset
    elif filter_type == 'boxcar':
        window_rg = rubbersheet_params['boxcar']['filter_size_range']
        window_az = rubbersheet_params['boxcar']['filter_size_azimuth']
        kernel = np.ones((window_az, window_rg), dtype=np.float32) / (window_az * window_rg)
        return signal.convolve2d(offset, kernel, mode='same')
    elif filter_type == 'median':
        window_rg = rubbersheet_params['median']['filter_size_range']
        window_az = rubbersheet_params['median']['filter_size_azimuth']
        return ndimage.median_filter(offset, [window_az, window_rg])
    elif filter_type == 'gaussian':
        sigma_range = rubbersheet_params['gaussian']['sigma_range']
        sigma_azimuth = rubbersheet_params['gaussian']['sigma_azimuth']
        return ndimage.gaussian_filter(offset, [sigma_azimuth, sigma_range])
    else:
        err_str = "Not a valid filter option to filter rubbersheeted offsets"
        error_channel.log(err_str)
        raise ValueError(err_str)



if __name__ == "__main__":
    '''
    Run rubbersheet to filter out outliers in the
    slant range and azimuth offset fields. Fill no
    data values left by outliers holes and resample
    culled offsets to reference RSLC shape.
    '''
    # Prepare rubbersheet parser & runconfig
    rubbersheet_parser = YamlArgparse()
    args = rubbersheet_parser.parse()
    rubbersheet_runconfig = RubbersheetRunConfig(args)

    # Prepare RIFG. Culled offsets will be
    # allocated in RIFG product
    out_paths = prepare_insar_hdf5.run(rubbersheet_runconfig.cfg)
    run(rubbersheet_runconfig.cfg, out_paths['RIFG'])
