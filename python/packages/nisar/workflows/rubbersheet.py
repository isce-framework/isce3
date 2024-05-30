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
                                     get_ground_track_velocity_product)
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
                    sum_off_path = f'{str(out_dir / geo_off)}.vrt'
                    _write_vrt(str(geo_offset_dir / geo_off),
                               resamp_off_path,
                               sum_off_path, ref_radar_grid.width,
                               ref_radar_grid.length, 'Float32', 'sum')

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
                   datatype=gdal.GDT_Float32):
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

    # Pull parameters from cfg
    threshold = rubbersheet_params['threshold']
    window_rg = rubbersheet_params['median_filter_size_range']
    window_az = rubbersheet_params['median_filter_size_azimuth']
    metric = rubbersheet_params['culling_metric']
    error_channel = journal.error('rubbersheet.run.identify_outliers')

    # Open offsets
    ds = gdal.Open(f'{offsets_dir}/dense_offsets')
    offset_az = ds.GetRasterBand(1).ReadAsArray()
    offset_rg = ds.GetRasterBand(2).ReadAsArray()
    ds = None

    # Identify outliers based on user-defined metric
    if metric == 'snr':
        # Open SNR
        ds = gdal.Open(f'{offsets_dir}/snr')
        snr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        mask_data = np.where(snr < threshold)
    elif metric == 'median_filter':
        # Use offsets to compute "median absolute deviation" (MAD)
        median_az = ndimage.median_filter(offset_az, [window_az, window_rg])
        median_rg = ndimage.median_filter(offset_rg, [window_az, window_rg])
        mask_data = (np.abs(offset_az - median_az) > threshold) | (
                np.abs(offset_rg - median_rg) > threshold)
    elif metric == 'covariance':
        # Use offsets azimuth and range covariance elements
        ds = gdal.Open(f'{offsets_dir}/covariance')
        cov_az = ds.GetRasterBand(1).ReadAsArray()
        cov_rg = ds.GetRasterBand(2).ReadAsArray()
        ds = None
        mask_data = (cov_az > threshold) | (cov_rg > threshold)
    else:
        err_str = f"{metric} invalid metric to filter outliers"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # If required, apply refinement mask to remove residual outliers
    # (i.e., clustered areas of 2-5 pixels)
    if rubbersheet_params['mask_refine_enabled']:
        filter_size = rubbersheet_params['mask_refine_filter_size']
        threshold = rubbersheet_params['mask_refine_threshold']
        offset_az[mask_data] = 0
        offset_rg[mask_data] = 0
        median_rg = ndimage.median_filter(offset_rg, filter_size)
        median_az = ndimage.median_filter(offset_az, filter_size)
        mask_refine = (np.abs(offset_az - median_az) > threshold) | (
                np.abs(offset_rg - median_rg) > threshold)
        # Compute final mask
        mask = mask_data | mask_refine
    else:
        mask = mask_data

    # Label identified outliers as NaN
    offset_rg[mask] = np.nan
    offset_az[mask] = np.nan

    return offset_az, offset_rg


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

    if method == 'nearest_neighbor':
        # Use nearest neighbor interpolation from scipy.ndimage
        invalid = np.isnan(offset)
        indices = ndimage.distance_transform_edt(invalid,
                                                 return_distances=True,
                                                 return_indices=True)
        offset_temp = offset[tuple(indices)]
    elif method == 'fill_smoothed':
        filter_size = rubbersheet_params['fill_smoothed']['kernel_size']
        iterations = rubbersheet_params['fill_smoothed']['iterations']
        _fill_nan_with_mean(offset, offset, filter_size)

        # If NaNs are still present, perform iterative filling
        nan_count = np.count_nonzero(np.isnan(offset))
        while nan_count != 0 and iterations != 0:
            iterations -= 1
            nan_count = np.count_nonzero(np.isnan(offset))
            _fill_nan_with_mean(offset, offset, filter_size)
    else:
        err_str = f"{method} invalid method to fill outliers holes"
        error_channel.log(err_str)
        raise ValueError(err_str)

    return offset


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
    kernel = np.ones((neighborhood_size, neighborhood_size))
    kernel /= kernel.sum()

    # Iterate over each NaN pixel in the array
    for i in range(arr_in.shape[0]):
        for j in range(arr_in.shape[1]):
            if nan_mask[i, j]:
                # Extract the local neighborhood around the NaN pixel
                neighborhood = arr_out[max(i - neighborhood_size // 2, 0): min(i + neighborhood_size // 2 + 1, arr_out.shape[0]),
                                       max(j - neighborhood_size // 2, 0): min(j + neighborhood_size // 2 + 1, arr_out.shape[1])]

                # Compute the mean of the non-NaN values in the neighborhood
                neighborhood_mean = np.nanmean(neighborhood)
                # Replace the NaN value with the computed mean
                filled_arr[i, j] = neighborhood_mean


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
            _fill_nan_with_mean(offset_az, offset_az_culled, filter_size)

        if nan_count_rg > 0:
            _, offset_rg_culled = identify_outliers(str(off_product_dir / layer_key),
                                                    rubbersheet_params)
            _fill_nan_with_mean(offset_rg, offset_rg_culled, filter_size)

    return offset_az, offset_rg


def _interpolate_offsets(offset, interp_method):
    '''
    Replace NaN in offset with interpolated values

    Parameters
    ---------
    offset: np.ndarray
        Numpy array containing residual outliers (NaN)
    interp_method: str
        Interpolation method

    Returns
    -------
    offset_interp: np.ndarray
        Interpolated numpy array
    '''
    x = np.arange(0, offset.shape[1])
    y = np.arange(0, offset.shape[0])
    xx, yy = np.meshgrid(x, y)
    new_x = xx[~np.isnan(offset)]
    new_y = yy[~np.isnan(offset)]
    new_array = offset[~np.isnan(offset)]
    offset_interp = interpolate.griddata((new_x, new_y), new_array.ravel(),
                                         (xx, yy),
                                         method=interp_method,
                                         fill_value=0)
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


def _write_vrt(file1, file2, out_vrt, width, length, data_type,
               function_name, description='Sum'):
    '''
    Write VRT file using GDAL pixel function capabilities

    Parameter
    ----------
    file1:  str
        First source file to use in the VRT generation
    file2:  str
        Second source file to use in VRT generation
    out_vrt: str
        Filepath to the output VRT
    width:
        Width of output vrt
    length:
        Length of VRT output file
    data_type:
        Data type of output VRT
    function_name:
        Name of pixel function used to create VRT
    description: str
        Description of the pixel function to create the VRT
    '''
    vrttmpl = f'''
    <VRTDataset rasterXSize="{width}" rasterYSize="{length}">
      <VRTRasterBand dataType="{data_type}" band="1" subClass="VRTDerivedRasterBand">
        <Description>{description}</Description>
        <PixelFunctionType>{function_name}</PixelFunctionType>
        <SimpleSource>
          <SourceFilename>{file1}</SourceFilename>
        </SimpleSource>
        <SimpleSource>
          <SourceFilename>{file2}</SourceFilename>
        </SimpleSource>
      </VRTRasterBand>
    </VRTDataset>'''

    with open(out_vrt, 'w') as fid:
        fid.write(vrttmpl)


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
