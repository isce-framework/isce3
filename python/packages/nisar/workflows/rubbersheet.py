'''
Wrapper for rubbersheet
'''

import pathlib
import journal
import time
import h5py
import os
import numpy as np
import isce3
from osgeo import gdal
from scipy import ndimage
from scipy import interpolate
from scipy import signal
from nisar.products.readers import SLC
from nisar.workflows import h5_prep
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.rubbersheet_runconfig import RubbersheetRunConfig


def run(cfg: dict, output_hdf5: str = None):
    '''
    Run rubbersheet
    '''

    # Pull parameters from cfg dictionary
    ref_hdf5 = cfg['input_file_group']['reference_rslc_file']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    rubbersheet_params = cfg['processing']['rubbersheet']
    geo2rdr_offsets_path = pathlib.Path(rubbersheet_params['geo2rdr_offsets_path'])
    off_product_enabled = cfg['processing']['offsets_product']['enabled']

    # If not set, set output HDF5 file
    if output_hdf5 is None:
        output_hdf5 = cfg['product_path_group']['sas_output_file']

    # Set info and error channels
    info_channel = journal.info('rubbersheet.run')
    error_channel = journal.error('rubbersheet.run')

    # Initialize parameters share by frequency A and B
    ref_slc = SLC(hdf5file=ref_hdf5)

    # Start offset culling
    t_all = time.time()
    with h5py.File(output_hdf5, 'r+', libver='latest', swmr=True) as dst_h5:
        for freq, pol_list in freq_pols.items():
            # Rubbersheet directory and frequency group in RIFG product
            rubbersheet_dir = scratch_path / 'rubbersheet_offsets' / f'freq{freq}'
            freq_group_path = f'/science/LSAR/RIFG/swaths/frequency{freq}'

            # Set the path to geometric offsets dir
            geo_offset_dir = geo2rdr_offsets_path / 'geo2rdr' / f'freq{freq}'

            # Loop over polarizations
            for pol in pol_list:
                # Create output directory, identify proper pixelOffsets path in RIFG
                # and get dense_offsets_dir
                out_dir = rubbersheet_dir / pol
                out_dir.mkdir(parents=True, exist_ok=True)
                pol_group_path = f'{freq_group_path}/pixelOffsets/{pol}'
                if not off_product_enabled:
                    dense_offsets_path = pathlib.Path(
                        rubbersheet_params['dense_offsets_path'])
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
                else:
                    # Offsets product is enabled, implement pyramidal filling
                    off_product_path = pathlib.Path(
                        rubbersheet_params['offsets_product_path'])
                    off_product_dir = off_product_path / 'offsets_product' / f'freq{freq}' / pol
                    # Cull the outlier of the offsets layer at finest resolution
                    offset_az, offset_rg = identify_outliers(
                        str(off_product_dir / 'layer1'),
                        rubbersheet_params)

                    layer_keys = [key for key in
                                  cfg['processing']['offsets_product'].keys() if
                                  key.startswith('layer') and key != 'layer1']

                    # Loop over different layer keys (different from 1)
                    length, width = offset_az.shape
                    temp_off_az = np.zeros((length, width), dtype=np.float32)
                    temp_off_rg = np.zeros((length, width), dtype=np.float32)
                    for key in layer_keys:
                        temp_off_az_culled, temp_off_rg_culled = identify_outliers(
                            str(off_product_dir / key),
                            rubbersheet_params)
                        off_az_filled = fill_outliers_holes(
                            temp_off_az_culled,
                            rubbersheet_params)
                        off_rg_filled = fill_outliers_holes(
                            temp_off_rg_culled,
                            rubbersheet_params)
                        temp_off_az += off_az_filled
                        temp_off_rg += off_rg_filled
                    # Divide by the number of layers excluding layer1
                    temp_off_az = temp_off_az / (len(layer_keys) - 1)
                    temp_off_rg = temp_off_rg / (len(layer_keys) - 1)
                    offset_az[np.isnan(offset_az)] = temp_off_az[
                        np.isnan(offset_az)]
                    offset_rg[np.isnan(offset_rg)] = temp_off_rg[
                        np.isnan(offset_rg)]

                # Update offset field in HDF5 file
                offset_az_prod = dst_h5[f'{pol_group_path}/alongTrackOffset']
                offset_rg_prod = dst_h5[f'{pol_group_path}/slantRangeOffset']
                offset_az_prod[...] = offset_az
                offset_rg_prod[...] = offset_rg

                # Save culled offsets to disk for resampling
                off_path = [out_dir/'culled_az_offsets', out_dir/'culled_rg_offsets']
                offsets = [offset_az, offset_rg]

                for path, off_array in zip(off_path, offsets):
                    write_to_disk(str(path), off_array)

                # Expand culled offsets to SLC shape and save as temporary data
                ref_raster_str = f'HDF5:{ref_hdf5}:/{ref_slc.slcPath(freq, pol)}'
                ref_raster_slc = isce3.io.Raster(ref_raster_str)

                for path in off_path:
                    outpath = str(path).replace('culled', 'resampled')
                    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
                    gdal.Translate(outpath, ds,
                                   width=ref_raster_slc.width,
                                   height=ref_raster_slc.length, format="ENVI")

                # Add geometric offsets to rubbersheet offsets using gdal_pixel functions
                # The sum of geometric and rubbersheet offsets is stored in rubbersheet_folder
                geo_path = ['azimuth.off', 'range.off']
                for geo_off, out_off in zip(geo_path, off_path):
                    out_path = f'{str(out_dir / geo_off)}.vrt'
                    write_vrt(str(geo_offset_dir / geo_off),
                              str(out_off).replace('culled', 'resampled'),
                              out_path, ref_raster_slc.width,
                              ref_raster_slc.length, 'Float32', 'sum')

    t_all_elapsed = time.time() - t_all
    info_channel.log(
        f"Successfully ran rubbersheet in {t_all_elapsed:.3f} seconds")


def write_to_disk(outpath, array, format='ENVI',
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
    """Identify outliers in the offset fields.
       Outliers are identified by thresholding a
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
            """

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

    # Check if the user has required to compute a refinement mask.
    # The aim of the refinement is to remove small, residual outliers
    # areas of 3-5 pixels
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

    # Label outliers as NaN
    offset_rg[mask] = np.nan
    offset_az[mask] = np.nan

    return offset_az, offset_rg


def fill_outliers_holes(offset, rubbersheet_params):
    """Fill no data values according to user-preference.
       No data values are filled using one of the following:
       - fill_smoothed: replace no data values with smoothed value
         in a neighborhood.
       - nearest_neighbor: replace no data with nearest neighbor
         interpolation
       - hybrid: Use one iteration of fill smoothed followed by
         linear interpolation
        Parameters
        ----------
        offset: array, float
            2D array with no data values (NaNs) to be filled
        rubbersheet_params: cfg
            Dictionary containing rubbersheet parameters
            from runconfig
        Returns
        -------
        offset_filled: array, float
            2D array with no data values filled with data values
            from one of the algorithms (fill_smoothed, nearest_neighbor,
            hybrid)
        """
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
        # Iteratively replace NaNs with smoothed values in a
        # neighborhood until the desired number of iterations is reached.
        iterations = rubbersheet_params['fill_smoothed']['iterations']
        filter_size = rubbersheet_params['fill_smoothed']['kernel_size']
        nan_count = np.count_nonzero(np.isnan(offset))
        offset_temp = offset
        while nan_count != 0 and iterations != 0:
            iterations -= 1
            nan_count = np.count_nonzero(np.isnan(offset_temp))
            offset_temp = ndimage.generic_filter(offset_temp,
                                                 function=replace_smoothed,
                                                 footprint=np.ones((filter_size, filter_size)),
                                                 mode='wrap')
    else:
        err_str = f"{method} invalid method to fill outliers holes"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Check if NaNs are still present. If yes, perform interpolation
    nan_count = np.count_nonzero(np.isnan(offset_temp))
    if nan_count != 0:
        x = np.arange(0, offset.shape[1])
        y = np.arange(0, offset.shape[0])
        xx, yy = np.meshgrid(x, y)
        new_x = xx[~np.isnan(offset_temp)]
        new_y = yy[~np.isnan(offset_temp)]
        new_array = offset_temp[~np.isnan(offset_temp)]
        offset_filled = interpolate.griddata((new_x, new_y), new_array.ravel(),
                                             (xx, yy),
                                             method=rubbersheet_params['interpolation_method'],
                                             fill_value=0)
    else:
        offset_filled = offset_temp

    # To improve noise (at the expense of resolution), offsets can (or not) be
    # filtered prior to resampling. We filter the offsets according to the
    # user-preferences.
    filter_type = rubbersheet_params['offsets_filter']
    if filter_type == 'none':
        offset_smoothed = offset_filled
    elif filter_type == 'boxcar':
        window_rg = rubbersheet_params['boxcar']['filter_size_range']
        window_az = rubbersheet_params['boxcar']['filter_size_azimuth']
        kernel = np.ones((window_az, window_rg), dtype=np.float32) / (window_az * window_rg)
        offset_smoothed = signal.convolve2d(offset_filled, kernel, mode='same')
    elif filter_type == 'median':
        window_rg = rubbersheet_params['median']['filter_size_range']
        window_az = rubbersheet_params['median']['filter_size_azimuth']
        offset_smoothed = ndimage.median_filter(offset_filled,
                                                [window_az, window_rg])
    elif filter_type == 'gaussian':
        sigma_range = rubbersheet_params['gaussian']['sigma_range']
        sigma_azimuth = rubbersheet_params['gaussian']['sigma_azimuth']
        offset_smoothed = ndimage.gaussian_filter(offset_filled, [sigma_azimuth,
                                                  sigma_range])
    else:
        err_str = "Not a valid filter option to filter rubbersheeted offsets"
        error_channel.log(err_str)
        raise ValueError(err_str)

    return offset_smoothed


def replace_smoothed(x):
    """Replace missing data at the center of x
    with the median of values in x
    Parameters
    ----------
    x: array, float
        Array of float values used for computation
    Returns
    -------
    x[center]: scalar, float
        Center of x array. If no data, x[center] is
        replace by the median of values in x
    """
    length = len(x)
    center = int(length / 2)
    if np.isnan(x[center]) and not np.isnan(np.delete(x, center)).all():
        return np.nanmedian(np.delete(x, center))
    else:
        return x[center]


def write_vrt(file1, file2, out_vrt, width, length, data_type,
              function_name, description='Sum'):
    """Write VRT file using GDAL pixel function capabilities
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

        Returns
        -------
            -
        """
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
    out_paths = h5_prep.run(rubbersheet_runconfig.cfg)
    run(rubbersheet_runconfig.cfg, out_paths['RIFG'])
