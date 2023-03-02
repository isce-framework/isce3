import journal
import numpy as np
import pathlib
from osgeo import gdal

from scipy.ndimage import median_filter
from scipy.ndimage import map_coordinates
from isce3.signal.filter_data import (get_raster_block,
                                      block_param_generator)
from isce3.atmosphere.ionosphere_filter import write_array

def preprocess_wrapped_igram(igram, coherence, water_mask, mask=None,
                             mask_type='coherence', threshold=0.5,
                             filter_size=9,
                             filling_enabled=True,
                             filling_method='distance_interpolator',
                             distance=5):
    '''
    Preprocess wrapped interferograms prior to phase unwrapping.

    Removes invalid pixels in wrapped interferograms based on
    user-defined metric. Invalid pixels may be identified using
    1) a water mask; 2) thresholding low-coherence pixels; 3) thresholding
    the median absolute deviation of the interferogram phase from the local median.
    Invalid pixels are replaced with values computed with a distance-weighted
    interpolation approach from Chen et al., 2015. The magnitude of the complex
    interferogram is discarded.

    Parameters
    ----------
    igram: numpy.ndarray
        Wrapped interferogram to pre-process
    coherence: numpy.ndarray
        Normalized InSAR coherence
    water_mask: numpy.ndarray
        Binary water mask (water:1, nonwater:0)
    mask: numpy.ndarray or None
        Optional binary mask (1: invalid; 0: valid) to identify invalid pixels.
        If a mask is provided, data-driven masking is not performed (other
        masking options are ignored;
    mask_type: str, {'median_filter', 'coherence', 'water'}, optional
        Type of mask to identify invalid pixels
        'median_filter':
        Compute mask of invalid pixels by thresholding the median absolute
        deviation w.r.t. the local neighborhood around each pixel.

        'coherence':
        The default mode. Compute mask of invalid pixels by thresholding
        the normalized InSAR coherence.

        'water':
        Project the water mask to radar grid and masks out the invalid
        pixels

    threshold: float
        Threshold to identify invalid pixels.
        If 'mask_type' is 'coherence' pixels with coherence below threshold
        are considered invalid
        If 'mask_type' is 'median_filter' pixels with median absolute
        deviation (MAD) above this threshold are considered outliers
    filter_size: int
        Size of median filter for median absolute deviation
        outlier identification method
    filling_method: str
        Algorithm to fill invalid pixels. 'distance_interpolator'
        applies distance weighted interpolation from Chen et al., 2015
    distance: int
        Distance metric for interpolation. For distance interpolator in
        Chen et al [1]_ is distance is intended as radius

    Returns
    -------
    filt_igram: numpy.ndarray
        Wrapped interferogram with outlier pixel being filtered
        out and replaced with pixels computed by the selected
        'filling_method'. The magnitude of the input wrapped interferogram
        is discarded.

    References
    ----------
    .. [1] J. Chen, H. A. Zebker,and R. Knight, ""A persistent scatterer interpolation
       for retrieving accurate ground deformation over InSAR-decorrelated
       Agricultural fields", Geoph. Res. Lett., 42(21), 9294-9301, (2015).
    '''

    # Extract some preprocess options
    error_channel = journal.error('unwrap.run.preprocess_wrapped_igram')

    # Create mask of invalid pixels
    invalid_mask = np.full(igram.shape, dtype=bool, fill_value=False)

    # Identify invalid pixels and store them in a mask.
    # Criteria to identify invalid pixels:
    # 1) based on user-provided mask
    if mask is not None:
        invalid_mask[mask == 1] = True
    # 2) Based on InSAR correlation values
    elif mask_type == 'coherence':
        invalid_mask[coherence < threshold] = True
    # 3) Based on median absolute deviation (MAD)
    elif mask_type == 'median_filter':
        igram_pha = np.angle(igram)
        mad = median_absolute_deviation(igram_pha, filter_size)
        invalid_mask[mad > threshold] = True
    # 4) Based on water mask
    elif mask_type == 'water':
        invalid_mask[water_mask==1] = True
    # Not a valid algorithm to mask pixels
    else:
        err_str = f"{mask_type} is an invalid selection for mask_type"
        error_channel.log(err_str)
        raise ValueError(err_str)

    if filling_enabled:
        # Fill invalid interferogram pixels using user-defined algorithm
        # Distance-based interpolator Chen et al. _[1]
        if filling_method == 'distance_interpolator':
            pha_filt = distance_interpolator(np.angle(igram), distance,
                                            invalid_mask)
        else:
            err_str = f"{filling_method} is an invalid selection for filling_method"
            error_channel.log(err_str)
            raise ValueError(err_str)
    else:
        igram[invalid_mask==1] = 0
        pha_filt = np.angle(igram)
    # Go to complex value
    igram_filt = np.exp(-1j * pha_filt)

    return igram_filt


def distance_interpolator(arr, radius, invalid_mask):
    '''
    Interpolate pixels based on distance from valid pixels
    following Chen et al [1]_.

    Parameters
    ----------
    arr: numpy.ndarray
        Array containing invalid pixel locations to fill
    radius: int
        Radius of the sampling/filling window
    invalid_mask: numpy.ndarray
        Boolean mask identifying invalid pixels (True:invalid)

    Returns
    -------
    fill_arr: numpy.ndarray
        Array with interpolated values at invalid pixel locations

    References
    __________
    .. [1] J. Chen, H. A. Zebker,and R. Knight, ""A persistent scatterer interpolation
       for retrieving accurate ground deformation over InSAR-decorrelated
       Agricultural fields", Geoph. Res. Lett., 42(21), 9294-9301, (2015).
    '''
    arr_filt = np.copy(arr)

    # Get center locations
    x_cent, y_cent = np.where(invalid_mask == True)

    # Find the coordinates of valid pixels
    x, y = np.where(invalid_mask == False)

    for xc, yc in zip(x_cent, y_cent):
        # Compute distance between center pixel and valid pixels
        ps_dist = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        # Compute weights based on distance and selected radius
        w = np.exp(-ps_dist ** 2 / 2 * radius)
        # Compute Eq. 2 of Chen at al [1]_
        weighted_arr = arr_filt[x, y].flatten() * w
        arr_filt[xc, yc] = np.nansum(weighted_arr) / np.nansum(w)

    return arr_filt


def median_absolute_deviation(arr, filter_size):
    '''
    Compute the median absolute deviation (MAD) of `arr`
    defined as median(abs(arr - median(arr))

    Parameters
    ----------
    arr: numpy.ndarray
        Array for which to compute MAD
    filter_size: int
        Size of median filter, in pixels

    Returns
    -------
    mad: numpy.ndarray
        Median absolute deviation of `arr`
    '''
    med = np.abs(arr - median_filter(arr, [filter_size, filter_size]))
    mad = median_filter(med, [filter_size, filter_size])
    return mad

def read_gdal_with_bbox(gdal_raster, bbox):
    '''
    Extract image from the gdal-supported file with bbox
    Parameters
    ----------
    gdal_raster: osgeo.gdal.Dataset
        gdal dataset to extract the subset image
    bbox: list
        list of [xmin, ymin, xmax, ymax]

    Returns
    -------
    subset_data: numpy.ndarray
        Median absolute deviation of `arr`
    [sub_x0, sub_y0, dx, dy]: list
        sub_x0: x coordinate of upper left
        sub_y0: y coordinate of upper left
        dx: x spacing
        dy: y spacing
    '''
    xmin, ymin, xmax, ymax = bbox

    geotransform = gdal_raster.GetGeoTransform()
    x0 = geotransform[0]
    y0 = geotransform[3]
    dx = geotransform[1]
    dy = geotransform[5]

    idx_start = int(np.floor((xmin - x0) / dx))
    idx_end = int(np.ceil((xmax - x0) / dx))

    if dy > 0:
        idy_start = int(np.floor((ymin - y0) / dy))
        idy_end = int(np.ceil((ymax - y0) / dy))
        sub_y0 = idy_start*dy + y0

    else:
        idy_start = int(np.floor((ymax - y0) / dy))
        idy_end = int(np.ceil((ymin - y0) / dy))

    if idx_start < 0:
        idx_start = 0
    if idy_start < 0:
        idy_start = 0

    x_width = idx_end - idx_start
    y_length = idy_end - idy_start

    if x_width > gdal_raster.RasterXSize:
        x_width = gdal_raster.RasterXSize
    if y_length > gdal_raster.RasterYSize:
        y_length = gdal_raster.RasterYSize

    sub_x0 = idx_start*dx + x0
    sub_y0 = idy_start*dy + y0
    raster_band = gdal_raster.GetRasterBand(1)
    subset_data = raster_band.ReadAsArray(idx_start,
                                          idy_start,
                                          x_width,
                                          y_length)

    return subset_data, [sub_x0, sub_y0, dx, dy]

def decimate_with_looks(input_path, output_path, rlooks, alooks):
    '''
    Decimate the 2 D image with range and azimuth looks

    Parameters
    ----------
    input_path: str
        input file path to be decimated
    output_path: list
        output file path of decimated image
    rlooks: int
        number of range look
    alooks: int
        number of azimuth look
    '''
    input_raster = gdal.Open(input_path)
    input_band = input_raster.GetRasterBand(1)
    data_type = input_band.DataType

    input_shape = [input_raster.RasterYSize, input_raster.RasterXSize]
    output_shape = [int(input_shape[0]/alooks), int(input_shape[1]/rlooks)]

    block_params = block_param_generator(
        alooks * 500, data_shape=input_shape, pad_shape=(0, 0))

    for block_ind, block_param in enumerate(block_params):
        input_data = get_raster_block(input_path, block_param)
        decimated_data = input_data[int(alooks/2):-int(alooks/2):alooks,
                                    int(rlooks/2):-int(rlooks/2):rlooks]

        write_array(output_path, decimated_data,
                    data_type=data_type,
                    block_row=int(block_param.write_start_line/alooks),
                    data_shape=output_shape)

def project_map_to_radar(cfg, input_data_path, freq):
    '''
    Project map coordinate image to radar grid

    Parameters
    ----------
    cfg: dict
        input runconfig file
    input_data_path: str
        input file path for map coordinate image
    freq: str
        frequency to be projected

    Returns
    -------
    rdr_data: numpy.ndarray
        projected data into radar grid  absolute
    '''

    input_hdf5 = cfg['input_file_group']['reference_rslc_file']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    rdr2geo_path = f'{scratch_path}/rdr2geo'

    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]

    gdal_type_to_np_dict = {1: "int8",
                            2: "uint16",
                            3: "int16",
                            4: "uint32",
                            5: "int32",
                            6: "float32",
                            7: "float64",
                            10: "complex64",
                            11: "complex128",}

    lines_per_block = az_looks * 200

    data_raster = gdal.Open(input_data_path)
    data_band = data_raster.GetRasterBand(1)
    data_type = data_band.DataType
    np_data_type = gdal_type_to_np_dict[data_type]

    for xy in ['x', 'y']:
        topo_x_str = f'{rdr2geo_path}/freq{freq}/{xy}.rdr'
        decimate_topo_str = \
            f'{rdr2geo_path}/freq{freq}/{xy}_r{rg_looks}_a{az_looks}.rdr'
        decimate_with_looks(topo_x_str, decimate_topo_str, rg_looks, az_looks)

    decimate_topo_x_str = \
        f'{rdr2geo_path}/freq{freq}/x_r{rg_looks}_a{az_looks}.rdr'
    decimate_topo_y_str = \
        f'{rdr2geo_path}/freq{freq}/y_r{rg_looks}_a{az_looks}.rdr'

    gdal_obj = gdal.Open(decimate_topo_x_str)
    rows = gdal_obj.RasterYSize
    cols = gdal_obj.RasterXSize
    del gdal_obj

    block_params = block_param_generator(
        lines_per_block, data_shape=(rows, cols), pad_shape=(0, 0))

    rdr_data = np.zeros([int(rows), int(cols)], dtype=np_data_type)

    for block_ind, block_param in enumerate(block_params):
        xx_bin = get_raster_block(decimate_topo_x_str, block_param)
        yy_bin = get_raster_block(decimate_topo_y_str, block_param)

        bbox = [np.nanmin(xx_bin),
                np.nanmin(yy_bin),
                np.nanmax(xx_bin),
                np.nanmax(yy_bin)]

        data_sub, [sub_x0, sub_y0, sub_dx, sub_dy] = \
            read_gdal_with_bbox(data_raster, bbox=bbox)

        dest_yy = ((yy_bin - sub_y0) / sub_dy)
        dest_xx = ((xx_bin - sub_x0) / sub_dx)

        sr_data_temp = np.zeros(yy_bin.shape, dtype=np_data_type)

        coordinates = (dest_yy, dest_xx)

        map_coordinates(data_sub,
                        coordinates,
                        output=sr_data_temp,
                        mode='nearest',
                        cval=np.nan,
                        prefilter=False)

        multi_look_start = int(np.round(block_param.write_start_line))
        multi_look_end = multi_look_start + \
                         int(np.round(block_param.block_length))
        rdr_data[multi_look_start:multi_look_end, :] = sr_data_temp
    del data_raster

    return rdr_data
