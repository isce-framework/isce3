import pathlib
import journal
import numpy as np
from osgeo import gdal

from scipy.ndimage import median_filter, map_coordinates


def preprocess_wrapped_igram(igram, coherence, mask=None,
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
    # 1-1) Based on user-provided mask
    # 1-2) Based on water mask
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
    # Not a valid algorithm to mask pixels
    else:
        err_str = f"{mask_type} is an invalid selection for mask_type"
        error_channel.log(err_str)
        raise ValueError(err_str)

    if filling_enabled:
        # Fill invalid interferogram pixels using user-defined algorithm
        # Distance-based interpolator Chen et al. _[1]
        if filling_method == 'distance_interpolator':
            phase_filt = distance_interpolator(np.angle(igram), distance,
                                            invalid_mask)
        else:
            err_str = f"{filling_method} is an invalid selection for filling_method"
            error_channel.log(err_str)
            raise ValueError(err_str)
    else:
        igram[invalid_mask==1] = 0
        phase_filt = np.angle(igram)
    # Go to complex value
    igram_filt = np.exp(1j * phase_filt)

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


def _gdal_type_to_np_type_str(gd_type):
    '''
    Convenience function to convert GDAL data type to numpy data type string
    '''
    gdal_type_to_np_dict = {1: "int8",
                            2: "uint16",
                            3: "int16",
                            4: "uint32",
                            5: "int32",
                            6: "float32",
                            7: "float64",
                            10: "complex64",
                            11: "complex128",}
    return gdal_type_to_np_dict[gd_type]


def _get_gdal_raster_shape_type(raster_path):
    '''
    Convenience function to get shape and numpy data type of GDAL-openable
    raster
    '''
    data_raster = gdal.Open(raster_path)

    data_shape = [data_raster.RasterYSize, data_raster.RasterXSize]

    data_band = data_raster.GetRasterBand(1)
    data_type = data_band.DataType
    np_data_type = _gdal_type_to_np_type_str(data_type)

    return data_shape, np_data_type


def _read_gdal_with_bbox(gdal_raster, bbox):
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

    idx_start = max(0, idx_start)
    idy_start = max(0, idy_start)

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
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    rdr2geo_path = f'{scratch_path}/rdr2geo'

    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]
    # prepare input paths
    topo_paths = {xy: f'{rdr2geo_path}/freq{freq}/{xy}.rdr' for xy in 'xy'}

    # get input shape and type - input type also output type
    _, output_dtype = _get_gdal_raster_shape_type(input_data_path)
    geo_data_raster = gdal.Open(input_data_path)

    # for both x and y rasters, decimate and get extents
    decimated_blocks = {}
    decimated_extents = {}
    for xy, input_path in topo_paths.items():
        # open input raster for reading
        input_data_raster = gdal.Open(input_path)
        input_data = input_data_raster.ReadAsArray()

        # take center pixels of block to decimate
        decimated_arr = \
            input_data[int(az_looks/2):-int(az_looks/2)+1:az_looks,
                       int(rg_looks/2):-int(rg_looks/2)+1:rg_looks]

        # save decimated extents and array for current axis
        decimated_extents[xy] = [np.nanmin(decimated_arr),
                                 np.nanmax(decimated_arr)]
        decimated_blocks[xy] = decimated_arr
        del input_data

    # get bounding for decimated extents
    bbox = [decimated_extents['x'][0], decimated_extents['y'][0],
            decimated_extents['x'][1], decimated_extents['y'][1]]

    # read map bounded by decimated extents of xy block
    input_arr_block, [block_x0, block_y0, block_dx, block_dy] = \
        _read_gdal_with_bbox(geo_data_raster, bbox)

    # prepare output array
    output_arrays = np.zeros(decimated_blocks['y'].shape,
                             dtype=output_dtype)

    # prepare coordinates to map to
    coordinates = ((decimated_blocks['y'] - block_y0) / block_dy,
                   (decimated_blocks['x'] - block_x0) / block_dx)
    # map input raster to decimated coordinates
    map_coordinates(input_arr_block,
                    coordinates,
                    output=output_arrays,
                    mode='nearest',
                    order=0,
                    cval=np.nan,
                    prefilter=False)

    # stack to make whole then return
    return output_arrays
