import pathlib
import journal
import numpy as np
from osgeo import gdal, osr

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


def _read_gdal_with_bbox(input_raster, bbox, bbox_epsg=4326):
    """
    Read only the raster subset intersecting the input bbox, and return it
    reprojected to bbox_epsg.

    Parameters
    ----------
    input_raster : gdal.Dataset
        Input GDAL raster
    bbox : list[float]
        [xmin, ymin, xmax, ymax]
    bbox_epsg : int
        EPSG of bbox coordinates

    Returns
    -------
    arr : numpy.ndarray
        Raster subset reprojected to bbox_epsg
    raster_info : list[float]
        [block_x0, block_y0, block_dx, block_dy] in bbox_epsg coordinates
    """
    gt = input_raster.GetGeoTransform()
    proj = input_raster.GetProjection()
    band = input_raster.GetRasterBand(1)

    if band is None:
        raise RuntimeError("Failed to access raster band.")

    if gt is None:
        raise RuntimeError("Raster geotransform is missing.")

    # north-up only, same practical assumption as most GeoTIFF use cases here
    if gt[2] != 0 or gt[4] != 0:
        raise NotImplementedError(
            "_read_gdal_with_bbox currently supports only north-up rasters."
        )

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(proj)

    try:
        raster_srs.AutoIdentifyEPSG()
    except Exception:
        pass

    raster_epsg = raster_srs.GetAuthorityCode(None)
    if raster_epsg is None:
        raise RuntimeError("Could not determine raster EPSG from projection.")
    raster_epsg = int(raster_epsg)

    xmin, ymin, xmax, ymax = map(float, bbox)

    # target SRS = bbox_epsg
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(int(bbox_epsg))
    try:
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass

    # First, transform bbox corners to raster CRS only to check overlap
    if bbox_epsg != raster_epsg:
        tx_to_raster = osr.CoordinateTransformation(dst_srs, raster_srs)

        corners = [
            tx_to_raster.TransformPoint(xmin, ymin)[:2],
            tx_to_raster.TransformPoint(xmin, ymax)[:2],
            tx_to_raster.TransformPoint(xmax, ymin)[:2],
            tx_to_raster.TransformPoint(xmax, ymax)[:2],
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        xmin_src, xmax_src = min(xs), max(xs)
        ymin_src, ymax_src = min(ys), max(ys)
    else:
        xmin_src, xmax_src = xmin, xmax
        ymin_src, ymax_src = ymin, ymax

    x0 = gt[0]
    dx = gt[1]
    y0 = gt[3]
    dy = gt[5]

    raster_width = input_raster.RasterXSize
    raster_height = input_raster.RasterYSize

    raster_xmin = min(x0, x0 + raster_width * dx)
    raster_xmax = max(x0, x0 + raster_width * dx)
    raster_ymin = min(y0, y0 + raster_height * dy)
    raster_ymax = max(y0, y0 + raster_height * dy)

    xmin_i = max(xmin_src, raster_xmin)
    xmax_i = min(xmax_src, raster_xmax)
    ymin_i = max(ymin_src, raster_ymin)
    ymax_i = min(ymax_src, raster_ymax)

    if xmin_i >= xmax_i or ymin_i >= ymax_i:
        raise ValueError("Input bbox does not overlap raster.")

    src_nodata = band.GetNoDataValue()
    if src_nodata is None:
        src_nodata = 0

    # Use source pixel size magnitude to define output resolution in target CRS.
    # This keeps behavior simple and avoids very strange defaults from Warp.
    # For your case (bbox_epsg=4326), output grid becomes lon/lat.
    if bbox_epsg == raster_epsg:
        out_xres = abs(dx)
        out_yres = abs(dy)
    else:
        tx_to_bbox = osr.CoordinateTransformation(raster_srs, dst_srs)

        # transform two neighboring source points to estimate target resolution
        p00 = tx_to_bbox.TransformPoint(x0, y0)[:2]
        p10 = tx_to_bbox.TransformPoint(x0 + dx, y0)[:2]
        p01 = tx_to_bbox.TransformPoint(x0, y0 + dy)[:2]

        est_dx = abs(p10[0] - p00[0])
        est_dy = abs(p01[1] - p00[1])

        # fallback in case estimate becomes zero or numerically unstable
        if not np.isfinite(est_dx) or est_dx <= 0:
            est_dx = max((xmax - xmin) / 1000.0, 1e-6)
        if not np.isfinite(est_dy) or est_dy <= 0:
            est_dy = max((ymax - ymin) / 1000.0, 1e-6)

        out_xres = est_dx
        out_yres = est_dy

    # Warp directly to the requested bbox / requested CRS
    warped_ds = gdal.Warp(
        "",
        input_raster,
        format="MEM",
        dstSRS=dst_srs.ExportToWkt(),
        outputBounds=(xmin, ymin, xmax, ymax),
        outputBoundsSRS=dst_srs.ExportToWkt(),
        xRes=out_xres,
        yRes=out_yres,
        resampleAlg=gdal.GRA_NearestNeighbour,
        srcNodata=src_nodata,
        dstNodata=src_nodata,
        targetAlignedPixels=False,
        multithread=False,
    )

    if warped_ds is None:
        raise RuntimeError("gdal.Warp failed to create warped subset.")

    warped_band = warped_ds.GetRasterBand(1)
    if warped_band is None:
        raise RuntimeError("Failed to access warped raster band.")

    arr = warped_band.ReadAsArray()
    if arr is None:
        raise RuntimeError("Failed to read warped raster window.")

    warped_gt = warped_ds.GetGeoTransform()
    if warped_gt is None:
        raise RuntimeError("Warped raster geotransform is missing.")

    block_x0 = warped_gt[0]
    block_y0 = warped_gt[3]
    block_dx = warped_gt[1]
    block_dy = warped_gt[5]

    return arr, [block_x0, block_y0, block_dx, block_dy]


def _find_rdr2geo_paths(scratch_path, freq):
    """
    Find x.rdr and y.rdr files for the given frequency inside scratch_path.

    The function searches recursively because the files may exist in several
    possible directories such as:

        scratch/rdr2geo/freqA/
        scratch/ionosphere/main_diff_ms_band/rdr2geo/freqB/
        scratch/ionosphere/main_side_band/rdr2geo/freqB/

    Parameters
    ----------
    scratch_path : pathlib.Path
    freq : str

    Returns
    -------
    dict
        {"x": path_to_x_rdr, "y": path_to_y_rdr}

    Raises
    ------
    FileNotFoundError
        If the rdr2geo files cannot be located.
    """

    scratch_path = pathlib.Path(scratch_path)

    candidates = list(
        scratch_path.glob(f"**/rdr2geo/freq{freq}/x.rdr")
    )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find any x.rdr under {scratch_path} "
            f"for frequency {freq}."
        )

    # choose the shallowest path (closest to scratch root)
    candidates.sort(key=lambda p: len(p.parts))
    x_path = candidates[0]

    y_path = x_path.parent / "y.rdr"

    if not y_path.exists():
        raise FileNotFoundError(
            f"Found {x_path} but corresponding y.rdr does not exist."
        )

    return {"x": str(x_path), "y": str(y_path)}


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

    az_looks = cfg["processing"]["crossmul"]["azimuth_looks"]
    rg_looks = cfg["processing"]["crossmul"]["range_looks"]
    unw_az_looks = cfg["processing"]["phase_unwrap"]["azimuth_looks"]
    unw_rg_looks = cfg["processing"]["phase_unwrap"]["range_looks"]
    if unw_az_looks != 1:
        az_looks = unw_az_looks
    if unw_rg_looks != 1:
        rg_looks = unw_rg_looks

    topo_paths = _find_rdr2geo_paths(scratch_path, freq)

    _, output_dtype = _get_gdal_raster_shape_type(input_data_path)
    geo_data_raster = gdal.Open(input_data_path)

    # for both x and y rasters, decimate and get extents
    decimated_blocks = {}
    decimated_extents = {}
    for xy, input_path in topo_paths.items():
        # open input raster for reading
        input_data_raster = gdal.Open(input_path)
        input_data = input_data_raster.ReadAsArray()
        rows, cols = input_data.shape
        # if multi-looks are 1 or 2,
        # slice_az_end and slice_rg_end are 0. To avoid the positive
        # number, we take None.
        az_size = rows // az_looks
        rg_size = cols // rg_looks
        slice_az_start = int(az_looks / 2)
        slice_az_end = az_size * az_looks
        slice_rg_start = int(rg_looks / 2)
        slice_rg_end = rg_size * rg_looks

        # take center pixels of block to decimate
        decimated_arr = \
            input_data[slice_az_start:slice_az_end:az_looks,
                       slice_rg_start:slice_rg_end:rg_looks]

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

    return output_arrays


def interpret_subswath_mask(subswath_mask, nodata=255):
    """
    Interprets a subswath mask integer by decoding its digits into boolean
    flags indicating reference validity, secondary validity, and water
    presence.

    Parameters
    ----------
    subswath_mask : numpy.array
        Each digit represents a specific flag:
        - Units digit (1s place): Secondary subswath mask
            Non-zero indicates valid; zero indicates invalid.
        - Tens digit (10s place): Reference subswath mask
            Non-zero indicates valid; zero indicates invalid.
        - Hundreds digit (100s place): Water presence flag.
            Non-zero indicates presence of water; zero indicates absence.
    nodata : int, default 255

    Returns
    -------
    reference_valid : bool
        True if the reference is valid (tens digit is non-zero),
        False otherwise.
    secondary_valid : bool
        True if the secondary is valid (units digit is non-zero),
        False otherwise.
    water : bool
        True if water is present (hundreds digit is non-zero),
        False otherwise.
    """
    arr = np.asarray(subswath_mask)

    nd = (arr == nodata)

    secondary_valid = subswath_mask % 10 != 0
    reference_valid = (subswath_mask // 10) % 10 != 0
    water = (subswath_mask // 100) % 10 != 0

    secondary_valid = np.where(nd, False, secondary_valid)
    reference_valid = np.where(nd, False, reference_valid)
    water = np.where(nd, False, water)

    return reference_valid, secondary_valid, water
    