import journal
import numpy as np

from scipy.ndimage import median_filter


def preprocess_wrapped_igram(igram, coherence, mask=None,
                             mask_type='coherence', threshold=0.5,
                             filter_size=9,
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
    mask_type: str, {'median_filter', 'coherence'}, optional
        Type of mask to identify invalid pixels
        'median_filter':
        Compute mask of invalid pixels by thresholding the median absolute 
        deviation w.r.t. the local neighborhood around each pixel.

        'coherence':
        The default mode. Compute mask of invalid pixels by thresholding
        the normalized InSAR coherence.   
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
    # Not a valid algorithm to mask pixels
    else:
        err_str = f"{mask_type} is an invalid selection for mask_type"
        error_channel.log(err_str)
        raise ValueError(err_str)

    # Fill invalid interferogram pixels using user-defined algorithm
    # Distance-based interpolator Chen et al. _[1]
    if filling_method == 'distance_interpolator':
        pha_filt = distance_interpolator(np.angle(igram), distance,
                                         invalid_mask)
    else:
        err_str = f"{filling_method} is an invalid selection for filling_method"
        error_channel.log(err_str)
        raise ValueError(err_str)
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