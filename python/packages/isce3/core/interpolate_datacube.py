import numpy as np
from scipy.interpolate import interp1d

def interpolate_datacube(data_cube_arr, ori_heights, new_heights):
    """
    Interpolate the datacube with 'ori_heights' to 'new_heights' using linear interpolation.

    Parameters
    ---------
    data_cube_arr : numpy.ndarray
        The data cube array
    ori_heights: array_like
        List of heights along the first dimension of the `data_cube_arr`
    new_heights: array_like
        List of new heights need to be interpolated

    Returns
    ----------
    new_data_cube : numpy.ndarray
        The interpolated data cube
    """

    ori_heights = np.asarray(ori_heights)
    new_heights = np.asarray(new_heights)

    # Original heights must be the same with the first dimension of the data_cube_arr
    if len(ori_heights) != data_cube_arr.shape[0]:
        raise ValueError(
            "size mismatch: len(ori_heights) must equal data_cube_arr.shape[0],"
            f" got {len(ori_heights)} != {data_cube_arr.shape[0]}")

    # Get the min and max heights of the original heights
    min_height, max_height = min(ori_heights), max(ori_heights)

    # Check if the height is out of bounds
    for height in new_heights:
        if (height < min_height) or (height > max_height):
            raise ValueError(f'{height} is out of bounds of {min_height} and {max_height}')

    # Build the interpolator along the height
    interp_func = interp1d(ori_heights, data_cube_arr, axis=0)
    new_data_cube = interp_func(new_heights)

    return new_data_cube