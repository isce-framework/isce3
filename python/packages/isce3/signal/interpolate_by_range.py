import numpy as np


def decimate_freq_a_array(
        slant_main,
        slant_side,
        target_runw):
    """Decimate target_runw of main band to have same size as side band,
    assuming slant_main and slant_side are evenly spaced.

    Parameters
    ----------
    slant_main : numpy.ndarray
        Slant range array of frequency A band
    slant_side : numpy.ndarray
        Slant range array of frequency B band
    target_runw : numpy.ndarray
        RUNW array of frequency A band.
        Width of target_runw should be same as length of slant_main.

    Returns
    -------
    decimated_array : numpy.ndarray
        Decimated RUNW array with width == len(slant_side).
    """
    _, width = target_runw.shape

    first_index = np.argmin(np.abs(slant_main - slant_side[0]))
    spacing_main = slant_main[1] - slant_main[0]
    spacing_side = slant_side[1] - slant_side[0]

    # make sure stride is at least 1
    resampling_scale_factor = max(
        1, int(np.round(spacing_side / spacing_main))
    )

    n_side = len(slant_side)

    # slice whatever overlaps (no shifting); then pad left/right as needed
    end_excl = min(width, first_index + n_side * resampling_scale_factor)
    decimated_array = target_runw[
        :, first_index:end_excl:resampling_scale_factor
    ]
    # how many side samples fall outside main on each side?
    # (assumes increasing slant arrays)
    left_missing = int(
        np.ceil(
            max(0.0, (slant_main[0] - slant_side[0]) / spacing_side)
        )
    )
    right_missing = int(
        np.ceil(
            max(0.0, (slant_side[-1] - slant_main[-1]) / spacing_side)
        )
    )

    # clamp in case both sides miss (very long slant_side)
    total_missing = max(0, n_side - decimated_array.shape[1])
    left_missing = min(left_missing, total_missing)
    right_missing = min(right_missing, total_missing - left_missing)

    if left_missing > 0 or right_missing > 0:
        decimated_array = np.pad(
            decimated_array,
            pad_width=((0, 0), (left_missing, right_missing)),
            mode="constant",
            constant_values=0,
        )

    return decimated_array


def interpolate_freq_b_array(
        slant_main,
        slant_side,
        array_side):
    """interpolate array that have the size of side band (frequency B)
    to have same size with main band assuming slant_main and slant_side
    are evenly spaced

    Parameters
    ----------
    slant_main : numpy.ndarray
        slant range array of frequency A band
    slant_side : numpy.ndarray
        slant range array of frequency B band
    array_side : numpy.ndarray
        array with same size of side-band (frequencyB)
        width of array_side should be same with length of slant_side

    Returns
    -------
    array_main : numpy.ndarray
        oversampled array
    """
    row_side, _ = array_side.shape
    array_main = np.zeros([row_side, len(slant_main)])

    for row_ind in range(0, row_side):

        array_main[row_ind, :] = np.interp(slant_main,
                                           slant_side,
                                           array_side[row_ind, :])

    return array_main
