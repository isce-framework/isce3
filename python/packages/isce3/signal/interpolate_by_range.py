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
    """Interpolate/resample an array from side-band slant grid to main-band grid.

    This function supports:
      - integer / uint8 / bool masks: nearest-neighbor interpolation
      - real-valued float arrays: linear interpolation
      - complex arrays: phase-preserving phasor interpolation by default

    Parameters
    ----------
    slant_main : numpy.ndarray
        Target slant range array, usually frequency A / main band.
    slant_side : numpy.ndarray
        Source slant range array, usually frequency B / side band.
    array_side : numpy.ndarray
        2D array on the side-band grid.
    Returns
    -------
    array_main : numpy.ndarray
        Array resampled to the main-band slant grid.
    """
    slant_main = np.asarray(slant_main)
    slant_side = np.asarray(slant_side)
    array_side = np.asarray(array_side)

    if array_side.ndim != 2:
        raise ValueError(
            f"`array_side` must be 2D, but got shape {array_side.shape}"
        )

    row_side, width_side = array_side.shape
    if len(slant_main) == width_side:
        return array_side

    if len(slant_side) != width_side:
        raise ValueError(
            "`len(slant_side)` must match `array_side.shape[1]`.\n"
            f"len(slant_side)     = {len(slant_side)}\n"
            f"array_side.shape[1] = {width_side}"
        )

    # If the target and source grids are already the same, return as-is.
    # Do not check only length; the coordinates should also match.
    if len(slant_main) == len(slant_side) and np.allclose(slant_main, slant_side):
        return array_side

    is_discrete = (
        np.issubdtype(array_side.dtype, np.integer)
        or np.issubdtype(array_side.dtype, np.bool_)
    )

    is_complex = np.issubdtype(array_side.dtype, np.complexfloating)

    # Case 1: integer / uint8 / bool masks
    # Use nearest neighbor to preserve categorical values and bit fields.
    if is_discrete:
        array_main = np.zeros(
            (row_side, len(slant_main)),
            dtype=array_side.dtype
        )

        idx = np.searchsorted(slant_side, slant_main)
        idx = np.clip(idx, 1, len(slant_side) - 1)

        left_idx = idx - 1
        right_idx = idx

        left_dist = np.abs(slant_main - slant_side[left_idx])
        right_dist = np.abs(slant_main - slant_side[right_idx])

        nearest_idx = np.where(left_dist <= right_dist, left_idx, right_idx)

        for row_ind in range(row_side):
            array_main[row_ind, :] = array_side[row_ind, nearest_idx]

        return array_main

    if is_complex:
        # Convert complex data to unit phasor so the phase is interpolated
        # without directly interpolating wrapped phase angles.
        amp = np.abs(array_side)

        phasor = np.zeros(array_side.shape, dtype=np.complex64)
        valid = amp > 0
        phasor[valid] = array_side[valid] / amp[valid]

        array_main = np.zeros(
            (row_side, len(slant_main)),
            dtype=np.complex64
        )

        for row_ind in range(row_side):
            real_interp = np.interp(
                slant_main,
                slant_side,
                np.real(phasor[row_ind, :])
            )
            imag_interp = np.interp(
                slant_main,
                slant_side,
                np.imag(phasor[row_ind, :])
            )

            z = real_interp + 1j * imag_interp

            # Normalize again to unit magnitude.
            z_abs = np.abs(z)
            valid_z = z_abs > 0

            z_out = np.zeros(z.shape, dtype=np.complex64)
            z_out[valid_z] = z[valid_z] / z_abs[valid_z]

            array_main[row_ind, :] = z_out

        return array_main.astype(array_side.dtype, copy=False)

    # Case 3: real-valued float science data
    # Use linear interpolation.
    if np.issubdtype(array_side.dtype, np.floating):
        array_main = np.zeros(
            (row_side, len(slant_main)),
            dtype=array_side.dtype
        )

        for row_ind in range(row_side):
            array_main[row_ind, :] = np.interp(
                slant_main,
                slant_side,
                array_side[row_ind, :]
            )

        return array_main

    raise TypeError(
        f"Unsupported dtype for interpolation: {array_side.dtype}"
    )
