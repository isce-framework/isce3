import numpy as np
import h5py
import warnings

from scipy.ndimage import distance_transform_edt

from isce3.math.offsets_polyfit import (ncoeffs,
                                        polyfit_offsets,
                                        predict_offsets)


def check_qfsp_flag(slc_path):
    """
    Check whether a NISAR RSLC reports an input-data exception.

    A missing dataset is treated as False for compatibility with
    older products. This does not establish that the input is
    free of artifacts.

    Parameters
    ----------
    slc_path : path-like
        Path to the input RSLC HDF5 product.

    Returns
    -------
    bool
        True if the product reports an input-data exception;
        False if the flag is false or the dataset is absent.

    Warns
    -----
    RuntimeWarning
        If the input-data exception dataset is absent.
    """
    qfsp_path = "/science/LSAR/identification/hasInputDataException"

    with h5py.File(slc_path, "r") as src:
        if qfsp_path not in src:
            warnings.warn(
                f"{slc_path} does not contain {qfsp_path}; "
                "treating the input-data exception flag as False.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        value = src[qfsp_path][()]

    return bool(np.asarray(value).item())


def _fit_2d_polynomial_surface(
    data,
    valid_mask,
    order,
    crit_value,
    max_iterations,
    quality_weights=None,
    max_samples=None,
    minimum_quality=0.05,
):
    """
    Robustly fit a weighted 2D polynomial phase surface using
    ``polyfit_offsets``.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional phase array.

    valid_mask : numpy.ndarray
        Boolean mask identifying samples available for polynomial fitting.

    order : int
        Total degree of the two-dimensional polynomial.

    crit_value : float
        Critical value passed directly to ``polyfit_offsets`` for its
        iterative w-test. Larger values reject fewer samples.

    max_iterations : int
        Maximum number of outliers removed by ``polyfit_offsets``. Because
        one sample is removed per iteration, this is also the maximum number
        of rejected samples.

    quality_weights : numpy.ndarray or None, optional
        Optional quality values, such as coherence, with the same shape as
        ``data``. If omitted, all valid samples receive equal weight.

    max_samples : int or None, optional
        Maximum number of samples used for fitting. If necessary,
        deterministic subsampling is applied.

    minimum_quality : float, optional
        Minimum quality value used when constructing fitting weights.

    Returns
    -------
    surface : numpy.ndarray
        Fitted polynomial surface with the same shape as ``data``.

    fit_info : dict
        Polynomial coefficients and outlier-rejection diagnostics.
    """
    data = np.asarray(data, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if data.ndim != 2:
        raise ValueError("data must be 2D")

    if valid_mask.shape != data.shape:
        raise ValueError(
            "valid_mask must have the same shape as data"
        )

    if quality_weights is not None:
        quality_weights = np.asarray(
            quality_weights,
            dtype=float,
        )

        if quality_weights.shape != data.shape:
            raise ValueError(
                "quality_weights must have the same shape as data"
            )

    nrows, ncols = data.shape

    if nrows < 2 or ncols < 2:
        raise ValueError(
            "data must contain at least two rows and two columns"
        )

    good = valid_mask & np.isfinite(data)

    if quality_weights is not None:
        good &= np.isfinite(quality_weights)
        good &= quality_weights > 0

    flat_indices = np.flatnonzero(good)
    n_available = flat_indices.size
    n_coefficients = ncoeffs(order)

    if n_available <= n_coefficients:
        raise ValueError(
            "Not enough valid pixels for polynomial fitting: "
            f"{n_available} samples for "
            f"{n_coefficients} coefficients"
        )

    # Deterministically reduce the fitting samples when requested.
    if (
        max_samples is not None
        and n_available > max_samples
    ):
        selected = np.linspace(
            0,
            n_available - 1,
            max_samples,
            dtype=np.int64,
        )
        flat_indices = flat_indices[selected]

    n_samples = flat_indices.size

    if n_samples <= n_coefficients:
        raise ValueError(
            "Not enough samples after subsampling: "
            f"{n_samples} samples for "
            f"{n_coefficients} coefficients"
        )
    lines, pixels = np.unravel_index(
        flat_indices,
        data.shape,
    )

    phase_samples = data.ravel()[flat_indices]
    n_samples = phase_samples.size

    if quality_weights is None:
        fit_weights = np.ones(
            n_samples,
            dtype=float,
        )
    else:
        quality_samples = quality_weights.ravel()[
            flat_indices
        ]

        quality_samples = np.clip(
            quality_samples,
            minimum_quality,
            1.0,
        )

        # polyfit_offsets squares the supplied weight in the
        # weighted least-squares objective. The square root makes
        # the effective objective weight approximately equal to
        # the supplied quality value.
        fit_weights = np.sqrt(quality_samples)

    # polyfit_offsets input columns:
    #
    # [id, line, pixel, dL, dP, correlation_weight]
    #
    # The phase is fitted as dL. The unused dP observation is zero.
    # Original flattened pixel indices are used as IDs so rejected
    # samples can be mapped back to image coordinates.
    fit_data = np.column_stack(
        (
            flat_indices.astype(float),
            lines.astype(float),
            pixels.astype(float),
            phase_samples,
            np.zeros(n_samples, dtype=float),
            fit_weights,
        )
    )

    min_line = 0.0
    max_line = float(nrows - 1)
    min_pixel = 0.0
    max_pixel = float(ncols - 1)

    fit_result = polyfit_offsets(
        fit_data,
        degree=order,
        crit_value=crit_value,
        max_iterations=max_iterations,
        minL=min_line,
        maxL=max_line,
        minP=min_pixel,
        maxP=max_pixel,
    )

    yy, xx = np.indices(
        (nrows, ncols),
        dtype=float,
    )

    surface, _ = predict_offsets(
        lines=yy,
        pixels=xx,
        coefL=fit_result["coefL"],
        coefP=fit_result["coefP"],
        degree=order,
        minL=min_line,
        maxL=max_line,
        minP=min_pixel,
        maxP=max_pixel,
    )

    removed_flat_indices = np.asarray(
        fit_result["removed_indices"],
        dtype=np.int64,
    )

    if removed_flat_indices.size > 0:
        removed_rows, removed_cols = np.unravel_index(
            removed_flat_indices,
            data.shape,
        )
    else:
        removed_rows = np.array([], dtype=np.int64)
        removed_cols = np.array([], dtype=np.int64)
    n_removed = removed_flat_indices.size

    fit_info = {
        "coefficients": fit_result["coefL"],
        "crit_value": crit_value,
        "max_iterations": max_iterations,
        "n_available": n_available,
        "n_samples": n_samples,
        "n_inliers": fit_result["inliers"].shape[0],
        "n_removed": n_removed,
        "reached_removal_limit": (
            max_iterations > 0
            and n_removed >= max_iterations
        ),
        "removed_flat_indices": removed_flat_indices,
        "removed_rows": removed_rows,
        "removed_cols": removed_cols,
    }

    return surface, fit_info


def _find_column_groups(artifact_mask):
    """
    Find contiguous column groups containing artifact pixels.

    Parameters
    ----------
    artifact_mask : numpy.ndarray
        Two-dimensional boolean array in which True indicates an
        artifact-affected pixel.

    Returns
    -------
    groups : list[slice]
        Slices identifying contiguous artifact column groups.
        Each slice includes its start index and excludes its stop index.
    """
    affected_cols = np.any(artifact_mask, axis=0)

    # Pad with False to detect groups touching either image boundary.
    padded = np.concatenate(([False], affected_cols, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])

    return [
        slice(int(start), int(end))
        for start, end in zip(edges[::2], edges[1::2])
    ]


def _moving_average_1d(x, win):
    """
    Apply a NaN-aware moving average to a one-dimensional array.

    The moving average is computed using only finite samples within each
    window. Non-finite samples do not contribute to either the sum or the
    number of samples used to calculate the average.

    Parameters
    ----------
    x : array_like
        One-dimensional input data.
    win : int
        Size of the moving-average window. If `win` is less than or equal
        to 1, a copy of the input array is returned without smoothing.

    Returns
    -------
    out : numpy.ndarray
        Smoothed floating-point array with the same shape as `x`. Output
        samples are NaN where the corresponding window contains no finite
        input samples.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()

    win = min(win, x.size)

    if win <= 1:
        return x.copy()

    valid = np.isfinite(x).astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)
    kernel = np.ones(win, dtype=float)

    num = np.convolve(x0, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")

    out = np.full_like(x, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def make_feather_weight(
    artifact_mask,
    inner_shrink=2,
    outer_feather=10,
):
    """
    Create a smooth feathering weight for artifact correction.

    Parameters
    ----------
    artifact_mask : 2D bool array
        True where the artifact correction should be strongest.

    inner_shrink : int
        Pixels inside the artifact edge where weight begins tapering.
        Larger values make the full-strength region smaller.

    outer_feather : int
        Number of pixels outside artifact_mask over which correction tapers to 0.

    Returns
    -------
    weight : 2D float array
        Smooth weight from 0 to 1.
    """
    artifact_mask = np.asarray(artifact_mask, dtype=bool)

    if inner_shrink < 0:
        raise ValueError("inner_shrink must be non-negative")

    if outer_feather < 0:
        raise ValueError("outer_feather must be non-negative")

    if inner_shrink == 0 and outer_feather == 0:
        return artifact_mask.astype(np.float32)

    dist_inside = distance_transform_edt(artifact_mask)
    dist_outside = distance_transform_edt(~artifact_mask)

    # Positive inside the artifact and negative outside.
    signed_distance = dist_inside - dist_outside

    transition_width = inner_shrink + outer_feather

    weight = (
        signed_distance + outer_feather
    ) / float(transition_width)

    weight = np.clip(weight, 0.0, 1.0)

    # Smooth the linear feather weight using a cubic polynomial
    #
    #     S(w) = a*w**3 + b*w**2 + c*w + d.
    #
    # Requiring S(0)=0, S(1)=1, and zero endpoint slopes,
    # S'(0)=S'(1)=0, gives a=-2, b=3, and c=d=0. Thus,
    #
    #     S(w) = 3*w**2 - 2*w**3 = w**2 * (3 - 2*w).
    #
    # The zero endpoint slopes reduce correction-gradient discontinuities
    # at the feather boundaries.
    weight = weight * weight * (3.0 - 2.0 * weight)

    return weight.astype(np.float32)


def correct_qfsp_phase_artifact(
    phase,
    fit_background_mask,
    artifact_mask,
    background_order,
    background_crit_value,
    background_max_iterations,
    background_max_samples,
    background_minimum_quality,
    template_smooth_win,
    inner_shrink,
    outer_feather,
    background_weights=None,
    fill_value=np.nan,
    debug_flag=False,
):
    """
    Estimate and remove a range-dependent qFSP phase artifact.

    A smooth two-dimensional background surface is first fitted to valid,
    non-artifact pixels and subtracted from the input phase. For each
    contiguous group of artifact-affected range columns, a one-dimensional
    artifact template is estimated from the residual phase by averaging over
    rows. The template is optionally smoothed, extended beyond the detected
    artifact columns, replicated along the row direction, and subtracted
    from the original phase using a feathered correction weight.

    Parameters
    ----------
    phase : numpy.ndarray
        Two-dimensional differential interferogram unwrapped phase array with shape
        ``(nrows, ncols)``. Non-finite values and zero-valued pixels are
        treated as invalid and are excluded from background fitting and
        template estimation.

    fit_background_mask : numpy.ndarray
        Boolean array with the same shape as ``phase`` identifying pixels
        available for background fitting and artifact-template estimation.
        Artifact pixels do not need to be removed from this mask: they are
        excluded internally from the background fit using ``artifact_mask``
        but retained for template estimation. Pixels set to ``False`` are
        excluded from both operations.

    artifact_mask : numpy.ndarray
        Boolean array with the same shape as ``phase``. Pixels set to
        ``True`` identify locations affected by the qFSP artifact.

    background_order : int, default=2
        Polynomial order of the two-dimensional surface fitted to the valid,
        non-artifact phase pixels. For example, 0 fits a constant, 1 fits a
        planar surface, and 2 includes quadratic terms.

    background_crit_value : float
        Critical value passed directly to ``polyfit_offsets`` for iterative
        outlier rejection. Larger values result in fewer rejected samples.

    background_max_iterations : int
        Maximum number of samples removed during iterative outlier rejection.
        ``polyfit_offsets`` removes at most one sample per iteration.

    background_max_samples : int or None
        Maximum number of phase samples used for polynomial fitting.

    background_minimum_quality : float
        Lower bound applied to quality values before converting them to
        fitting weights.

    template_smooth_win : int, default=3
        Window length, in range pixels, used to smooth each one-dimensional
        artifact template. Values less than or equal to 1 disable template
        smoothing.

    inner_shrink : int, default=2
        Width, in pixels, of the transition applied inside the artifact-mask
        boundary by ``make_feather_weight``. This reduces the correction
        strength near the inner boundary of the detected artifact region.

    outer_feather : int, default=10
        Number of pixels by which the artifact model and correction region
        are extended outside each detected artifact group. The correction
        weight tapers across this region to reduce boundary discontinuities.

    background_weights : numpy.ndarray or None, optional
        Optional phase-quality values, such as coherence, with the same shape
        as ``phase``. If ``None``, all valid samples receive equal weight.

    fill_value : float, default=numpy.nan
        Value used to initialize ``artifact_2d`` at pixels where no artifact
        model is estimated. Non-finite fill values prevent correction at
        those pixels through ``valid_corr``.

    debug_flag : bool, optional
        If True, return the corrected phase and diagnostic outputs as a
        dictionary. If False, return only the corrected phase array.
        Defaults to False.

    Returns
    -------
    result : numpy.ndarray or dict
        Corrected phase array when debug_flag is False.
        When debug_flag is True, a dictionary containing the corrected
        phase and diagnostic outputs.

        ``"corrected_phase"`` : numpy.ndarray
            Copy of the input phase after subtracting the weighted artifact
            model where a valid correction is available.

        ``"artifact_2d"`` : numpy.ndarray
            Estimated two-dimensional artifact model. Each one-dimensional
            range template is replicated along the row direction.

        ``"background"`` : numpy.ndarray
            Fitted two-dimensional polynomial background surface.

        ``"residual"`` : numpy.ndarray
            Detrended phase, computed as ``phase - background``.

        ``"fit_mask"`` : numpy.ndarray
            Boolean mask of valid, non-artifact pixels used for background
            fitting.

        ``"template_estimation_mask"`` : numpy.ndarray
            Boolean mask of valid artifact pixels used to estimate the
            one-dimensional templates.

        ``"groups"`` : list[slice]
            Slices identifying contiguous artifact column groups.
            Each slice includes its start index and excludes its stop index.

        ``"templates"`` : list of numpy.ndarray
            One-dimensional residual-phase template estimated for each
            detected artifact group. A template may contain non-finite
            values where insufficient valid samples are available.

        ``"correction_weight"`` : numpy.ndarray
            Two-dimensional feather weight applied to the artifact model.
            (same dimensions as `phase`)

        ``"valid_corr"`` : numpy.ndarray
            Boolean mask identifying pixels where the correction was
            actually applied.
    """
    phase = np.asarray(phase, dtype=float)
    fit_background_mask = np.asarray(fit_background_mask, dtype=bool)
    artifact_mask = np.asarray(artifact_mask, dtype=bool)

    if phase.ndim != 2:
        raise ValueError("phase must be 2D")

    nrows, ncols = phase.shape

    for name, arr in [
        ("fit_background_mask", fit_background_mask),
        ("artifact_mask", artifact_mask),
    ]:
        if arr.shape != phase.shape:
            raise ValueError(f"{name} must have same shape as phase")

    valid_phase = np.isfinite(phase) & (phase != 0)

    # Pixels used to fit the background: clean, valid, non-artifact pixels.
    fit_mask = (
        fit_background_mask
        & (~artifact_mask)
        & valid_phase
    )

    # Pixels used to estimate the artifact template: valid artifact pixels.
    template_estimation_mask = (
        artifact_mask
        & valid_phase
        & fit_background_mask
    )
    background, _ = (
        _fit_2d_polynomial_surface(
            phase,
            fit_mask,
            order=background_order,
            crit_value=background_crit_value,
            max_iterations=background_max_iterations,
            quality_weights=background_weights,
            max_samples=background_max_samples,
            minimum_quality=background_minimum_quality,
        )
    )

    residual = phase - background

    groups = _find_column_groups(artifact_mask)

    artifact_2d = np.full_like(phase, fill_value, dtype=float)
    templates = [] if debug_flag else None

    for col_slice in groups:
        group_residual = residual[:, col_slice]

        valid_for_template = (
            template_estimation_mask[:, col_slice]
            & np.isfinite(group_residual)
        )

        tmp = np.where(valid_for_template, group_residual, np.nan)
        template = np.nanmean(tmp, axis=0)

        if np.all(~np.isfinite(template)):
            if debug_flag:
                templates.append(template)
            continue

        if template_smooth_win > 1:
            template = _moving_average_1d(template, template_smooth_win)

        if debug_flag:
            templates.append(template)

        # Extend the artifact model slightly outside the artifact columns
        # so feathering can smoothly taper the correction.
        c0, c1 = col_slice.start, col_slice.stop
        c0_ext = max(0, c0 - outer_feather)
        c1_ext = min(ncols, c1 + outer_feather)

        template_ext = np.interp(
            np.arange(c0_ext, c1_ext),
            np.arange(c0, c1),
            template,
            left=template[0],
            right=template[-1],
        )

        artifact_group_ext = np.tile(template_ext[None, :], (nrows, 1))

        valid_ext = np.isfinite(artifact_group_ext)
        artifact_2d[:, c0_ext:c1_ext][valid_ext] = (
            artifact_group_ext[valid_ext]
        )

    correction_weight = make_feather_weight(
        artifact_mask,
        inner_shrink=inner_shrink,
        outer_feather=outer_feather,
    )

    corrected_phase = phase.copy()

    valid_corr = (
        (correction_weight > 0)
        & np.isfinite(artifact_2d)
        & valid_phase
    )

    corrected_phase[valid_corr] -= (
        correction_weight[valid_corr] * artifact_2d[valid_corr]
    )

    if not debug_flag:
        return corrected_phase

    return {
        "corrected_phase": corrected_phase,
        "artifact_2d": artifact_2d,
        "background": background,
        "residual": residual,
        "fit_mask": fit_mask,
        "template_estimation_mask": template_estimation_mask,
        "groups": groups,
        "templates": templates,
        "correction_weight": correction_weight,
        "valid_corr": valid_corr,
    }
