import itertools

import numpy as np
import scipy

def ncoeffs(degree: int) -> int:
    """
    Compute the number of coefficients in a 2D polynomial
    with total degree less than or equal to a given value.

    Parameters
    ----------
    degree : int
        The maximum total degree of the 2D polynomial.
        (Same in both dimensions)

    Returns
    -------
    int
        The number of unique coefficients in the polynomial.
    """
    return (degree + 1) * (degree + 2) // 2


def normalize_to_unit_range(x, xmin=None, xmax=None):
    """
    Linearly normalize values to the range [0, 1].

    Parameters
    ----------
    x : array_like
        Input array to be normalized.
    xmin : float or int
        Minimum value of the input range (default: None).
    xmax : float or int
        Maximum value of the input range (default: None).

    Returns
    -------
    ndarray
        Normalized array with values scaled to the interval [0, 1].
        If `xmax` equals `xmin`, returns an array of zeros with the same shape as `x`.
    """

    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax

    if xmax == xmin:
        return np.zeros_like(x, dtype=float)

    return (x - xmin) / (xmax - xmin)

def _expo_pairs(degree: int):
    """
    Generate exponent pairs for a 2D polynomial of a given total degree.

    For each total degree ``j = 0..degree`` and partial degree ``k = 0..j``,
    this function produces the exponents ``(a, b)`` corresponding to the term:

        term = l**a * p**b

    The resulting arrays define the exponent structure used to construct the
    polynomial design matrix or to evaluate the polynomial.

    Parameters
    ----------
    degree : int
        Total degree of the 2D polynomial.

    Returns
    -------
    A : (M,) ndarray of int
        Exponents for the line variable `l`, where
        ``M = (degree + 1) * (degree + 2) // 2``.
    B : (M,) ndarray of int
        Exponents for the pixel variable `p`, matching the order of `A`.
    """

    A, B = [], []
    for j in range(degree + 1):
        k = np.arange(j + 1)
        A.append(j - k)
        B.append(k)
    return np.concatenate(A).astype(np.int64), np.concatenate(B).astype(np.int64)

def build_design_matrix(lines, pixels, degree,
                        minL, maxL, minP, maxP):
    """
    Build the full 2D polynomial design matrix for arrays of line and pixel coordinates.

    Each row corresponds to one (line, pixel) pair, and each column corresponds to
    a polynomial term of the form ``l**a * p**b`` for all combinations of exponents
    satisfying ``a + b <= degree``. Normalization of the coordinates ensures
    consistent scaling with the fitting process.

    Parameters
    ----------
    lines : (N,) array_like
        Array of line coordinates.
    pixels : (N,) array_like
        Array of pixel coordinates.
    degree : int
        Total degree of the 2D polynomial.
    minL, maxL : float
        Minimum and maximum line coordinates used for normalization.
    minP, maxP : float
        Minimum and maximum pixel coordinates used for normalization.

    Returns
    -------
    A : (N, M) ndarray
        Design matrix where ``M = (degree + 1) * (degree + 2) // 2``.
        Each element ``A[i, j]`` represents the j-th polynomial term evaluated
        at the i-th normalized coordinate pair.
    """

    lines = np.asarray(lines, dtype=float)
    pixels = np.asarray(pixels, dtype=float)

    # Normalize
    l = normalize_to_unit_range(lines, minL, maxL)
    p = normalize_to_unit_range(pixels, minP, maxP)

    l = np.ravel(l)
    p = np.ravel(p)

    Aidx, Bidx = _expo_pairs(degree)
    ar = np.arange(degree + 1, dtype=np.int64)

    l_pows = np.power(l[:, None], ar[None, :])
    p_pows = np.power(p[:, None], ar[None, :])

    return l_pows.take(Aidx, axis=1) * p_pows.take(Bidx, axis=1)

def cholesky_solve(N_mat, rhs):
    """
    Solve a symmetric positive-definite linear system using Cholesky decomposition.

    This function solves the system:

        N_mat * x = rhs

    where `N_mat` is a symmetric positive-definite matrix, by performing
    a Cholesky decomposition `N_mat = L @ L.T` and solving the two
    triangular systems `L y = rhs` and `L.T x = y`.

    Parameters
    ----------
    N_mat : (M, M) array_like
        Symmetric positive-definite coefficient matrix.
    rhs : (M,) or (M, K) array_like
        Right-hand side vector or matrix.

    Returns
    -------
    x : ndarray
        Solution vector or matrix satisfying `N_mat @ x = rhs`.
    L : ndarray
        Lower-triangular Cholesky factor of `N_mat`.
    """

    L = scipy.linalg.cholesky(N_mat, lower=True)
    x = scipy.linalg.cho_solve((L, True), rhs)

    return x, L

def invert_from_cholesky(L):
    """
    Compute the inverse of a symmetric positive-definite matrix from its
    Cholesky factorization.

    Given a matrix `A` such that `A = L @ L.T`, this function computes the
    inverse of `A` as:

        A_inv = (L^-1).T @ (L^-1)

    Parameters
    ----------
    L : (M, M) array_like
        Lower-triangular Cholesky factor of a symmetric positive-definite
        matrix `A`.

    Returns
    -------
    A_inv : (M, M) ndarray
        Inverse of the original matrix `A`.
    """

    Linv = np.linalg.inv(L)
    return Linv.T @ Linv

def polyfit_offsets(
    data,degree=2,
    crit_value=0.1,
    max_iterations=50,
    minL=None, maxL=None,
    minP=None, maxP=None,
    prf=None, abw=None,
    rsr=None, rbw=None):

    """
    Robustly fit 2D polynomial models to SAR line/pixel offsets with iterative
    outlier rejection.

    The function estimates two polynomial surfaces (line and pixel directions)
    from tie-point measurements by building a 2D polynomial design matrix up to
    `degree`, solving with weighted normal equations, and iteratively removing
    outliers using a w-test controlled by `crit_value`.

    Parameters
    ----------
    data : (N, 6) array_like
        Input tie-point array where each row is:
        ``[id, line, pixel, dL, dP, corr_peak]``.
        - *id* : int-like identifier.
        - *line, pixel* : coordinates of the tie point.
        - *dL, dP* : measured offsets in line and pixel directions.
        - *corr_peak* : correlation peak or quality metric in interval (0, 1], and measurements with larger values get more weight.
    degree : int, optional
        Total degree of the 2D polynomial (``1`` = affine, ``2`` = quadratic, ...).
        Default is ``2``.
    crit_value : float, optional
        Critical value for the w-test used to flag outliers at each iteration.
        Smaller values are stricter. Default is ``0.1``.
    max_iterations : int, optional
        Maximum number of outlier-rejection iterations. Default is ``50``.
    minL, maxL : float, optional
        Optional bounds on acceptable *line* offsets (units of `dL`).
        Points with `dL` outside ``[minL, maxL]`` are removed before fitting.
    minP, maxP : float, optional
        Optional bounds on acceptable *pixel* offsets (units of `dP`).
        Points with `dP` outside ``[minP, maxP]`` are removed before fitting.
    prf, abw, rsr, rbw : float, optional
        SAR system parameters (pulse repetition frequency, azimuth bandwidth,
        range sampling rate, range bandwidth) used, if provided, to derive
        prior standard deviations for weighting the line/pixel residuals.

    Returns
    -------
    result : dict
        Dictionary with the following keys:
        - **coefL** : (M,) ndarray
          Polynomial coefficients for the *line* direction, ordered consistently
          with the design matrix construction, where
          ``M = (degree + 1) * (degree + 2) // 2``.
        - **coefP** : (M,) ndarray
          Polynomial coefficients for the *pixel* direction.
        - **inliers** : (K, 6) ndarray
          Subset of `data` retained after robust outlier removal (``K <= N``).
        - **removed_indices** : list of int
          Row indices (w.r.t. the original `data`) that were removed as outliers
          or by hard bounds.
        - **degree** : int
          Polynomial degree actually used.
        - **design_nunk** : int
          Number of unknown coefficients ``M``.
    """

    # Prior std for w-test
    sigmaL = 0.15 / ((prf / abw) if (None not in [prf, abw]) else 1.1)
    sigmaP = 0.10 / ((rsr / rbw) if (None not in [rsr, rbw]) else 1.1)

    maxL = data[:, 1].max() if maxL is None else maxL
    minL = data[:, 1].min() if minL is None else minL
    maxP = data[:, 2].max() if maxP is None else maxP
    minP = data[:, 2].min() if minP is None else minP

    if minL >= maxL:
        raise ValueError(
            f"minL value {minL} greater than or equal to maxL value {maxL}. "
            "minL must be less than maxL.")

    if minP >= maxP:
        raise ValueError(
            f"minP value {minP} greater than or equal to maxP value {maxP}. "
            "minP must be less than maxP.")

    # number of unknonw parameters or coefficients for the polyfitting
    Nunk = ncoeffs(degree)
    eps = np.sqrt(np.finfo(float).eps)

    # The indices that will be removed
    removed_indices = []

    # Build the design matrix
    lines, pixels = data[:, 1], data[:, 2]
    A = build_design_matrix(lines, pixels, degree,
                            minL, maxL, minP, maxP)

    for iteration in itertools.count():
        yL, yP = data[:, 3:4], data[:, 4:5]
        Nobs = data.shape[0]
        if Nobs <= Nunk:
            raise ValueError("No sufficient points for the rubbersheet polyfitting")

        # Weights
        w = np.clip(data[:, 5], eps, 1.0)
        Qy_diag = 1.0 / (w * w)
        W = w[:, None]

        # Weighted normal equations
        A_til = A * W
        At = A_til.T
        Nmat = At @ A_til
        rhsL, rhsP = At @ (yL * W), At @ (yP * W)

        try:
            xL, Lc = cholesky_solve(Nmat, rhsL)
            xP, _  = cholesky_solve(Nmat, rhsP)
        except np.linalg.LinAlgError:
            I = np.eye(Nmat.shape[0])
            xL, Lc = cholesky_solve(Nmat + eps * I, rhsL)
            xP, _  = cholesky_solve(Nmat + eps * I, rhsP)

        Qx_hat = invert_from_cholesky(Lc)

        # Residuals
        yL_hat, yP_hat = A @ xL, A @ xP
        eL, eP = yL - yL_hat, yP - yP_hat

        # diag_Qe = Qy_diag - Qyhat_diag with weights applied in Qy
        Qyhat_diag = np.einsum("ij,jk,ik->i", A, Qx_hat, A)
        diag_Qe = np.clip(Qy_diag - Qyhat_diag, eps, None)

        # w-test
        s = np.sqrt(diag_Qe)
        wL, wP = (eL[:, 0] / (s * sigmaL)), (eP[:, 0] / (s * sigmaP))
        max_any = max(np.abs(wL).max(), np.abs(wP).max())
        if (max_any <= crit_value) or (iteration >= max_iterations):
            return {
                "coefL": xL[:, 0],
                "coefP": xP[:, 0],
                "inliers": data,
                "removed_indices": removed_indices,
                "degree": degree,
                "design_nunk": Nunk,
            }

        # Remove worst outlier
        worst = int(np.argmax(wL * wL + wP * wP))
        removed_indices.append(int(data[worst, 0]))
        data = np.delete(data, worst, axis=0)
        A = np.delete(A, worst, axis=0)

def predict_offsets(lines, pixels, coefL, coefP,
                    degree, minL, maxL, minP, maxP):
    """
    Predict line and pixel offsets (ΔL, ΔP) for arrays of coordinates using
    a 2D polynomial model.

    The prediction uses the same polynomial basis and normalization as the fit:
    normalized coordinates are
    ``l = (lines - minL) / (maxL - minL)`` and
    ``p = (pixels - minP) / (maxP - minP)``, and offsets are computed via
    the 1D design vector ``A(l, p)`` (all terms with total degree ``<= degree``):

        ΔL = A(l, p) · coefL
        ΔP = A(l, p) · coefP

    Parameters
    ----------
    lines : array_like
        Line coordinates of points to predict. Must have the same shape as `pixels`.
    pixels : array_like
        Pixel coordinates of points to predict. Must have the same shape as `lines`.
    coefL : (M,) array_like
        Polynomial coefficients for the line-offset model, where
        ``M = (degree + 1) * (degree + 2) // 2``.
    coefP : (M,) array_like
        Polynomial coefficients for the pixel-offset model (same ordering as `coefL`).
    degree : int
        Total degree of the 2D polynomial used during fitting.
    minL, maxL : float
        Minimum and maximum line coordinates used for normalization.
    minP, maxP : float
        Minimum and maximum pixel coordinates used for normalization.

    Returns
    -------
    dL : ndarray
        Predicted line offsets with the same shape as `lines`/`pixels`.
    dP : ndarray
        Predicted pixel offsets with the same shape as `lines`/`pixels`.
    """

    lines = np.asarray(lines)
    pixels = np.asarray(pixels)
    out_shape = np.broadcast_shapes(lines.shape, pixels.shape)

    A = build_design_matrix(
        np.broadcast_to(lines, out_shape),
        np.broadcast_to(pixels, out_shape),
        degree, minL, maxL, minP, maxP
    )

    C = np.column_stack((np.asarray(coefL, dtype=float).ravel(),
                         np.asarray(coefP, dtype=float).ravel()))
    Y = A @ C

    dL = Y[:, 0].reshape(out_shape)
    dP = Y[:, 1].reshape(out_shape)

    return dL, dP