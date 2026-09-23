
import numpy as np
from isce3.math import offsets_polyfit

def _make_test_data(N=400, degree=2, seed=42):
    """
    Build synthetic tie-point data for a quadratic (degree=2) model
    with known coefficients and no noise.
    """

    rng = np.random.default_rng(seed)

    lines = rng.uniform(0.0, 1000.0, size=N)
    pixels = rng.uniform(0.0, 2000.0, size=N)

    minL, maxL = lines.min(), lines.max()
    minP, maxP = pixels.min(), pixels.max()

    A = offsets_polyfit.build_design_matrix(lines, pixels, degree,
                                            minL, maxL, minP, maxP)

    M = offsets_polyfit.ncoeffs(degree)
    assert A.shape == (N, M)

    true_coefL = np.array([0.3, -0.2, 0.1, 0.05, -0.04, 0.02])
    true_coefP = np.array([-0.7, 0.5, 0.2, -0.03, 0.06, -0.01])

    true_coefL = true_coefL[:M]
    true_coefP = true_coefP[:M]

    yL = A @ true_coefL
    yP = A @ true_coefP

    ids = np.arange(N, dtype=np.int64)

    # equal weights
    corr_peak = np.ones(N, dtype=float)

    data = np.column_stack([ids, lines, pixels, yL, yP, corr_peak])
    return data, true_coefL, true_coefP, (minL, maxL, minP, maxP)

def test_polyfit_offsets_quadratic_no_outliers():
    """
    With perfect quadratic data (degree=2) and no outliers, polyfit_offsets should:
    - recover the true quadratic coefficients (within tight tolerance)
    - not remove any points.
    """
    degree = 2
    data, true_coefL, true_coefP, _ = _make_test_data(
        N=400, degree=degree
    )

    result = offsets_polyfit.polyfit_offsets(
        data.copy(),
        degree=degree,
        crit_value=0.5,   # residuals are exactly 0, so this is generous
        max_iterations=20,
    )

    est_coefL = result["coefL"]
    est_coefP = result["coefP"]

    # No points should be removed
    assert result["removed_indices"] == []
    assert result["inliers"].shape[0] == data.shape[0]

    # Coefficients should match very closely
    np.testing.assert_allclose(est_coefL, true_coefL, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(est_coefP, true_coefP, rtol=1e-6, atol=1e-6)


def test_polyfit_offsets_quadratic_removes_outliers():
    """
    Add a few very bad outliers to quadratic data.
    polyfit_offsets should:
    - remove those outliers
    - still recover coefficients close to the true ones.
    """
    degree = 2
    N = 400
    n_outliers = 8

    data, true_coefL, true_coefP, _ = _make_test_data(
        N=N, degree=degree, seed=123
    )

    rng = np.random.default_rng(123)
    outlier_ids = rng.choice(N, size=n_outliers, replace=False)

    # Inject strong outliers in dL/dP
    data[outlier_ids, 3] += 30.0  # dL
    data[outlier_ids, 4] -= 25.0  # dP

    result = offsets_polyfit.polyfit_offsets(
        data.copy(),
        degree=degree,
        crit_value=0.1,
        max_iterations=100,
    )

    est_coefL = result["coefL"]
    est_coefP = result["coefP"]
    removed_ids = set(result["removed_indices"])

    # We expect all injected outliers to be removed
    assert removed_ids.issuperset(set(data[outlier_ids, 0]))

    # At least as many removed as injected (robust loop may kill a few extra)
    assert len(removed_ids) >= n_outliers

    # Coefficients should still be reasonably close to the true ones
    np.testing.assert_allclose(est_coefL, true_coefL, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(est_coefP, true_coefP, rtol=1e-2, atol=1e-2)