import numpy as np
import pytest

from nisar.workflows.qfsp_ionosphere import (
    correct_qfsp_phase_artifact,
)


@pytest.fixture
def synthetic_case():
    rows, cols = np.indices((40, 60), dtype=float)
    background = 10.0 + 0.02 * rows + 0.03 * cols

    artifact_mask = np.zeros(background.shape, dtype=bool)
    artifact_mask[:, 20:30] = True

    phase = background.copy()
    phase[:, 20:30] += np.linspace(1.0, 3.0, 10)[None, :]

    options = {
        "fit_background_mask": np.ones_like(artifact_mask),
        "artifact_mask": artifact_mask,
        "background_order": 1,
        "background_crit_value": 3.0,
        "background_max_iterations": 0,
        "background_max_samples": None,
        "background_minimum_quality": 0.05,
        "template_smooth_win": 1,
        "inner_shrink": 0,
        "outer_feather": 0,
    }
    return phase, background, options


@pytest.mark.parametrize("debug_flag", [False, True])
def test_removes_known_artifact(synthetic_case, debug_flag):
    phase, background, options = synthetic_case

    result = correct_qfsp_phase_artifact(
        phase, **options, debug_flag=debug_flag
    )
    corrected = result["corrected_phase"] if debug_flag else result

    np.testing.assert_allclose(
        corrected, background, rtol=0, atol=1e-8
    )

    # No correction should be applied outside the artifact mask.
    clean = ~options["artifact_mask"]
    np.testing.assert_array_equal(corrected[clean], phase[clean])


@pytest.mark.parametrize("debug_flag", [False, True])
def test_preserves_invalid_pixels(synthetic_case, debug_flag):
    # Verify that correction preserves invalid pixels (0, NaN, and ±Inf)
    # and does not modify the input phase array.
    phase, _, options = synthetic_case
    phase = phase.copy()
    phase[0, 22] = 0.0
    phase[1, 23] = np.nan
    phase[2, 24] = np.inf
    phase[3, 25] = -np.inf
    original = phase.copy()

    result = correct_qfsp_phase_artifact(
        phase, **options, debug_flag=debug_flag
    )
    corrected = result["corrected_phase"] if debug_flag else result

    assert corrected[0, 22] == 0.0
    assert np.isnan(corrected[1, 23])
    assert np.isposinf(corrected[2, 24])
    assert np.isneginf(corrected[3, 25])
    np.testing.assert_array_equal(phase, original)