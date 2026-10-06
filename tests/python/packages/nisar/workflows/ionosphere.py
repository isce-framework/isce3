"""Unit tests and integration smoke tests for the ionosphere workflow.

Run with pytest in an ISCE3 environment with iscetest data available.
Mask decoding is mocked to test workflow logic, not NISAR bit definitions.
"""

import argparse
import os

import h5py
import numpy as np
import pytest

import iscetest
from nisar.products.insar.product_paths import RIFGGroupsPaths, RUNWGroupsPaths
from nisar.workflows import insar
from nisar.workflows import ionosphere as iono
from nisar.workflows.h5_prep import get_products_and_paths
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence


@pytest.fixture
def mask_decoders(monkeypatch):
    """Mock decoding to test mask selection, not NISAR bit encoding."""

    def valid_decoder(mask):
        return (mask & 1) != 0, (mask & 2) != 0

    def fallback_decoder(mask):
        ref, sec = valid_decoder(mask)
        return ref, sec, np.zeros(mask.shape, dtype=bool)

    monkeypatch.setattr(iono, "interpret_valid_data_mask", valid_decoder)
    monkeypatch.setattr(iono, "interpret_subswath_mask", fallback_decoder)


@pytest.mark.parametrize("preferred_exists", [False, True])
def test_get_mask_raster_selection(tmp_path, preferred_exists):
    """Prefer an existing mask even when it is all zero; otherwise fall back."""
    shape = (3, 4)

    swath_path = RIFGGroupsPaths().SwathsPath
    ifg_path = f"{swath_path}/frequencyA/interferogram"
    valid_mask_path = f"{ifg_path}/HH/validDataMask"
    fallback_mask_path = f"{ifg_path}/mask"

    with h5py.File(tmp_path / "mask_RIFG.h5", "w") as h5:
        h5.create_dataset(
            fallback_mask_path, data=np.ones(shape, dtype=np.uint8)
        )
        if preferred_exists:
            h5.create_dataset(
                valid_mask_path, data=np.zeros(shape, dtype=np.uint8)
            )

        raster, uses_preferred = iono._get_mask_raster(
            h5, valid_mask_path, fallback_mask_path
        )
        expected_path = (
            valid_mask_path if preferred_exists else fallback_mask_path
        )
        assert uses_preferred is preferred_exists
        assert raster.name == expected_path
        np.testing.assert_array_equal(
            raster[:], np.full(shape, 0 if preferred_exists else 1)
        )


@pytest.mark.parametrize("first_real", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_differential_phase_values(tmp_path, first_real, in_place):
    """Check differential phase for complex and real-valued inputs.

    Test multiple polarizations, partial-block processing, and output
    written either to a separate file or to the first input file.

    The real-input, same-file case uses synthetic RUNW and RIFG groups
    within one file to exercise shared-handle behavior.
    """
    shape = (5, 7)
    rows, cols = np.indices(shape)
    first_phase = 0.2 + 0.17 * rows + 0.11 * cols
    second_phase = -0.3 + 0.03 * rows - 0.04 * cols

    first_name = "first_RUNW.h5" if first_real else "first_RIFG.h5"
    first_file = str(tmp_path / first_name)
    second_file = str(tmp_path / "second_RIFG.h5")
    output_file = (
        first_file if in_place else str(tmp_path / "difference_RIFG.h5")
    )

    rifg_swath_path = RIFGGroupsPaths().SwathsPath
    runw_swath_path = RUNWGroupsPaths().SwathsPath
    first_swath_path = runw_swath_path if first_real else rifg_swath_path
    first_dataset = (
        "unwrappedPhase" if first_real else "wrappedInterferogram"
    )
    polarizations = ("HH", "HV")

    first_pol_paths = [
        f"{first_swath_path}/frequencyA/interferogram/{pol}"
        for pol in polarizations
    ]
    second_pol_paths = [
        f"{rifg_swath_path}/frequencyA/interferogram/{pol}"
        for pol in polarizations
    ]

    first_data_paths = [
        f"{path}/{first_dataset}" for path in first_pol_paths
    ]
    second_data_paths = [
        f"{path}/wrappedInterferogram" for path in second_pol_paths
    ]
    output_data_paths = second_data_paths.copy()

    first_valid_mask_paths = [
        f"{path}/validDataMask" for path in first_pol_paths
    ]
    second_valid_mask_paths = [
        f"{path}/validDataMask" for path in second_pol_paths
    ]

    # Create the first input: real phase or complex interferogram.
    with h5py.File(first_file, "w") as h5:
        for index, path in enumerate(first_data_paths):
            phase = first_phase + index * 0.4
            data = (
                phase if first_real else 2.0 * np.exp(1j * phase)
            )
            h5.create_dataset(path, data=data)

    # Create the second input as a complex interferogram.
    with h5py.File(second_file, "w") as h5:
        for path in second_data_paths:
            h5.create_dataset(
                path,
                data=3.0 * np.exp(1j * second_phase),
            )

    with h5py.File(output_file, "a") as h5:
        for path in output_data_paths:
            # Complex in-place processing overwrites the input dataset.
            if in_place and not first_real:
                assert path in h5
                continue
            h5.create_dataset(
                path,
                shape=shape,
                dtype=np.complex128,
                fillvalue=complex(np.nan, np.nan),
            )

    # Five rows with two rows per block exercises a partial last block.
    # Mask datasets are unnecessary because masking is disabled.
    iono.compute_differential_phase(
        phase_first=first_file,
        phase_second=second_file,
        output_path=output_file,
        first_data_path=first_data_paths,
        second_data_path=second_data_paths,
        output_data_path=output_data_paths,
        lines_per_block=2,
        subswath_mask_enabled=False,
        first_valid_mask_paths=first_valid_mask_paths,
        second_valid_mask_paths=second_valid_mask_paths,
    )

    with h5py.File(output_file, "r") as h5:
        for index, path in enumerate(output_data_paths):
            # Real phase is converted to unit-amplitude complex values.
            amplitude = 3.0 if first_real else 6.0
            expected = amplitude * np.exp(
                1j * (first_phase + index * 0.4 - second_phase)
            )
            actual = h5[path][:]
            assert actual.shape == shape
            assert np.all(np.isfinite(actual))
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-6,
                atol=1e-6,
            )


@pytest.mark.parametrize(
    "first_preferred,second_preferred",
    [(True, True), (True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize("mask_enabled", [False, True])
def test_differential_phase_masking(
    tmp_path, mask_decoders, first_preferred, second_preferred, mask_enabled
):
    """Check mask precedence, per-file fallback, and optional masking."""
    shape = (5, 7)

    # Test-only encoding supplied by the mask_decoders fixture:
    # 3: both valid; 2: reference invalid; 1: secondary invalid.
    first_codes = np.full(shape, 3, dtype=np.uint8)
    second_codes = first_codes.copy()
    first_codes[0, 0] = 2
    first_codes[1, 1] = 1
    second_codes[2, 2] = 2
    second_codes[4, 6] = 1  # Exercise the final partial block.

    swath_path = RIFGGroupsPaths().SwathsPath
    freq_path = f"{swath_path}/frequencyA"
    pol_path = f"{freq_path}/interferogram/HH"
    phase_path = f"{pol_path}/wrappedInterferogram"
    valid_mask_path = f"{pol_path}/validDataMask"
    fallback_mask_path = f"{freq_path}/interferogram/mask"

    first_file = str(tmp_path / "first_RIFG.h5")
    second_file = str(tmp_path / "second_RIFG.h5")
    output_file = str(tmp_path / "difference_RIFG.h5")

    for filename, phase, codes, preferred in (
        (first_file, 0.8, first_codes, first_preferred),
        (second_file, 0.3, second_codes, second_preferred),
    ):
        with h5py.File(filename, "w") as h5:
            h5.create_dataset(
                phase_path,
                data=np.full(shape, np.exp(1j * phase)),
            )
            # A conflicting fallback verifies that validDataMask wins.
            fallback_codes = np.zeros_like(codes) if preferred else codes
            h5.create_dataset(fallback_mask_path, data=fallback_codes)
            if preferred:
                h5.create_dataset(valid_mask_path, data=codes)

    with h5py.File(output_file, "w") as h5:
        h5.create_dataset(
            phase_path,
            shape=shape,
            dtype=np.complex128,
            fillvalue=complex(np.nan, np.nan),
        )

    fill_value = -999.0
    iono.compute_differential_phase(
        phase_first=first_file,
        phase_second=second_file,
        output_path=output_file,
        first_data_path=[phase_path],
        second_data_path=[phase_path],
        output_data_path=[phase_path],
        lines_per_block=2,
        subswath_mask_enabled=mask_enabled,
        first_mask_path=fallback_mask_path,
        second_mask_path=fallback_mask_path,
        invalid_fill_value=fill_value,
        first_valid_mask_paths=[valid_mask_path],
        second_valid_mask_paths=[valid_mask_path],
    )

    expected = np.full(shape, np.exp(0.5j))
    if mask_enabled:
        invalid = (first_codes != 3) | (second_codes != 3)
        expected[invalid] = fill_value

    with h5py.File(output_file, "r") as h5:
        actual = h5[phase_path][:]

    assert actual.shape == shape
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "yaml_name, method",
    [
        ("ionosphere_test.yaml", "split_main_band"),
        ("ionosphere_test.yaml", "main_diff_low_high_subband"),
        ("ionosphere_main_side_test.yaml", "main_side_band"),
        ("ionosphere_main_side_test.yaml", "main_diff_ms_band"),
    ],
)
def test_ionosphere_run(yaml_name, method, tmp_path, monkeypatch):
    """Check that each ionosphere method completes."""
    data_dir = os.path.abspath(iscetest.data)
    monkeypatch.chdir(tmp_path)

    yaml_path = os.path.join(data_dir, yaml_name)
    with open(yaml_path) as fh:
        test_yaml = (
            fh.read()
            .replace("@ISCETEST@", data_dir)
            .replace("@TEST_OUTPUT@", "RUNW.h5")
            .replace("@TEST_PRODUCT_TYPES@", "RUNW")
            .replace("@TEST_RDR2GEO_FLAGS@", "True")
            .replace(
                "spectral_diversity:",
                f"spectral_diversity: {method}",
            )
        )

    args = argparse.Namespace(
        run_config_path=test_yaml,
        log_file=False,
    )

    runconfig = InsarRunConfig(args)
    runconfig.geocode_common_arg_load()
    runconfig.yaml_check()

    _, out_paths = get_products_and_paths(runconfig.cfg)
    persist = Persistence(restart=True, logfile_path="ionosphere.log")

    # Disable CPU-unsupported offset steps and baseline metadata requirements.
    for step in (
        "dense_offsets",
        "rubbersheet",
        "fine_resample",
        "baseline",
    ):
        persist.run_steps[step] = False

    insar.run(runconfig.cfg, out_paths, persist.run_steps)
