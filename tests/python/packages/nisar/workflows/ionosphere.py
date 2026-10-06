"""Unit tests and integration smoke tests for the ionosphere workflow.

Run with pytest in an ISCE3 environment with iscetest data available.
Mask decoding is mocked to test workflow logic, not NISAR bit definitions.
"""

import argparse
import os

import pytest

import iscetest
from nisar.workflows import insar
from nisar.workflows.h5_prep import get_products_and_paths
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence


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
    """Check that each ionosphere method completes.
    tmp_path and monkeypatch are built-in pytest fixtures."""
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
