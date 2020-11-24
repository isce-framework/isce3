import os

import numpy.testing as npt

import pybind_isce3 as isce3
from pybind_nisar.workflows import gpu_check


def test_gpu_check():
    if hasattr(isce3, "cuda"):
        gpu_enabled = True
        gpu_id = -1
        with npt.assert_raises(ValueError):
            gpu_check.use_gpu(gpu_enabled, gpu_id)
