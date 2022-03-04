import os

import numpy.testing as npt

import isce3


def test_gpu_check():
    if hasattr(isce3, "cuda"):
        gpu_enabled = True
        gpu_id = -1
        with npt.assert_raises(ValueError):
            isce3.core.gpu_check.use_gpu(gpu_enabled, gpu_id)
