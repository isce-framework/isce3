import iscetest
from pybind_nisar.workflows import focus
import os
import numpy.testing as npt

def get_test_cfg():
    # Don't just call main() because need to update raw data file name.
    rawname = os.path.join(iscetest.data, "REE_L0B_out17.h5")
    cfgname = rawname.replace(".h5", ".yaml")
    cfg = focus.load_config(cfgname)
    cfg.runconfig.groups.InputFileGroup.InputFilePath = [rawname]
    return cfg

def test_focus():
    cfg = get_test_cfg()
    focus.focus(cfg)

def test_invalid_device():
    import pybind_isce3 as isce3

    if hasattr(isce3, "cuda"):
        cfg = get_test_cfg()
        cfg.runconfig.groups.worker.gpu_enabled = True
        cfg.runconfig.groups.worker.gpu_id = -1
        with npt.assert_raises(ValueError):
            focus.focus(cfg)
