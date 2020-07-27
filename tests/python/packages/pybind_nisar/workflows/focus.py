import iscetest
from pybind_nisar.workflows import focus
import os

def test_focus():
    # Don't just call main() because need to update raw data file name.
    rawname = os.path.join(iscetest.data, "REE_L0B_out17.h5")
    cfgname = rawname.replace(".h5", ".yaml")
    cfg = focus.load_config(cfgname)
    cfg.runconfig.groups.InputFileGroup.InputFilePath = [rawname]
    focus.focus(cfg)
