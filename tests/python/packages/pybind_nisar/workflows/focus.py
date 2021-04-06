import iscetest
from pybind_nisar.workflows import focus
import pybind_nisar as nisar
import os
import numpy as np
import numpy.testing as npt


def get_test_cfg():
    # Don't just call main() because need to update raw data file name.
    rawname = os.path.join(iscetest.data, "REE_L0B_out17.h5")
    cfgname = rawname.replace(".h5", ".yaml")
    cfg = focus.load_config(cfgname)
    cfg.runconfig.groups.InputFileGroup.InputFilePath = [rawname]
    return cfg


def slc_is_baseband(filename: str, tol=2*np.pi/100, frequency="A", polarization="HH") -> bool:
    rslc = nisar.products.readers.SLC(hdf5file=filename)
    ds = rslc.getSlcDataset(frequency, polarization)
    # work around h5py/numpy awkwardness with Complex{Float16}
    z = nisar.types.read_c4_dataset_as_c8(ds)
    dz = z[:, 1:] * z[:, :-1].conj()
    return abs(np.angle(dz.sum())) < tol


def test_focus():
    cfg = get_test_cfg()
    focus.focus(cfg)
    assert slc_is_baseband(cfg.runconfig.groups.ProductPathGroup.SASOutputFile)
