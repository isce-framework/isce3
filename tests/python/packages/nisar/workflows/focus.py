import iscetest
from nisar.workflows import focus
import nisar
from pathlib import Path
import numpy as np
import numpy.testing as npt


def get_test_cfg():
    "Load runconfig and update input file names."
    def locate(fname: str) -> str:
        return str(Path(iscetest.data) / "focus" / fname)
    cfg = focus.load_config(locate("runconfig.yaml"))
    cfg.runconfig.groups.input_file_group.input_file_path = [locate("REE_L0B_out17.h5")]
    cfg.runconfig.groups.dynamic_ancillary_file_group.orbit = locate("orbit.xml")
    cfg.runconfig.groups.dynamic_ancillary_file_group.pointing = locate("attitude.xml")
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
    filename = cfg.runconfig.groups.product_path_group.sas_output_file

    assert slc_is_baseband(filename)

    # Check that epochs are consistent.
    rslc = nisar.products.readers.SLC(hdf5file=filename)
    orbit = rslc.getOrbit()
    grid = rslc.getRadarGrid()
    assert orbit.reference_epoch == grid.ref_epoch
    # Currently there's no isce3 API to get RSLC attitude data, so don't check
    # its epoch.  Similarly it's not possible to get the time epoch associated
    # with the Doppler LUT (though this info is in the RSLC metadata), but we
    # can at least check that its time domain overlaps with the orbit.
    carrier = rslc.getDopplerCentroid()
    assert orbit.contains(carrier.y_start)
