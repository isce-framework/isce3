from glob import glob
import isce3
import iscetest
from nisar.workflows import focus
from nisar.workflows.point_target_analysis import slc_pt_performance
import nisar
from pathlib import Path
import numpy as np
import numpy.testing as npt
import os


def get_test_cfg():
    "Load runconfig and update input file names."
    def locate(fname: str) -> str:
        return str(Path(iscetest.data) / "focus" / fname)
    cfg = focus.load_config(locate("runconfig.yaml"))
    cfg.runconfig.groups.input_file_group.input_file_path = [locate("REE_L0B_out17.h5")]
    aux = cfg.runconfig.groups.dynamic_ancillary_file_group
    aux.orbit = locate("orbit.xml")
    aux.pointing = locate("attitude.xml")
    aux.internal_calibration = locate("REE_INSTRUMENT_TABLE.h5")
    aux.antenna_pattern = locate("REE_ANTPAT_CUTS_DATA.h5")
    return cfg


def slc_is_baseband(filename: str, tol=2*np.pi/100, frequency="A", polarization="HH") -> bool:
    rslc = nisar.products.readers.SLC(hdf5file=filename)
    ds = rslc.getSlcDataset(frequency, polarization)
    # work around h5py/numpy awkwardness with Complex{Float16}
    z = isce3.core.types.read_c4_dataset_as_c8(ds)
    dz = z[:, 1:] * z[:, :-1].conj()
    return abs(np.angle(dz.sum())) < tol


def test_focus():
    cfg = get_test_cfg()
    focus.focus(cfg)
    filename = cfg.runconfig.groups.product_path_group.sas_output_file

    assert slc_is_baseband(filename)

    # Check that target shows up where expected.
    llh = [-54.579586258, 3.177088785, 0.0]  # units (deg, deg, m)
    cr = slc_pt_performance(filename, "A", "HH", llh, predict_null=False,
                            nov=128)
    # NISAR requirement is 10% absolute, 1/128 relative.
    assert abs(cr["range"]["offset"]) <= 1 / 128.
    assert abs(cr["azimuth"]["offset"]) <= 1 / 128.

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


def test_schema():
    # Check that workflow test config files conform to schema.
    # Note that schema mostly reflects needs of PCM/PGE, and the SAS is a bit
    # more flexible.
    import yamale  # delayed import since other test doesn't need yamale
    src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        *(os.pardir,) * 5)
    schema = yamale.make_schema(
        os.path.join(src_dir, "share", "nisar", "schemas", "focus.yaml"),
        parser="ruamel")
    workflow_test_dir = os.path.join(src_dir,
        "tools", "imagesets", "runconfigs")
    config_files = glob(os.path.join(workflow_test_dir, "*rslc*.yaml"))
    for filename in config_files:
        data = yamale.make_data(filename, parser="ruamel")
        yamale.validate(schema, data)
