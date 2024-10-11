import iscetest
import nisar
from nisar.products.insar.granule_id import get_insar_granule_id
from nisar.products.writers.BaseWriterSingleInput import (
    get_granule_id_single_input)
from nisar.products.writers.SLC import fill_partial_granule_id
from pathlib import Path
import pytest
import shapely


@pytest.fixture
def rslc_filename():
    return str(Path(iscetest.data) / "winnipeg.h5")


@pytest.fixture
def raw_filename():
    return str(Path(iscetest.data) / "REE_L0B_out17.h5")


def test_insar(rslc_filename):
    ref = sec = rslc_filename

    partial_granule_id = ("NISAR_{Level}_PR_{ProductType}_034_080_A_010_{MODE}"
        "_{PO}_A_{RefStartDateTime}_{RefEndDateTime}"
        "_{SecStartDateTime}_{SecEndDateTime}_D00340_P_P_J_001")

    expected_granule_id = ("NISAR_L1_PR_RIFG_034_080_A_010_2000_SH_A"
        "_20120717T143647_20120717T144244_20120717T143647_20120717T144244"
        "_D00340_P_P_J_001")

    granule_id = get_insar_granule_id(ref, sec, partial_granule_id, ["HH"])

    assert granule_id == expected_granule_id


def test_gslc_gcov(rslc_filename):
    reader = nisar.products.readers.RSLC(hdf5file=rslc_filename)

    partial_granule_id = ("NISAR_L2_PR_GSLC_001_001_A_004_{MODE}_{POLE}_A"
        "_{StartDateTime}_{EndDateTime}_D00340_P_P_J_001")

    expected_granule_id = ("NISAR_L2_PR_GSLC_001_001_A_004_2000_SHNA_A"
        "_20120717T143647_20120717T144244_D00340_P_P_J_001")

    granule_id = get_granule_id_single_input(reader, partial_granule_id,
        {"A": ["HH"]})

    assert granule_id == expected_granule_id


@pytest.mark.parametrize("frame_wkt,expected_granule_id", [
    # unknown coverage (X) when frame polygon not provided/empty
    ("POLYGON EMPTY",
        "NISAR_L1_PR_RSLC_001_001_A_004_2000_SHNA_A_20210701T032000_20210701T032006_D00340_P_X_J_001"),
    # partial coverage (P) for < 75% overlap
    ("POLYGON ((-54.775 3.410, ""-54.413 3.410, -54.413 2.938, -54.775 2.938, -54.775 3.410))",
        "NISAR_L1_PR_RSLC_001_001_A_004_2000_SHNA_A_20210701T032000_20210701T032006_D00340_P_P_J_001"),
    # full coverage (F) for >= 75% coverage
    ("POLYGON ((-54.7 3.410, -54.5 3.410, -54.5 2.938, -54.7 2.938, -54.7 3.410))",
        "NISAR_L1_PR_RSLC_001_001_A_004_2000_SHNA_A_20210701T032000_20210701T032006_D00340_P_F_J_001"),
])
def test_rslc(raw_filename, frame_wkt, expected_granule_id):
    raw = nisar.products.readers.Raw.open_rrsd(raw_filename)
    mode = nisar.mixed_mode.PolChannelSet.from_raw(raw)
    t0 = raw.identification.zdStartTime
    t1 = raw.identification.zdEndTime
    frame_polygon = shapely.from_wkt(frame_wkt)
    image_polygon = shapely.from_wkt(raw.identification.boundingPolygon)

    partial_granule_id = ("NISAR_L1_PR_RSLC_001_001_A_004_{MODE}_{POLE}_A"
        "_{StartDateTime}_{EndDateTime}_D00340_P_{C}_J_001")

    granule_id = fill_partial_granule_id(partial_granule_id, mode, t0, t1,
        frame_polygon, image_polygon)

    assert granule_id == expected_granule_id
