import iscetest
from nisar.mixed_mode import (Band, null_band, PolChannel, PolChannelSet,
    find_overlapping_channel)
from nisar.products.readers.Raw import Raw
import numpy.testing as npt
from pathlib import Path
import pytest


L05 = Band(1221.5e6, 5.0e6) # mode 134
L20 = Band(1229e6, 20e6)    # mode 128
L40 = Band(1239e6, 40e6)    # mode 137
L80 = Band(1257.5e6, 77e6)  # mode 131
AUX = Band(1293.5e6, 5e6)   # mode 128, upper band
QQ5 = Band(1236.5e6, 5.0e6) # mode 146, upper band
NUL = null_band()

# from mode table v47
modes = {
    # background land, DP HH/HV
    128: PolChannelSet([
        PolChannel("A", "HH", L20),
        PolChannel("A", "HV", L20),
        PolChannel("B", "HH", AUX),
        PolChannel("B", "HV", AUX),
    ]),
    # background land soil moisture, QQ
    129: PolChannelSet([
        PolChannel("A", "HH", L20),
        PolChannel("A", "HV", L20),
        PolChannel("B", "VH", AUX),
        PolChannel("B", "VV", AUX),
    ]),
    # land ice, SP HH
    131: PolChannelSet([
        PolChannel("A", "HH", L80),
    ]),
    # sea ice, SP VV
    134: PolChannelSet([
        PolChannel("B", "VV", L05),
    ]),
    # urban areas & Himalayas, DP HH/HV
    137: PolChannelSet([
        PolChannel("A", "HH", L40),
        PolChannel("A", "HV", L40),
        PolChannel("B", "HH", AUX),
        PolChannel("B", "HV", AUX),
    ]),
    # agriculture, QP
    140: PolChannelSet([
        PolChannel("A", "HH", L40),
        PolChannel("A", "HV", L40),
        PolChannel("A", "VH", L40),
        PolChannel("A", "VV", L40),
        PolChannel("B", "HH", AUX),
        PolChannel("B", "HV", AUX),
        PolChannel("B", "VH", AUX),
        PolChannel("B", "VV", AUX),
    ]),
    # ISRO sea ice alternate (oddball mode), QQ 5+5
    # Note that L0A records both on freq A, must be split in L0B
    146: PolChannelSet([
        PolChannel("A", "HH", L05),
        PolChannel("A", "HV", L05),
        PolChannel("B", "VH", QQ5),
        PolChannel("B", "VV", QQ5),
    ]),
}


def test_band_intersections():
    # Modes are designed to nest.
    modes = (L80, L40, L20, L05)
    for i in range(len(modes)):
        mi = modes[i]
        # Intersection with null band should be null.
        npt.assert_(mi & NUL == NUL)
        # Intersection with self and any smaller mode should be the smaller one.
        for j in range(i, len(modes)):
            mj = modes[j]
            npt.assert_(mi & mj == mj)
    # Only L80 overlaps with AUX band.
    npt.assert_(L80 & AUX == AUX)
    for mode in (L40, L20, L05):
        npt.assert_(mode & AUX == NUL)


def test_band_isvalid():
    npt.assert_(NUL.isvalid == False)
    for mode in (L80, L40, L20, L05):
        npt.assert_(mode.isvalid)
    npt.assert_(L80.intersection(L05).isvalid)


def test_pol_channel_intersection():
    AHH40 = PolChannel("A", "HH", L40)
    AHH20 = PolChannel("A", "HH", L20)
    AHV20 = PolChannel("A", "HV", L20)
    BHH20 = PolChannel("B", "HH", L20)
    BHH = PolChannel("B", "HH", AUX)
    BHV = PolChannel("B", "HV", AUX)

    npt.assert_((AHH40 & AHH40) == AHH40)       # identity
    npt.assert_((AHH40 & AHH20) == AHH20)       # 40 & 20 -> 20
    npt.assert_((AHH20 & BHH20) == AHH20)       # A & B -> A
    npt.assert_(not (AHH20 & AHV20).isvalid)    # HH & HV -> null
    npt.assert_(not (BHH & BHV).isvalid)        # HH & HV -> null
    npt.assert_(not (AHH20 & BHH).isvalid)      # 20 & AUX -> null


def get_raw():
    # This L0B file is single-pol HH 20 MHz
    fn = str(Path(iscetest.data) / "REE_L0B_out17.h5")
    return Raw(hdf5file=fn)


def test_pol_channel_search():
    raw = get_raw()
    desired = PolChannel("A", "HH", L80)
    available = find_overlapping_channel(raw, desired)
    npt.assert_(desired & available)
    # Should match regardless of freq_id, but returned freq_id should match
    # what is in the data file.
    desired = PolChannel("B", "HH", L80)
    available = find_overlapping_channel(raw, desired)
    npt.assert_(desired & available)
    npt.assert_(available.freq_id == "A")
    # No VV data
    desired = PolChannel("A", "VV", L80)
    with pytest.raises(ValueError):
        find_overlapping_channel(raw, desired)


def test_channel_set():
    raw = get_raw()
    mode1 = PolChannelSet.from_raw(raw)

    # open again and check intersection
    mode2 = PolChannelSet.from_raw(raw)
    npt.assert_(mode1 == mode2)
    npt.assert_((mode1 & mode2) == mode1)
    # add a band to mode2 that's not in mode1
    mode2.add(PolChannel("A", "HV", L20))
    # now should drop those bands when intersected with mode1
    common = mode1 & mode2
    npt.assert_(common == mode1)
    npt.assert_(common != mode2)

    # check some actual mode intersections
    # 20+5 & 40+5 == 20+5
    npt.assert_(modes[128] & modes[137] == modes[128])
    npt.assert_(modes[137] & modes[128] == modes[128])
    # HH 80 & DP 20+5 == HH 20+5
    npt.assert_(modes[128] & modes[131] == PolChannelSet([
        PolChannel("A", "HH", L20), PolChannel("B", "HH", AUX)
    ]))
    # SP VV & DP HH/HV == empty
    npt.assert_(len(modes[128] & modes[134]) == 0)
    # DP & QP == DP
    npt.assert_(modes[128] & modes[140] == modes[128])
    # QQ & QP == QQ
    npt.assert_(modes[129] & modes[140] == modes[129])
    # oddball
    npt.assert_(modes[128] & modes[146] == PolChannelSet([
        PolChannel("A", "HH", L05), PolChannel("A", "HV", L05)
    ]))


def test_quasi_dual_5p5():
    mode = PolChannelSet([
        PolChannel("B", "HH", L05),
        PolChannel("B", "VV", QQ5)
    ])
    # verify lower band gets relabeled as "A"
    npt.assert_(mode.regularized() == PolChannelSet([
        PolChannel("A", "HH", L05),
        PolChannel("B", "VV", QQ5)
    ]))
