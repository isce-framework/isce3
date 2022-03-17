import iscetest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import nisar


def test_raw():
    fn = Path(iscetest.data) / "REE_L0B_out17.h5"
    raw = nisar.products.readers.Raw.open_rrsd(fn)

    npt.assert_equal(raw.frequencies, ["A"])
    npt.assert_equal(raw.polarizations, {"A": ["HH"]})

    freq = raw.frequencies[0]
    pol = raw.polarizations[freq][0]
    tx = pol[0]

    # Basically just call the methods to make sure they don't crash.
    chirp = raw.getChirp(frequency=freq, tx=tx)
    npt.assert_equal(len(chirp), 481)
    orbit = raw.getOrbit()
    attitude = raw.getAttitude()
    r = raw.getRanges(frequency=freq, tx=tx)
    t = raw.getPulseTimes(frequency=freq, tx=tx)
    side = raw.identification.lookDirection
    fc = raw.getCenterFrequency(frequency=freq, tx=tx)
    t, grid = raw.getRadarGrid(frequency=freq, tx=tx)
    ds = raw.getRawDataset(freq, pol)
    swaths = raw.getSubSwaths(frequency=freq, tx=tx)
    prf = raw.getNominalPRF(freq, tx)
    bandwidth = raw.getRangeBandwidth(freq, tx)

    # Verify assumptions.
    npt.assert_equal(orbit.reference_epoch, attitude.reference_epoch)
    npt.assert_equal(side, "right")
    npt.assert_equal(raw.isDithered(freq, tx), False)
    npt.assert_equal(raw.sarBand, 'L')
    npt.assert_equal(ds.ndim, 2)
    npt.assert_equal(ds.dtype, np.complex64)
    print(f'Datatype of raw dataset before decoding -> {ds.dtype_storage}')
    
    # Check quaternion convention.
    # RCS frame has Y-axis nearly parallel to velocity (for small rotations).
    # In this case they should be exactly aligned since roll == pitch == 0.
    # When right-looking, the projection onto velocity is negative.
    tref = (t[0] + t[-1]) / 2
    x, v = orbit.interpolate(tref)
    R = attitude.interpolate(tref).to_rotation_matrix()
    elevation_axis = R[:,1]
    npt.assert_allclose(v.dot(elevation_axis) / np.linalg.norm(v), -1.0)
