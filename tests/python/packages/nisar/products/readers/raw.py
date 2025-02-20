import h5py
import isce3
import iscetest
import numpy as np
import numpy.testing as npt
from pathlib import Path
import nisar
from tempfile import NamedTemporaryFile


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
    num_rgl = t[1].size
    side = raw.identification.lookDirection
    fc = raw.getCenterFrequency(frequency=freq, tx=tx)
    t, grid = raw.getRadarGrid(frequency=freq, tx=tx)
    ds = raw.getRawDataset(freq, pol)
    swaths = raw.getSubSwaths(frequency=freq, tx=tx)
    prf = raw.getNominalPRF(freq, tx)
    bandwidth = raw.getRangeBandwidth(freq, tx)
    caltone = raw.getCaltone(freq, pol)
    dm_flag = raw.identification.diagnosticModeFlag
    rd, wd, wl = raw.getRdWdWl(freq, pol)
    list_rx_trm = raw.getListOfRxTRMs(freq, pol)
    is_tx_off = raw.is_tx_off(freq, pol)

    # Verify assumptions.
    npt.assert_equal(orbit.reference_epoch, attitude.reference_epoch)
    npt.assert_equal(side, "Right")
    npt.assert_equal(raw.isDithered(freq, tx), False)
    npt.assert_equal(raw.sarBand, 'L')
    npt.assert_equal(ds.ndim, 2)
    npt.assert_equal(ds.dtype, np.complex64)
    print(f'Datatype of raw dataset before decoding -> {ds.dtype_storage}')
    npt.assert_equal(dm_flag, 0,
                     err_msg='Bad value for "diagnosticModeFlag"!')
    print(f'Diagnostic Mode Name -> "{raw.identification.diagnosticModeName}"')
    # check shape/size of RD/WD/WL
    rd_shape = rd.shape
    npt.assert_((rd_shape==wd.shape) and (rd_shape==wl.shape),
                msg='Shape mismatch between RD, WD, and WL arrays')
    npt.assert_equal(rd_shape, (num_rgl, list_rx_trm.size),
                     err_msg='Wrong shape for RD/WD/WL!')
    # check whether TX is off
    npt.assert_equal(
        is_tx_off, False,
        err_msg=f'TX is not off for frequency {freq} and pol {pol}!'
    )

    # Check quaternion convention.
    # RCS frame has Y-axis nearly parallel to velocity (for small rotations).
    # In this case they should be exactly aligned since roll == pitch == 0.
    # When right-looking, the projection onto velocity is negative.
    tref = (t[0] + t[-1]) / 2
    x, v = orbit.interpolate(tref)
    R = attitude.interpolate(tref).to_rotation_matrix()
    elevation_axis = R[:,1]
    npt.assert_allclose(v.dot(elevation_axis) / np.linalg.norm(v), -1.0)


def make_test_decoder_file(filename: str) -> np.ndarray:
    """
    Creates an HDF5 file with the given name that contains the datasets

    /complex32/z
    /complex64/z
    /bfpq/BFPQLUT
    /bfpq/z

    Returns the values expected to be decoded from each one.
    """
    shape = (20, 10)
    values = np.arange(np.prod(shape)).reshape(shape)

    with h5py.File(filename, mode="w") as h5:
        g = h5.create_group("complex32")
        z = np.zeros(shape, dtype=isce3.core.types.complex32)
        z['r'][:] = values
        g.create_dataset("z", data=z)

        g = h5.create_group("complex64")
        z = np.zeros(shape, dtype=np.complex64)
        z.real[:] = values
        g.create_dataset("z", data=z)

        g = h5.create_group("bfpq")
        lut = np.arange(2**16, dtype=np.float32)
        z = np.zeros(shape, dtype=np.dtype([('r', 'u2'), ('i', 'u2')]))
        z['r'][:] = values
        g.create_dataset("BFPQLUT", data=lut)
        g.create_dataset("z", data=z)

    return values.astype(np.complex64)


def test_decoder():
    f = NamedTemporaryFile(suffix=".h5")
    expected = make_test_decoder_file(f.name)
    with h5py.File(f.name, mode="r") as h5:
        decoder = nisar.products.readers.Raw.DataDecoder(h5["bfpq/z"])
        npt.assert_equal(decoder[:,:], expected)
        npt.assert_(decoder.dtype_storage == np.dtype([('r', 'u2'), ('i', 'u2')]))

        decoder = nisar.products.readers.Raw.DataDecoder(h5["complex64/z"])
        npt.assert_equal(decoder[:,:], expected)

        decoder = nisar.products.readers.Raw.DataDecoder(h5["complex32/z"])
        npt.assert_equal(decoder[:,:], expected)
