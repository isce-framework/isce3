import h5py
import numpy as np
from pybind_isce3 import core
from tempfile import mkstemp

def test_attitude():
    n = 11
    t = np.linspace(0, 1, n)
    epoch = core.DateTime(2023, 1, 1)

    # All quaternions no rotation.
    q = [core.Quaternion(1, 0, 0, 0) for i in range(n)]
    attitude = core.Attitude(t, q, epoch)

    def rotmat(t):
        return attitude.interpolate(t).to_rotation_matrix()

    ti = 0.5
    assert np.allclose(rotmat(ti), np.eye(3))

    # Make sure we don't crash at endpoints.
    assert np.allclose(rotmat(t[0]), np.eye(3))
    assert np.allclose(rotmat(t[-1]), np.eye(3))

    # Varying rotation around constant axis.  No rotation at time ti.
    np.random.seed(1234)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angles = np.linspace(-1, 1, n)
    w = np.cos(angles / 2)
    xyz = np.sin(angles / 2)[:,None] * axis[None,:]
    q = [core.Quaternion(w[i], *xyz[i]) for i in range(n)]

    attitude = core.Attitude(t, q, epoch)
    assert np.allclose(rotmat(ti), np.eye(3))

    # Check that rotation axis remains invariant with interpolation.
    ni = 1 + 32 * (n - 1)
    for x in np.linspace(t[0], t[-1], ni):
        R = rotmat(x)
        assert np.allclose(R.dot(axis), axis)


def dummy_attitude():
    t = [0.0, 0.1]
    q = (core.Quaternion(1,0,0,0),) * 2
    epoch = core.DateTime(2020, 1, 1)
    return core.Attitude(t, q, epoch)


def test_properties_exist():
    attitude = dummy_attitude()
    attitude.size
    attitude.time
    attitude.quaternions
    attitude.start_time
    attitude.end_time
    attitude.start_datetime
    attitude.end_datetime
    attitude.reference_epoch


def test_io():
    attitude = dummy_attitude()
    _, name = mkstemp()
    # write
    with h5py.File(name, "w") as h5:
        g = h5.create_group("/attitude")
        attitude.save_to_h5(g)
    # read
    with h5py.File(name, "r") as h5:
        core.Attitude.load_from_h5(h5["/attitude"])

