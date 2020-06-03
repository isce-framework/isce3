import numpy as np
from pybind_isce3 import core

def test_quaternion():
    n = 11
    t = np.linspace(0, 1, n)
    q = np.zeros((n, 4))

    # All quaternions no rotation.
    q[:,0] = 1.0
    attitude = core.Quaternion(t, q)

    ti = 0.5
    R = attitude.rotmat(ti)
    assert np.allclose(R, np.eye(3))

    # Make sure we don't crash at endpoints.
    assert np.allclose(attitude.rotmat(t[0]), np.eye(3))
    assert np.allclose(attitude.rotmat(t[-1]), np.eye(3))

    # Varying rotation around constant axis.  No rotation at time ti.
    np.random.seed(1234)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angles = np.linspace(-1, 1, n)
    q[:,0] = np.cos(angles / 2)
    q[:,1:] = np.sin(angles / 2)[:,None] * axis[None,:]

    attitude = core.Quaternion(t, q)
    R = attitude.rotmat(ti)
    assert np.allclose(R, np.eye(3))

    # Check that rotation axis remains invariant with interpolation.
    ni = 1 + 32 * (n - 1)
    for x in np.linspace(t[0], t[-1], ni):
        R = attitude.rotmat(x)
        assert np.allclose(R.dot(axis), axis)
