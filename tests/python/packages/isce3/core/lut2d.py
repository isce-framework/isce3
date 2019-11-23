def test_LUT2d():
    from isce3.extensions.isceextension import pyLUT2d
    import numpy as np
    import numpy.testing as npt
    vx = np.array([-2, -1.])
    vy = np.array([0,
                   1.])
    mz = np.array([[1, 2],
                   [3, 4.]])
    lut = pyLUT2d(x=vx, y=vy, z=mz)
    # eval(y,x)
    npt.assert_almost_equal(lut(0, -2), 1)
    npt.assert_almost_equal(lut(0, -1), 2)
    npt.assert_almost_equal(lut(1, -2), 3)
    npt.assert_almost_equal(lut(1, -1), 4)


if __name__ == '__main__':
    test_LUT2d()
