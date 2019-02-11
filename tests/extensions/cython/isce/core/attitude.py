#!/usr/bin/env python3

def test_CythonInterface():
    from isce3.extensions.isceextension import pyEulerAngles
    import numpy as np
    import numpy.testing as npt

    time = np.linspace(0.0, 10.0, 20)
    yaw = np.full(time.shape, 0.1)
    pitch = np.full(time.shape, 0.05)
    roll = np.full(time.shape, -0.1)

    attitude = pyEulerAngles(time=time, yaw=yaw, pitch=pitch, roll=roll)

    # Define the reference rotation matrix (YPR)
    R_ypr_ref = np.array([[0.993760669166, -0.104299329454, 0.039514330251],
                [0.099708650872, 0.989535160981, 0.104299329454],
                [-0.049979169271, -0.099708650872, 0.993760669166]])

    # Define the reference rotation matrix (RPY)
    R_rpy_ref = np.array([[0.993760669166, -0.099708650872, 0.049979169271],
                [0.094370001341, 0.990531416861, 0.099708650872],
                [-0.059447752410, -0.094370001341, 0.993760669166]])

    R_ypr = attitude.rotmat(5.0, 'ypr')
    npt.assert_array_almost_equal(R_ypr_ref, R_ypr, decimal=10, err_msg="YPR rotation")

    R_rpy = attitude.rotmat(5.0, 'rpy')
    npt.assert_array_almost_equal(R_rpy_ref, R_rpy, decimal=10, err_msg='RPY rotation')
