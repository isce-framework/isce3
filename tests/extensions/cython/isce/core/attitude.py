#!/usr/bin/env python3

def test_CythonInterface():
    from isceextension import pyEulerAngles
    import numpy as np
    import numpy.testing as npt

    attitude = pyEulerAngles(yaw=0.1, pitch=0.05, roll=-0.1)

    # Define the reference rotation matrix (YPR)
    R_ypr_ref = np.array([[0.993760669166, -0.104299329454, 0.039514330251],
                [0.099708650872, 0.989535160981, 0.104299329454],
                [-0.049979169271, -0.099708650872, 0.993760669166]])

    # Define the reference rotation matrix (RPY)
    R_rpy_ref = np.array([[0.993760669166, -0.099708650872, 0.049979169271],
                [0.094370001341, 0.990531416861, 0.099708650872],
                [-0.059447752410, -0.094370001341, 0.993760669166]])

    R_ypr = attitude.rotmat('ypr')
    npt.assert_array_almost_equal(R_ypr_ref, R_ypr, decimal=10, err_msg="YPR rotation")

    R_rpy = attitude.rotmat('rpy')
    npt.assert_array_almost_equal(R_rpy_ref, R_rpy, decimal=10, err_msg='RPY rotation')
