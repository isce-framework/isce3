import isce3
import numpy as np
import numpy.polynomial as npp
import numpy.testing as npt


def test_constant():
    '''
    Test constant value of poly2D
    (adapted from Poly2d C++ unit test)
    '''

    ref_value = 10.0
    coeffs = np.full((1,1), fill_value=ref_value)

    # Interpolate N values in x and y
    for k in range(1, 5):
        poly2d = isce3.core.Poly2d(coeffs, k * 1.0, 0.0, k * k * 1.0, 1.0)
        value = poly2d.eval(0.0, k * 1.0)
        npt.assert_almost_equal(ref_value, value)


def test_mean_shift():
    '''
    Test Poly2d range mean shift
    (adapted from C++ unit test)
    '''

    # Create and initialize poly2D object
    coeffs = np.array([[0.0, 1.0, 0.0]])
    ref_poly2d = isce3.core.Poly2d(coeffs, 0.0, 0.0, 1.0, 1.0)

    # Compare reference and new range mean shift
    for k in range(0, 5):
        new_poly2d = isce3.core.Poly2d(coeffs, 0.5 * k * k, 0.0, 1.0, 1.0)

        # Compare reference and new value
        ref_val = ref_poly2d.eval(0.0, 2.0 * k)
        new_val = new_poly2d.eval(0.0, 2.0 * k + 0.5 * k * k)

        npt.assert_almost_equal(ref_val, new_val)


def test_norm_shift():
    '''
    Test Poly2d range std shift
    (Adapted from Poly2d C++ unit test)
    '''

    # Create and initialize Poly2d object
    coeffs = np.array([[0.0, 0.0, 1.0]])
    ref_poly2d = isce3.core.Poly2d(coeffs, 0.0, 0.0, 1.0, 1.0)

    # Compare reference and new range std shift
    for k in range(1, 6):
        new_poly2d = isce3.core.Poly2d(coeffs, 0.0, 0.0, k * k * 1.0, 1.0)

        ref_val = ref_poly2d.eval(0.0, 2.5)
        new_val = new_poly2d.eval(0.0, 2.5 * k * k)

        npt.assert_almost_equal(ref_val, new_val)

class common_params:
    # Prepare input values
    n_x = 5
    x_vec = np.arange(n_x)
    n_y = 4
    y_vec = np.arange(1, n_y + 1)
    az_order = 3
    rg_order = 2

    # Prepare coefficients for poly2d
    az_mesh, rg_mesh = np.meshgrid(np.arange(az_order + 1),
                                   np.arange(rg_order + 1))
    coeffs = np.array(az_mesh + rg_mesh, dtype=np.double)

    # Poly2d obj
    poly2d = isce3.core.Poly2d(coeffs)

def test_grid_eval():
    '''
    Test poly2D with mesh input
    '''

    common = common_params()

    # Prepare evaluate
    grid_eval_vals = common.poly2d.evalgrid(common.y_vec, common.x_vec)

    # Perform numpy poly eval over same grid
    # Transpose coeffs to account for numpy reversed indexing
    npp_out = npp.polynomial.polygrid2d(common.x_vec, common.y_vec, common.coeffs.transpose())

    # Check values of mesh eval against point eval to ensure pybind ordering correct
    npt.assert_array_almost_equal(npp_out, grid_eval_vals)

def test_1d_eval():
    '''
    Test Poly2D.eval() with 1-D array input
    '''

    common = common_params()

    # Prepare input values
    n = 10
    x = np.random.uniform(-5.0, 5.0, size=n)
    y = np.random.uniform(0.0, 10.0, size=n)

    # Prepare poly2d and evaluate
    out = common.poly2d.eval(y, x)

    # Perform numpy poly eval over same grid
    # Transpose coeffs to account for numpy reversed indexing
    c = common.poly2d.coeffs.T
    ref = npp.polynomial.polyval2d(x, y, c)

    # Check values of mesh eval against point eval to ensure pybind ordering correct
    npt.assert_array_almost_equal(out, ref)

def test_2d_eval():
    '''
    Test poly2D with mesh input
    '''

    common = common_params()

    # Prepare input values
    x_mesh, y_mesh = np.meshgrid(common.x_vec, common.y_vec)

    # Prepare poly2d and evaluate
    mesh_eval_vals = common.poly2d.eval(y_mesh, x_mesh)

    # Perform numpy poly eval over same grid
    # Transpose coeffs to account for numpy reversed indexing
    npp_out = npp.polynomial.polyval2d(x_mesh, y_mesh, common.coeffs.transpose())

    # Check values of mesh eval against point eval to ensure pybind ordering correct
    npt.assert_array_almost_equal(npp_out, mesh_eval_vals)

def test_coeffs():
    '''
    Ensure coeffs match
    '''
    common = common_params()
    npt.assert_array_equal(common.poly2d.coeffs, common.coeffs)
