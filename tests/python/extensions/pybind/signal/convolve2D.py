import numpy as np
import numpy.testing as npt

import isce3.ext.isce3 as isce3

def get_dims():
    '''
    Return data and kernel dimensions
    '''
    length = 200
    width = 310
    kernel_length = 3
    kernel_width = 3
    return length, width, kernel_length, kernel_width

def make_inputs(length, width, kernel_length, kernel_width):
    '''
    Create real and imag data
    '''
    # Calculate padding
    pad_cols = kernel_width - 1
    pad_rows = kernel_length - 1
    width_pad = width + pad_cols
    length_pad = length + pad_rows

    # Populate real and imag data
    data_real = np.zeros([length_pad, width_pad], dtype=np.float64)
    data_imag = np.zeros([length_pad, width_pad], dtype=np.complex128)
    for line in range(pad_rows//2, length + pad_rows//2):
        for col in range(pad_cols//2, width + pad_cols//2):
            data_real[line, col] = line + col
            data_imag[line, col] = np.cos(line * col) + np.sin(line * col) * 1.0j

    return data_real, data_imag

def make_expected_output(out_shape, data, kernel_width, kernel_length):
    '''
    Calculate expected decimated output
    '''
    decimated = np.zeros(out_shape, dtype=data.dtype)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            decimated[i,j] = np.mean(data[i*kernel_width+1:(i+1)*kernel_width+1,\
                                          j*kernel_length+1:(j+1)*kernel_length+1])
    return decimated

def test_ea_convolve2d_with_mask():
    '''
    Test convolve2D without mask
    '''
    length, width, kernel_length, kernel_width = get_dims()

    # Create data
    input_real, input_imag = make_inputs(length, width, kernel_length, kernel_width)

    # Create mask
    mask = np.ones(input_real.shape, dtype=np.float64)

    # Create kernels
    kernel_cols = np.ones([1, kernel_width], dtype=np.float64)/kernel_width
    kernel_rows = np.ones([kernel_length, 1], dtype=np.float64)/kernel_length

    # Convolve
    pybind_real = isce3.signal.convolve2D(input_real, mask, kernel_cols, kernel_rows, True)
    pybind_imag = isce3.signal.convolve2D(input_imag, mask, kernel_cols, kernel_rows, True)

    # Calculate expected output
    out_shape = (length//kernel_width, width//kernel_length)
    expected_real = make_expected_output(out_shape, input_real,
                                         kernel_length, kernel_width)
    expected_imag = make_expected_output(out_shape, input_imag,
                                         kernel_length, kernel_width)

    # Check outputs
    npt.assert_allclose(pybind_real, expected_real, rtol=0.0, atol=1e-12)
    npt.assert_allclose(np.angle(pybind_imag), np.angle(expected_imag),
                        rtol=0.0, atol=1e-12)

def test_ea_convolve2d_no_mask():
    '''
    Test convolve2D without mask
    '''
    length, width, kernel_length, kernel_width = get_dims()

    # Create data
    input_real, input_imag = make_inputs(length, width, kernel_length, kernel_width)

    # Create kernels
    kernel_cols = np.ones([1, kernel_width], dtype=np.float64)/kernel_width
    kernel_rows = np.ones([kernel_length, 1], dtype=np.float64)/kernel_length

    # Convolve
    pybind_real = isce3.signal.convolve2D(input_real, kernel_cols, kernel_rows, True)
    pybind_imag = isce3.signal.convolve2D(input_imag, kernel_cols, kernel_rows, True)

    # Calculate expected output
    out_shape = (length//kernel_width, width//kernel_length)
    expected_real = make_expected_output(out_shape, input_real,
                                         kernel_length, kernel_width)
    expected_imag = make_expected_output(out_shape, input_imag,
                                         kernel_length, kernel_width)

    # Check outputs
    npt.assert_allclose(pybind_real, expected_real, rtol=0.0, atol=1e-12)
    npt.assert_allclose(np.real(pybind_imag), np.real(expected_imag),
                        rtol=0.0, atol=1e-12)
    npt.assert_allclose(np.imag(pybind_imag), np.imag(expected_imag),
                        rtol=0.0, atol=1e-12)

def test_scipy():

    from scipy import signal

    data = np.ones((8,13))
    kernel = np.ones((3,3))/9.0

    filt_data_scipy = signal.convolve2d(data, kernel, mode='same')

    data_padded = np.ones((10,15))
    data_padded[0,:] = 0
    data_padded[:,0] = 0
    data_padded[:,-1] = 0
    data_padded[-1,:] = 0

    kernel_cols = np.ones((1,3))/3.0
    kernel_rows = np.ones((3,1))/3.0
    filt_data_isce3 = isce3.signal.convolve2D(data_padded, kernel_cols, kernel_rows, False)

    npt.assert_allclose(filt_data_isce3, filt_data_scipy, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    test_ea_convolve2d_no_mask()
    test_ea_convolve2d_with_mask()
