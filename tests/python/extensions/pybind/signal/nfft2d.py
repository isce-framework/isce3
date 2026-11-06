import isce3.ext.isce3 as isce3
import numpy as np
import numpy.testing as npt


def make_delta_test_image(shape):
    z = np.zeros(shape, "c8")
    z[0, 0] = 1.0
    return z


def test_make_nfft2d():
    # Input is Kronecker delta.
    image = make_delta_test_image((128, 365))

    # NFFT parameters for both dimensions.
    params = {'rows': {'m': 4, 's': 4.0}, 'cols': {'m': 4, 's': 4.0}}

    # Use convenience function to create an interpolator.
    itp = isce3.signal.make_image_nfft2d(image, params)

    # Interp at origin.
    z0 = itp.interp((0.0, 0.0))

    npt.assert_allclose(z0, 1.0+0j, atol=1e-4)


def test_nfft2d():
    # Input is Kronecker delta.
    image = make_delta_test_image((128, 365))

    # Compute its spectrum.
    image_spectrum = np.fft.fft2(image)
    shape = image_spectrum.shape

    # NFFT parameters for both dimensions.
    m = s = 4

    # Plan 2D NFFT
    nfft = isce3.signal.NFFT2dF32((m, m), shape, [n * s for n in shape])

    # Execute to get time-domain image ready for interpolation.
    itp = nfft.transform(image_spectrum)

    # Interpolate at origin.
    z0 = itp.interp((0.0, 0.0))

    # Check result
    npt.assert_allclose(z0, 1.0+0j, atol=1e-4)

    # Check bindings
    npt.assert_equal(nfft.sizes, shape)
    npt.assert_equal(nfft.fft_sizes, [s * n for n in shape])
    npt.assert_equal(nfft.spectrum.shape, nfft.fft_sizes)