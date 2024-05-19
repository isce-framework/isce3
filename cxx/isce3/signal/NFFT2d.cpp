#include "NFFT2d.h"

#include <isce3/core/Interp2d.h>

template <typename T>
using Kernel = isce3::core::NFFTKernel<T>;

namespace isce3::signal {

// constructor
template <class T>
NFFT2d<T>::NFFT2d(const dims_t& m, const dims_t& sizes, const dims_t& fft_sizes)
    : m_(m), sizes_(sizes), fft_sizes_(fft_sizes), kernels_(
        {Kernel<T>{m[0], sizes[0], fft_sizes[0]},
        Kernel<T>{m[1], sizes[1], fft_sizes[1]}})

{
    size_t nout = static_cast<size_t>(fft_sizes[0]) * fft_sizes[1];
    xf_.resize(nout);
    xt_.resize(nout);

    int idims[2] = {fft_sizes_[0], fft_sizes_[1]};
    inv_plan_ = isce3::fft::planifft2d<T>(xt_.data(), xf_.data(), idims);

    // Pre-compute spectral weights (1/phi_hat in NFFT papers).
    // Also include factor of n since FFTW does not normalize DFT.
    for (int idim = 0; idim < ndims; ++idim) {
        weights_[idim].resize(sizes[idim]);
        T b = M_PI * (2.0 - 1.0 * sizes[idim] / fft_sizes[idim]);
        T norm = isce3::math::bessel_i0(b * m[idim]) / sizes[idim];
        size_t n2 = (sizes[idim] - 1) / 2 + 1;
        for (size_t i = 0; i < n2; ++i) {
            double f = 2 * M_PI * i / fft_sizes_[idim];
            weights_[idim][i] = norm /
                isce3::math::bessel_i0(m[idim] * std::sqrt(b * b - f * f));
        }
        for (size_t i = n2; i < sizes[idim]; ++i) {
            double f = 2 * M_PI * ((double)i - sizes[idim]) / fft_sizes[idim];
            weights_[idim][i] = norm /
                isce3::math::bessel_i0(m[idim] * std::sqrt(b * b - f * f));
        }
    }
}

// Digest some data.
template<class T>
void
NFFT2d<T>::set_spectrum(const dims_t& sizes, const dims_t& strides, const std::complex<T> *x)
{
    for (int idim = 0; idim < ndims; ++idim) {
        if (sizes[idim] != sizes_[idim]) {
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                "Spectrum size != NFFT size.");
        }
    }
    // Clear any old data.
    size_t nout = static_cast<size_t>(fft_sizes_[0]) * fft_sizes_[1];
    xf_.assign(nout, std::complex<T>(0, 0));

    const size_t m2 = sizes_[0] / 2;
    const size_t n2 = sizes_[1] / 2;

    // Zero-pad and scale spectrum.
    #pragma omp parallel for
    for (size_t i = 0; i < m2; ++i) {
        const auto wi = weights_[0][i];
        // pointer to row i of input data
        const auto pxi = x + (strides[0] * i);
        // pointer to row i of ifft buffer
        const auto pxfi = xf_.data() + (fft_sizes_[1] * i);
        // columns [0, n2)
        for (size_t j = 0; j < n2; ++j) {
            const auto wj = weights_[1][j];
            pxfi[j] = wi * wj * pxi[strides[1] * j];
        }
        // columns [-n2, 0)
        for (size_t j = n2; j > 0; --j) {
            const auto wj = weights_[1][sizes_[1] - j];
            pxfi[fft_sizes_[1] - j] = wi * wj * pxi[strides[1] * (sizes_[1] - j)];
        }
    }
    #pragma omp parallel for
    for (size_t i = m2; i > 0; --i) {
        const auto wi = weights_[0][sizes_[0] - i];
        // pointer to row (ny - i)
        const auto pxi = x + (strides[0] * (sizes_[0] - i));
        const auto pxfi = xf_.data() + (fft_sizes_[1] * (fft_sizes_[0] - i));
        for (size_t j = 0; j < n2; ++j) {
            const auto wj = weights_[1][j];
            pxfi[j] = wi * wj * pxi[strides[1] * j];
        }
        for (size_t j = n2; j > 0; --j) {
            const auto wj = weights_[1][sizes_[1] - j];
            pxfi[fft_sizes_[1] - j] = wi * wj * pxi[strides[1] * (sizes_[1] - j)];
        }
    }

    // NOTE For even lengths we're not splitting Nyquist bin.
    // Transform to (expanded) time-domain.
    inv_plan_.execute();
}

template <typename T>
std::complex<T> NFFT2d<T>::interp(const std::array<double, 2>& t) const
{
    constexpr int xdim = 1, ydim = 0;

    // scale time index to account for zero-padding of spectrum.
    double x = t[xdim] * fft_sizes_[xdim] / sizes_[xdim];
    double y = t[ydim] * fft_sizes_[ydim] / sizes_[ydim];

    return isce3::core::interp2d<T, std::complex<T>>(kernels_[xdim],
        kernels_[ydim], xt_.data(), fft_sizes_[xdim], /* stridex */ 1,
        fft_sizes_[ydim], /* stridey */ fft_sizes_[xdim], x, y,
        /* periodic */true);
       
}

}

template class isce3::signal::NFFT2d<float>;
template class isce3::signal::NFFT2d<double>;
