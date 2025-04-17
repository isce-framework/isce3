#include "NFFT2d.h"

#include <isce3/core/Interp2d.h>
#include <isce3/fft/FFTUtil.h>

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
std::complex<T> NFFT2d<T>::interp(const std::array<double, 2>& t, bool periodic) const
{
    constexpr int xdim = 1, ydim = 0;

    // scale time index to account for zero-padding of spectrum.
    double x = t[xdim] * fft_sizes_[xdim] / sizes_[xdim];
    double y = t[ydim] * fft_sizes_[ydim] / sizes_[ydim];

    return isce3::core::interp2d<T, std::complex<T>>(kernels_[xdim],
        kernels_[ydim], xt_.data(), fft_sizes_[xdim], /* stridex */ 1,
        fft_sizes_[ydim], /* stridey */ fft_sizes_[xdim], x, y, periodic);
}


template<typename T>
NFFT2d<T> makeImageNFFT2d(
    const Eigen::Ref<const isce3::core::EArray2D<std::complex<T>>>& image,
    const NFFT2dParams& params,
    bool pad_input)
{
    using isce3::fft::nextFastPower;

    auto rows_in = image.rows();
    auto cols_in = image.cols();
    using image_t = isce3::core::EArray2D<std::complex<T>>;
    auto image_copy = image_t(0, 0);

    // Pointer to input image or padded/copied version so we can have fewer
    // conditionals later.
    // FIXME figure out how to do this with an Eigen type...
    auto image_ptr = image.data();

    // Need to copy if image is not contiguous row-major since we don't have
    // high-level interface for strided FFTs.
    bool need_copy = (image.innerStride() != 1) or (image.outerStride() != cols_in);
    if (need_copy) {
        image_copy.resize(rows_in, cols_in);
        // assign later
    }

    if (pad_input) {
        auto padded_rows_in = nextFastPower(rows_in);
        auto padded_cols_in = nextFastPower(cols_in);
        if ((rows_in == padded_rows_in) and (cols_in == padded_cols_in)) {
            // User asked for padding but we don't actually need it.
            pad_input = false;
        } else {
            image_copy.resize(padded_rows_in, padded_cols_in);
            image_copy.setZero();
            rows_in = padded_rows_in;
            cols_in = padded_cols_in;
            // assign later
        }
    }

    if (need_copy or pad_input) {
        // This way NFFT2d::interp() coordinates are preserved, though user
        // will be able to get some extra data.
        image_copy.topLeftCorner(rows_in, cols_in) = image;
        image_ptr = image_copy.data();
    }

    // Use fft2 b/c planfft2d could modify inputs and we won't reuse it anyway.
    using dims_t = typename NFFT2d<T>::dims_t;
    dims_t dims = {
        static_cast<int>(rows_in),
        static_cast<int>(cols_in)};
    auto spectrum = image_t(dims[0], dims[1]);
    isce3::fft::fft2d(spectrum.data(), image_ptr, {dims[0], dims[1]});

    // Calculate sizes for padded inverse transform.
    dims_t dims_out = {
        nextFastPower(static_cast<int>(std::round(params.rows.s * dims[0]))),
        nextFastPower(static_cast<int>(std::round(params.cols.s * dims[1])))};

    const dims_t m = {params.rows.m, params.cols.m};
    auto interpolator = NFFT2d<T>(m, dims, dims_out);
    interpolator.set_spectrum(dims, {dims[1], 1}, spectrum.data());
    return interpolator;
}

}

template class isce3::signal::NFFT2d<float>;
template class isce3::signal::NFFT2d<double>;

template isce3::signal::NFFT2d<float>
isce3::signal::makeImageNFFT2d(
    const Eigen::Ref<const isce3::core::EArray2D<std::complex<float>>>& image,
    const isce3::signal::NFFT2dParams& params,
    bool pad_input);

template isce3::signal::NFFT2d<double>
isce3::signal::makeImageNFFT2d(
    const Eigen::Ref<const isce3::core::EArray2D<std::complex<double>>>& image,
    const isce3::signal::NFFT2dParams& params,
    bool pad_input);
