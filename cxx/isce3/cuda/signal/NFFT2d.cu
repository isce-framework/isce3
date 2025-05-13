#include "NFFT2d.h"

#include <isce3/cuda/core/Interp2d.h>
#include <isce3/cuda/fft/FFT.h>
#include <isce3/signal/NFFT2d.h>

template <typename T>
using Kernel = isce3::cuda::core::NFFTKernel<T>;

namespace isce3::cuda::signal {

// constructor
template <class T>
NFFT2d<T>::NFFT2d(const dims_t& m, const dims_t& sizes, const dims_t& fft_sizes)
    : m_(m), sizes_(sizes), fft_sizes_(fft_sizes), kernels_(
        {Kernel<T>{m[0], sizes[0], fft_sizes[0]},
        Kernel<T>{m[1], sizes[1], fft_sizes[1]}})
{
    size_t nout = static_cast<size_t>(fft_sizes[0]) * fft_sizes[1];
    xf_.resize(nout);

    // Just compute weights on CPU for now and then copy to GPU.

    // Pre-compute spectral weights (1/phi_hat in NFFT papers).
    // Also include factor of n since FFTW does not normalize DFT.
    for (int idim = 0; idim < ndims; ++idim) {
        auto weight = std::vector<T>(sizes[idim]);
        T b = M_PI * (2.0 - 1.0 * sizes[idim] / fft_sizes[idim]);
        T norm = isce3::math::bessel_i0(b * m[idim]) / sizes[idim];
        size_t n2 = (sizes[idim] - 1) / 2 + 1;
        for (size_t i = 0; i < n2; ++i) {
            double f = 2 * M_PI * i / fft_sizes_[idim];
            weight[i] = norm /
                isce3::math::bessel_i0(m[idim] * std::sqrt(b * b - f * f));
        }
        for (size_t i = n2; i < sizes[idim]; ++i) {
            double f = 2 * M_PI * ((double)i - sizes[idim]) / fft_sizes[idim];
            weight[i] = norm /
                isce3::math::bessel_i0(m[idim] * std::sqrt(b * b - f * f));
        }
        weights_[idim].assign(weight.begin(), weight.end());
    }
}

template <typename T>
__global__ void setSpectrum2d(thrust::complex<T>* xout, int rows_out, int cols_out, 
    thrust::complex<T>* xin, int rows_in, int cols_in, int row_stride_in,
    int col_stride_in, T* weights_rows, T* weights_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col > cols_in) || (row > rows_in)) {
        return;
    }

    // NOTE For even lengths we're not splitting Nyquist bins.
    const auto kin = static_cast<long>(row) * row_stride_in +
        col * col_stride_in;

    long row_out = 0, col_out = 0;

    const int m2 = rows_in / 2;
    const int n2 = cols_in / 2;

    if (row < m2) {
        row_out = row;
    } else {
        row_out = rows_out - (rows_in - row);
    }
    if (col < n2) {
        col_out = col;
    } else {
        col_out = cols_out - (cols_in - col);
    }

    const auto kout = row_out * cols_out + col_out;
    xout[kout] = weights_rows[row] * weights_cols[col] * xin[kin];
}

// Digest some data.
template<class T>
NFFT2dResult<T>
NFFT2d<T>::transform(const dims_t& sizes, const dims_t& strides, const std::complex<T> *x)
{
    for (int idim = 0; idim < ndims; ++idim) {
        if (sizes[idim] != sizes_[idim]) {
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                "Spectrum size != NFFT size.");
        }
    }
    // Clear any old data.
    auto nout = static_cast<size_t>(fft_sizes_[0]) * fft_sizes_[1];
    xf_.assign(nout, thrust::complex<T>(0, 0));

    // Copy input data to device.
    auto nin = static_cast<size_t>(sizes[0]) * sizes[1];
    auto px = reinterpret_cast<const thrust::complex<T>*>(x);
    thrust::device_vector<thrust::complex<T>> d_x(px, px + nin);

    // Pad and weight
    {
        dim3 cu_block(16, 16);
        dim3 cu_grid(
            (sizes[1] + cu_block.x - 1) / cu_block.x,
            (sizes[0] + cu_block.y - 1) / cu_block.y);

        setSpectrum2d<<<cu_grid, cu_block>>>(
            xf_.data().get(), fft_sizes_[0], fft_sizes_[1],
            d_x.data().get(), sizes[0], sizes[1], strides[0], strides[1],
            weights_[0].data().get(), weights_[1].data().get());

        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    }

    // Transform to time domain.
    auto result = NFFT2dResult<T>(m_, sizes_, fft_sizes_, kernels_);
    int dims[] = {fft_sizes_[0], fft_sizes_[1]};
    isce3::cuda::fft::ifft2d(result.xt_.data().get(), xf_.data().get(), dims);

    return result;
}

template<typename T>
NFFT2dResult<T>::operator isce3::signal::NFFT2dResult<T>() const
{
    using CpuKernel = isce3::core::NFFTKernel<T>;
    auto result = isce3::signal::NFFT2dResult<T>(m_, sizes_, fft_sizes_,
            {CpuKernel {kernels_[0]}, CpuKernel {kernels_[1]}});
    thrust::copy(xt_.begin(), xt_.end(), result.data());
    return result;
}

template<typename T>
NFFT2dResult<T>::NFFT2dResult(const isce3::signal::NFFT2dResult<T>& other)
    : m_ {other.kernel_radii()}, sizes_ {other.sizes()},
      fft_sizes_ {other.fft_sizes()},
      kernels_ {Kernel<T> {other.kernels()[0]}, Kernel<T> {other.kernels()[1]}}
{
    const auto npix = static_cast<size_t>(fft_sizes_[0]) * fft_sizes_[1];
    this->xt_.assign(other.data(), other.data() + npix);
}

template <typename T>
NFFT2dResultView<T>::NFFT2dResultView(const NFFT2dResult<T>& result) :
    sizes_{result.sizes_},
    fft_sizes_{result.fft_sizes_},
    pxt_{result.xt_.data().get()},
    kernels_{result.kernels_}
    {}

// FIXME If I put this here instead of inline in the header, then the unit
// test doesn't compile...
#if 0
template <typename T>
CUDA_DEV
thrust::complex<T>
NFFT2dResultView<T>::interp(const std::array<double, 2>& t, bool periodic) const
{
    constexpr int xdim = 1, ydim = 0;

    // scale time index to account for zero-padding of spectrum.
    double x = t[xdim] * fft_sizes_[xdim] / sizes_[xdim];
    double y = t[ydim] * fft_sizes_[ydim] / sizes_[ydim];

    return isce3::cuda::core::interp2d(kernels_[xdim],
        kernels_[ydim], pxt_, fft_sizes_[xdim], /* stridex */ 1,
        fft_sizes_[ydim], /* stridey */ fft_sizes_[xdim], x, y, periodic);
}
#endif

}

template class isce3::cuda::signal::NFFT2d<float>;
template class isce3::cuda::signal::NFFT2d<double>;
template class isce3::cuda::signal::NFFT2dResult<float>;
template class isce3::cuda::signal::NFFT2dResult<double>;
template class isce3::cuda::signal::NFFT2dResultView<float>;
template class isce3::cuda::signal::NFFT2dResultView<double>;
