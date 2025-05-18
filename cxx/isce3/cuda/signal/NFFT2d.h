#pragma once

#include "forward.h"

#include <isce3/core/EMatrix.h>
#include <isce3/cuda/core/Kernels.h>
#include <isce3/cuda/core/Interp2d.h>
#include <isce3/cuda/fft/FFTPlan.h>
#include <isce3/signal/forward.h>
#include <isce3/signal/NFFT2d.h>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <array>
#include <complex>

namespace isce3::cuda::signal {

template<typename T>
class NFFT2d {
    public:
        static constexpr int ndims = 2;
        using dims_t = std::array<int, ndims>;

        NFFT2d() = delete;
        NFFT2d(const dims_t& m, const dims_t& sizes, const dims_t& fft_sizes);

        // input data in host memory
        NFFT2dResult<T> transform_host(const dims_t& sizes,
            const dims_t& strides, const std::complex<T>* x);

        // input data already on device
        NFFT2dResult<T> transform_device(const dims_t& sizes,
            const dims_t& strides, const thrust::complex<T>* x);

        const dims_t& sizes() const { return sizes_; }
        const dims_t& fft_sizes() const { return fft_sizes_; }

        const thrust::complex<T>* spectrum() const { return xf_.data().get(); }
        
    private:
        dims_t m_, sizes_, fft_sizes_;
        thrust::device_vector<thrust::complex<T>> xf_;
        std::array<thrust::device_vector<T>, 2> weights_;
        std::array<isce3::cuda::core::NFFTKernel<T>, 2> kernels_;
};

template<typename T>
class NFFT2dResult {
    friend class NFFT2d<T>;
    friend class NFFT2dResultView<T>;

public:
    using dims_t = typename NFFT2d<T>::dims_t;

    NFFT2dResult() = delete;

    NFFT2dResult(const dims_t& m, const dims_t& sizes, const dims_t& fft_sizes,
            const std::array<isce3::cuda::core::NFFTKernel<T>, 2>& kernels,
            const thrust::complex<T>* xt = nullptr)
        : m_ {m}, sizes_ {sizes}, fft_sizes_ {fft_sizes}, kernels_ {kernels}
    {
        const auto n = static_cast<size_t>(fft_sizes[0]) * fft_sizes[1];
        if (xt == nullptr) {
            xt_.resize(n);
        } else {
            xt_.assign(xt, xt + n);
        }
    };

    /** copy to host */
    explicit operator isce3::signal::NFFT2dResult<T>() const;

    /** copy from host */
    NFFT2dResult(const isce3::signal::NFFT2dResult<T>& other);

    const dims_t& kernel_radii() const { return m_; }
    const dims_t& sizes() const { return sizes_; }
    const dims_t& fft_sizes() const { return fft_sizes_; }
    const auto& kernels() const { return kernels_; }

private:
    dims_t m_, sizes_, fft_sizes_;
    std::array<isce3::cuda::core::NFFTKernel<T>, 2> kernels_;
    thrust::device_vector<thrust::complex<T>> xt_;
};

template<typename T>
class NFFT2dResultView {
public:
    static constexpr int ndims = 2;
    using dims_t = std::array<int, ndims>;

    NFFT2dResultView() = delete;
    NFFT2dResultView(const NFFT2dResult<T>& result);

    CUDA_DEV inline
    thrust::complex<T> interp(
            const std::array<double, 2>& t, bool periodic = true) const
    {
        constexpr int xdim = 1, ydim = 0;

        // scale time index to account for zero-padding of spectrum.
        double x = t[xdim] * fft_sizes_[xdim] / sizes_[xdim];
        double y = t[ydim] * fft_sizes_[ydim] / sizes_[ydim];

        return isce3::cuda::core::interp2d(kernels_[xdim], kernels_[ydim], pxt_,
                fft_sizes_[xdim], /* stridex */ 1, fft_sizes_[ydim],
                /* stridey */ fft_sizes_[xdim], x, y, periodic);
    };

    CUDA_DEV
    const dims_t& fft_sizes() const { return fft_sizes_; }

private:
    dims_t sizes_, fft_sizes_;
    const thrust::complex<T>* pxt_;
    std::array<isce3::cuda::core::NFFTKernel<T>, 2> kernels_;
};


/**
 * @brief Create an NFFT2dResult object for interpolating an image.
 *
 * @tparam T        Format of real/imag pixel data, typically float or double
 * @param image     Input time-domain image.  A temporary copy will be made if
 *                  it is not row-major with a column stride of one.
 * @param m         Half-length of interpolator along {rows, columns}
 * @param s         Minimum factors (> 1) for frequency-domain zero-padding
 *                  along {rows, columns}.  Actual padding may be larger to
 *                  achieve efficient inverse transform size.
 * @param pad_input Whether to also zero-pad input data to an efficient
 *                  forward transform size.  Requires extra memory.
 *
 * @return NFFT2dResult<T> object for interpolating the image.
 */
template<typename T>
NFFT2dResult<T> makeImageNFFT2d(
    const Eigen::Ref<const isce3::core::EArray2D<std::complex<T>>>& image,
    const isce3::signal::NFFT2dParams& params = {},
    bool pad_input = false);

}
