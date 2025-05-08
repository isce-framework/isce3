#pragma once

#include "forward.h"

#include <isce3/cuda/core/Kernels.h>
#include <isce3/cuda/core/Interp2d.h>
#include <isce3/cuda/fft/FFTPlan.h>
#include <isce3/signal/forward.h>
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

        NFFT2dResult<T> transform(const dims_t& sizes, const dims_t& strides,
            const std::complex<T> *x);

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

    explicit operator isce3::signal::NFFT2dResult<T>() const;

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
}
