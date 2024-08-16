#pragma once

#include "forward.h"

#include <isce3/cuda/core/Kernels.h>
#include <isce3/cuda/fft/FFTPlan.h>
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
    friend class NFFT2dView<T>;
    public:
        static constexpr int ndims = 2;
        using dims_t = std::array<int, ndims>;

        NFFT2d() = delete;
        NFFT2d(const dims_t& m, const dims_t& sizes, const dims_t& fft_sizes);

        void set_spectrum(const dims_t& sizes, const dims_t& strides,
            const std::complex<T> *x);

        // CUDA_DEV
        // thrust::complex<T> interp(const std::array<double, 2>& t,
        //     bool periodic = true) const;

        const dims_t& sizes() const { return sizes_; }
        const dims_t& fft_sizes() const { return fft_sizes_; }

        const thrust::complex<T>* spectrum() const { return xf_.data().get(); }
        
    private:
        dims_t m_, sizes_, fft_sizes_;
        thrust::device_vector<thrust::complex<T>> xf_, xt_;
        std::array<thrust::device_vector<T>, 2> weights_;
        std::array<isce3::cuda::core::TabulatedKernel<T>, 2> kernels_;
        isce3::cuda::fft::InvFFTPlan<T> inv_plan_;
};

template<typename T>
class NFFT2dView {
    public:
        static constexpr int ndims = 2;
        using dims_t = std::array<int, ndims>;

        NFFT2dView() = delete;
        NFFT2dView(const NFFT2d<T>& nfft);

        CUDA_DEV
        thrust::complex<T> interp(const std::array<double, 2>& t,
            bool periodic = true) const;

        CUDA_DEV
        const dims_t& fft_sizes() const { return fft_sizes_; }

    private:
        dims_t sizes_, fft_sizes_;
        const thrust::complex<T>* pxt_;
        std::array<isce3::cuda::core::TabulatedKernelView<T>, 2> kernel_views_;
};

}
