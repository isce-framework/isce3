#pragma once

#include "forward.h"

#include <isce3/core/Kernels.h>
#include <isce3/fft/FFT.h>

#include <array>
#include <complex>


template<typename T>
class isce3::signal::NFFT2d {
    public:
        static constexpr int ndims = 2;
        using dims_t = std::array<int, ndims>;

        NFFT2d() = delete;
        NFFT2d(const dims_t& m, const dims_t& sizes, const dims_t& fft_sizes);

        void set_spectrum(const dims_t& sizes, const dims_t& strides,
            const std::complex<T> *x);

        std::complex<T> interp(const std::array<double, 2>& t) const;
        
    private:
        dims_t m_, sizes_, fft_sizes_;
        std::vector<std::complex<T>> xf_, xt_;
        std::array<std::vector<T>, 2> weights_;
        std::array<isce3::core::NFFTKernel<T>, 2> kernels_;
        isce3::fft::InvFFTPlan<T> inv_plan_;
};
