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

        /**
         * @brief Construct a new NFFT2d object
         *
         * @param m         Interpolator half-length along {rows, columns}
         * @param sizes     Image spectrum dimensions {rows, columns}
         * @param fft_sizes Transform sizes along {rows, columns}.
         *                  Usually larger than image size.
         */
        NFFT2d(const dims_t& m, const dims_t& sizes, const dims_t& fft_sizes);

        /**
         * @brief Ingest the image spectrum.
         *
         * @param sizes     Image spectrum dimensions {rows, columns}.
         *                  Must match dimensions provided in ctor.
         * @param strides   Strides (in pixels) along each dimension {rows, columns}.
         * @param x         Image spectrum.
         *
         * The spectrum will be zero-padded, pre-filtered, and transformed
         * to the time-domain.
         */
        void set_spectrum(const dims_t& sizes, const dims_t& strides,
            const std::complex<T> *x);

        /**
         * @brief Interpolate the image
         *
         * @param t         Desired pixel location {row, column}
         * @param periodic  Whether to use a periodic boundary condition.
         * @return          Interpolated value.
         */
        std::complex<T> interp(const std::array<double, 2>& t,
            bool periodic = true) const;

        /** Image spectrum dimensions */
        const dims_t& sizes() const { return sizes_; }

        /** Transform sizes */
        const dims_t& fft_sizes() const { return fft_sizes_; }

        /** Pointer to most recent spectral data (filtered and padded) */
        const std::complex<T>* spectrum() const { return xf_.data(); }

    private:
        dims_t m_, sizes_, fft_sizes_;
        std::vector<std::complex<T>> xf_, xt_;
        std::array<std::vector<T>, 2> weights_;
        std::array<isce3::core::NFFTKernel<T>, 2> kernels_;
        isce3::fft::InvFFTPlan<T> inv_plan_;
};
