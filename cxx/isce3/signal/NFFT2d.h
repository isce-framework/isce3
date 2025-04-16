#pragma once

#include "forward.h"

#include <isce3/core/EMatrix.h>
#include <isce3/core/Kernels.h>
#include <isce3/fft/FFT.h>

#include <array>
#include <complex>


namespace isce3::signal {

template<typename T>
class NFFT2d {
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


/**
 * @brief Create an NFFT2d object for interpolating an image.
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
 * @return NFFT2d<T> object for interpolating the image.
 */
template<typename T>
NFFT2d<T> makeImageNFFT2d(
    const Eigen::Ref<const isce3::core::EArray2D<std::complex<T>>>& image,
    const typename NFFT2d<T>::dims_t& m = {2, 2},
    const std::array<double, 2>& s = {2.0, 2.0},
    bool pad_input = false);

} // namespace isce3::signal