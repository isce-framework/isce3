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

/**
 * @brief GPU-accelerated Non-Uniform Fast Fourier Transform (NFFT) in 2D.
 *
 * This class performs a zero-padded, pre-filtered 2D inverse FFT to convert
 * image-spectrum data into the time domain for interpolation.  All transform
 * and intermediate spectral data reside in GPU device memory.
 */
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
         * @brief Transform image spectrum to time domain, input in host memory.
         *
         * @param sizes     Image spectrum dimensions {rows, columns}.
         *                  Must match dimensions provided in ctor.
         * @param strides   Strides (in pixels) along each dimension {rows, columns}.
         * @param x         Image spectrum.
         *
         * The input data is copied to device memory.  The spectrum will be
         * zero-padded, pre-filtered, and transformed to the time-domain.
         *
         * @return NFFT2dResult<T> object containing the time-domain data.
         */
        NFFT2dResult<T> transform_host(const dims_t& sizes,
            const dims_t& strides, const std::complex<T>* x);

        /**
         * @brief Transform image spectrum to time domain, input on device.
         *
         * @param sizes     Image spectrum dimensions {rows, columns}.
         *                  Must match dimensions provided in ctor.
         * @param strides   Strides (in pixels) along each dimension {rows, columns}.
         * @param x         Image spectrum (device pointer).
         *
         * The spectrum will be zero-padded, pre-filtered, and transformed
         * to the time-domain entirely on the GPU.
         *
         * @return NFFT2dResult<T> object containing the time-domain data.
         */
        NFFT2dResult<T> transform_device(const dims_t& sizes,
            const dims_t& strides, const thrust::complex<T>* x);

        /** Image spectrum dimensions */
        const dims_t& sizes() const { return sizes_; }

        /** Transform sizes */
        const dims_t& fft_sizes() const { return fft_sizes_; }

        /** Pointer to most recent spectral data (filtered and padded) */
        const thrust::complex<T>* spectrum() const { return xf_.data().get(); }

    private:
        dims_t m_, sizes_, fft_sizes_;
        thrust::device_vector<thrust::complex<T>> xf_;
        std::array<thrust::device_vector<T>, 2> weights_;
        std::array<isce3::cuda::core::NFFTKernel<T>, 2> kernels_;
};


/**
 * @brief Result of a GPU NFFT2d transform, holding time-domain data on the device.
 *
 * The time-domain data (xt_) is stored in GPU device memory as a
 * thrust::device_vector.  This object can be converted to the CPU
 * counterpart (isce3::signal::NFFT2dResult) or wrapped in a
 * NFFT2dResultView for GPU-side interpolation.
 */
template<typename T>
class NFFT2dResult {
    friend class NFFT2d<T>;
    friend class NFFT2dResultView<T>;

public:
    using dims_t = typename NFFT2d<T>::dims_t;

    NFFT2dResult() = delete;

    /**
     * @brief Construct a NFFT2dResult.
     *
     * @param m         Interpolator half-length along {rows, columns}
     * @param sizes     Image spectrum dimensions {rows, columns}
     * @param fft_sizes Transform sizes along {rows, columns}
     * @param kernels   Interpolator kernels along {rows, columns}
     * @param xt        Optional device pointer to time-domain data of length
     *                  fft_sizes[0] * fft_sizes[1].  If null, a zero-filled
     *                  device buffer is allocated.
     */
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

    /** Interpolator half-lengths along {rows, columns} */
    const dims_t& kernel_radii() const { return m_; }
    /** Image spectrum dimensions */
    const dims_t& sizes() const { return sizes_; }
    /** Transform sizes */
    const dims_t& fft_sizes() const { return fft_sizes_; }
    /** Interpolator kernels along {rows, columns} */
    const auto& kernels() const { return kernels_; }

private:
    dims_t m_, sizes_, fft_sizes_;
    std::array<isce3::cuda::core::NFFTKernel<T>, 2> kernels_;
    thrust::device_vector<thrust::complex<T>> xt_;
};


/**
 * @brief Lightweight view for GPU-side interpolation of NFFT2d results.
 *
 * This class wraps a NFFT2dResult and provides a CUDA device-callable
 * interpolation method.  It is intended for use within device kernels where
 * multiple pixels need to be interpolated without transferring data back to
 * the host.
 */
template<typename T>
class NFFT2dResultView {
public:
    static constexpr int ndims = 2;
    using dims_t = std::array<int, ndims>;

    NFFT2dResultView() = delete;

    /** Construct a lightweight view of an NFFT2dResult. */
    NFFT2dResultView(const NFFT2dResult<T>& result);

    /**
     * @brief Interpolate the image on the device.
     *
     * @param t         Desired pixel location {row, column}
     *                  Values should be in 0 <= t[i] < sizes()[i].
     * @param periodic  Whether to use a periodic boundary condition.
     * @return          Interpolated value.
     */
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

    /** Transform sizes */
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
