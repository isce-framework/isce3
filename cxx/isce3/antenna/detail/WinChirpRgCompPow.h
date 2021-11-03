/** @file WinChirpRgCompPow.h
 * Chirp and range compress related helper functionalities
 */
#pragma once

#include <complex>
#include <tuple>
#include <vector>

#include <isce3/core/EMatrix.h>

namespace isce3 { namespace antenna { namespace detail {

// Aliases, typedef:
using RowMatrixXcf = isce3::core::EMatrix2D<std::complex<float>>;

// Functions:

/**
 * Generate a raised-cosine window function with a desired pedestal.
 * Window is symmetric and has a peak equal to 1.0 at the center only
 * for odd window size. The end points are equal to pedestal value.
 * @param[in] size a positive integer for
 * size of the window > 1.
 * @param[in] ped pedestal of the widnow. A value
 * within [0.0, 1.0] where 1.0 is rectangular window
 * while 0.0 is Hann window. Ped=0.08 is a Hamming window.
 * @return a floating-point vector of window function.
 * @exception InvalidArgument
 * @note Given pedestal value "p" and size "L", the window function is
 * \f$\frac{1+p}{2} - \frac{1-p}{2} \times \cos(\frac{2\pi}{L-1}n)\f$
 * where \f$n = 0, 1, 2, \cdots, L-1\f$.
 */
template<typename T = float>
std::vector<T> genRaisedCosineWin(int size, double ped);

/**
 * Generate a unit-energy raised-cosine-windowed baseband complex analytical
 * chirp. That is a basebanded linear FM with a symmetric rectangular or
 * raised-cosine envelope.
 * @param[in] sample_freq sampling frequency in (Hz) or (MHz)
 * @param[in] chirp_slope chirp slope in (Hz/sec) or (MHz/usec) depending on
 * unit of "sample_freq".
 * @param[in] chirp_dur chirp duration in (sec) or (usec) depending on the
 * inverse unit of "sample_freq".
 * @param[in] win_ped raised-cosine window pedestal, a value within [0.0, 1.0].
 * Default is "Hann" window.
 * @param[in] norm a bool, whether or not normalize output to be unit-energy.
 * @return a vector of time-series complex floating-point chirp samples.
 * @exception InvalidArgument
 * @see isce3::focus::formLinearChirp(), genRaisedCosineWin()
 */
std::vector<std::complex<float>> genRcosWinChirp(double sample_freq,
        double chirp_slope, double chirp_dur, double win_ped = 0.0,
        bool norm = true);

/**
 * Averaged Power of range compressed complex raw echo over multiple range lines
 * as a function of true valid range bins. Only valid part of range compressed
 * data will be returned!
 * @param[in] echo_mat raw echo matrix, a row-major Eigen matrix of type
 * complex float. The rows represent range lines. The matrix shape is pulses
 * (azimuth bins) by range bins.
 * @param[in] chirp_ref Basebanded Chirp reference complex float vector used in
 * range compresssion. Its size shall not be larger than number of range bins or
 * columns of "echo_mat", otherwise, it throws LengthError.
 * @return An eigen vector of real-value double precision representing averaged
 * power over range lines of range compressed echo as a function of true and
 * valid range bins after pulse extension deconvolution. The number of range
 * bins will be smaller than that of raw echo (its columns) due to pulse
 * extension.
 * @exception LengthError
 */
Eigen::ArrayXd meanRgCompEchoPower(
        const Eigen::Ref<const RowMatrixXcf>& echo_mat,
        const std::vector<std::complex<float>>& chirp_ref);

/**
 * Path-loss corrected/calibrated averaged/decimated uniformly-sampled echo
 * power as a function of slant ranges.
 * Note that the last few samples might be thrown out depending on averaging
 * block size.
 * @param[in] echo_pow a vector of echo power uniformly-sampled in range.
 * @param[in] rg_start starting range in (meters)
 * @param[in] rg_spacing range spacing in (meters)
 * @param[in] size_avg block size for averaing. That is the number of range
 * bins to be averaged. This also deteremines the final range spacing which is
 * "size_avg * rg_space". Note that size_avg shall be within [1,echo_pow.size].
 * @return power array
 * @return slant-range array
 * @exception InvalidArgument, LengthError
 */
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> rangeCalibAvgEchoPower(
        const Eigen::Ref<const Eigen::ArrayXd>& echo_pow, double rg_start,
        double rg_spacing, int size_avg = 1);

}}} // namespace isce3::antenna::detail

// definition of the functions
#include "WinChirpRgCompPow.icc"
