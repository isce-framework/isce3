/** @file ElNullAnalyses.h
 * A collection of functions for EL Null analysis in range direction
 */
#pragma once

#include <complex>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include <isce3/core/EMatrix.h>
#include <isce3/core/Linspace.h>

#include "detail/BinarySearchFunc.h"

/** @namespace isce3::antenna */
namespace isce3 { namespace antenna {

// Aliases
using RowMatrixXcf = isce3::core::EMatrix2D<std::complex<float>>;
using Linspace_t = isce3::core::Linspace<double>;
// coef_left (complex), coef_right (complex), el angles (rad)
using tuple_ant = std::tuple<Eigen::ArrayXcd, Eigen::ArrayXcd, Eigen::ArrayXd>;
// echo null power pattern (dB) , el angles (rad), slant range (m)
using tuple_echo = std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, detail::ArrayXui>;

// functions:
/**
 * 1-D linear interpolation of a complex array of uniformly-sampled y(x) at
 * new array of points x0.
 * @param[in] x0 is array of values to be interpolated
 * @param[in] x is isce3 Linspace object for uniformly sampled input "x"
 * @param[in] y is array of function value "y(x)".
 * @return array of interpolated complex values at samples "x0".
 * @exception InvalidArgument
 */
Eigen::ArrayXcd linearInterpComplex1d(
        const Eigen::Ref<const Eigen::ArrayXd>& x0, const Linspace_t& x,
        const Eigen::Ref<const Eigen::ArrayXcd>& y);

/**
 * Generate weighting coefficients used in digital beamforming (DBF) and null
 * formation for a pair of adjacent, partially overlapping beams.
 *
 * Computes DBF coefficients by forming the complex conjugate of each input
 * antenna elevation pattern within the inner peak-to-peak region between the
 * two beams. The resulting coefficients may be resampled to finer resolution
 * than the input spacing, if desired. The function returns a vector of
 * coefficients for each beam, as well as a vector of the corresponding
 * elevation angle positions.
 * @param[in] el_cut_left is complex array of uniformly-sampled relative or
 * absolute EL-cut antenna pattern on the left side.
 * @param[in] el_cut_right is complex array of uniformly-sampled relative or
 * absolute EL-cut antenna pattern on the right side. It must have the same size
 * as left one!
 * @param[in] el_ang_start elevation angle of the first sample in the left/right
 * patterns, in (rad)
 * @param[in] el_ang_step elevation angular spacing for left/right patterns in
 * (rad)
 * @param[in] el_res_max (optional) max EL angle resolution in (rad). If
 * this value is smaller than the input spacing, `el_ang_step`, then the outputs
 * will be resampled via linear interpolation to either this finer resolution or
 * a value close to that depending on whether ratio `el_ang_step/el_res_max` is
 * an integer or not.
 * @return array of complex coeff for the left beam
 * @return array of complex coeff for the right beam
 * @return array of uniformly-sampled EL angles in (rad)
 * @exception InvalidArgument, RuntimeError
 */
tuple_ant genAntennaPairCoefs(
        const Eigen::Ref<const Eigen::ArrayXcd>& el_cut_left,
        const Eigen::Ref<const Eigen::ArrayXcd>& el_cut_right,
        double el_ang_start, double el_ang_step,
        std::optional<double> el_res_max = {});

/**
 * Locate antenna null by forming antenna null and get its min location in
 * EL direction from a pair of EL antenna pattern or its equivalent complex
 * conjugate coeffs.
 * @param[in] coef_left is complex array of coef for the left beam
 * @param[in] coef_right is complex array of coef for the right beam
 * @param[in] el_ang_vec is array of EL angles in (rad)
 * @return el angle of the null location in (rad)
 * @return index of null location
 * @return peak-normalized null magnitude in (linear)
 * @return 1-D peak-normalized antenna null power pattern (linear) with the
 * same size as `el_ang_vec`.
 * @exception RuntimeError
 */
std::tuple<double, Eigen::Index, double, Eigen::ArrayXd> locateAntennaNull(
        const Eigen::Ref<const Eigen::ArrayXcd>& coef_left,
        const Eigen::Ref<const Eigen::ArrayXcd>& coef_right,
        const Eigen::Ref<const Eigen::ArrayXd>& el_ang_vec);

/**
 * Form x-track echo null power (linear) averaged over range lines and formed
 * from a pair of echoes and a pair of weighting coefs as a function of both EL
 * angles (rad) and slant ranges (m).
 * @param[in] chirp_ref is complex chirp ref samples used in range compression.
 * @param[in] echo_left is complex 2-D array of raw echo samples (pulse by
 * range) for the left RX channel corresponding to the left beam.
 * @param[in] echo_right is complex 2-D array of raw echo samples (pulse by
 * range) for the right RX channel corresponding to the right beam.
 * @param[in] sr_start is start slant range (m) for both uniformly-sampled
 * echoes in range.
 * @param[in] sr_spacing is slant range spacing (m) for both uniformly-sampled
 * echoes in range.
 * @param[in] coef_left is complex array of coef for the left beam.
 * @param[in] coef_right is complex array of coef for the right beam.
 * @param[in] sr_coef array of slant ranges (m) for both left/right coeffs
 * @return array of echo null power pattern in (linear).
 * @return array of slant range values related to the null power pattern.
 * @return array of indices used for mapping null slant ranges to antenna EL
 * angles.
 * @exception RuntimeError
 */
tuple_echo formEchoNull(const std::vector<std::complex<float>>& chirp_ref,
        const Eigen::Ref<const RowMatrixXcf>& echo_left,
        const Eigen::Ref<const RowMatrixXcf>& echo_right, double sr_start,
        double sr_spacing, const Eigen::Ref<const Eigen::ArrayXcd>& coef_left,
        const Eigen::Ref<const Eigen::ArrayXcd>& coef_right,
        const Eigen::Ref<const Eigen::ArrayXd>& sr_coef);

}} // namespace isce3::antenna
