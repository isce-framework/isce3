/** @file polyfunc.h
 * Bunch of 1-D polynomial related functionality similar to numpy.polyfit, etc.
 */
#pragma once

#include <cmath>
#include <tuple>

#include <Eigen/Dense>

#include <isce3/core/Poly1d.h>
#include <isce3/except/Error.h>
/** @namespace isce3::math */
namespace isce3 { namespace math {

/**
 * Perform 1-D polyfitting for overdertmined problem via LSE using
 * full-pivot House holder QR decomposition.
 * <a
 * href="https://eigen.tuxfamily.org/dox/classEigen_1_1FullPivHouseholderQR.html"
 * target="_blank">See Eigen QR Solver</a>
 * To improve performance and stability, centering and scaling of the input
 * vector will be performed per request. <a
 * href="https://www.mathworks.com/help/matlab/ref/polyfit.html"
 * target="_blank">See centering and scaling in Matlab polyfit</a>
 * @param[in] x eigen vector of input values
 * @param[in] y eigen vector of "f(x)" and shall have the same size as "x"
 * @param[in] deg order of polynomial.
 * @param[in] center_scale if true will centerize and rescale "x" by its
 * mean and std, respectively, prior to solving for coeffs.
 * @return tuple of three values, a vector of coeff in ascending order, mean of
 * "x", and std of "x". If "center_scale" is false, no centering and scaling is
 * done and thus mean of "x" is set to  0 and std of "x" is set to 1.
 * @exception LengthError, InvalidArgument
 */
std::tuple<Eigen::ArrayXd, double, double> polyfit(
        const Eigen::Ref<const Eigen::ArrayXd>& x,
        const Eigen::Ref<const Eigen::ArrayXd>& y, int deg,
        bool center_scale = false);

/**
 * Perform 1-D polynomial evaluation via
 * <a href="https://en.wikipedia.org/wiki/Horner's_method" target="_blank">
 * Horner's method</a>
 * @param[in] coef a vector of polynomial coeff in ascending order.
 * @param[in] x desired x value to be evaluated.
 * @param[in] mean (optional) mean value or offset for centralizing input.
 * Default is 0.0 which implies no centralization.
 * @param[in] std (optional) std value or divider for scaling input.
 * Default is 1.0 which implies no scaling.
 * @return evaluated scalar value at "x".
 * @exception LengthError, InvalidArgument
 */
double polyval(const Eigen::Ref<const Eigen::ArrayXd>& coef, double x,
        double mean = 0.0, double std = 1.0);

/**
 * Perform 1-D polynomial evaluation.
 * @param[in] coef a vector of polynomial coeff in ascending order.
 * @param[in] x array of x values to be evaluated.
 * @param[in] mean (optional) mean value or offset for centralizing input.
 * Default is 0.0 which implies no centralization.
 * @param[in] std (optional) std value or divider for scaling input.
 * Default is 1.0 which implies no scaling.
 * @return array of evaluated values of x.
 * @exception LengthError, InvalidArgument
 */
Eigen::ArrayXd polyval(const Eigen::Ref<const Eigen::ArrayXd>& coef,
        const Eigen::Ref<const Eigen::ArrayXd>& x, double mean = 0.0,
        double std = 1.0);

/**
 * Perform first-derivative of 1-D polynomial coeff in ascending order
 * @param[in] coef a vector of polynomial coeff in ascending order.
 * @param[in] std (optional) std value or divider for scaling input.
 * Default is 1.0 which implies no scaling.
 * @return derivative of input coeff in ascending order
 * @exception LengthError, InvalidArgument
 */
Eigen::ArrayXd polyder(
        const Eigen::Ref<const Eigen::ArrayXd>& coef, double std = 1.0);

/**
 * Function to encapsulate polyfitted outputs in isce3 Poly1d object format.
 * @param[in] x eigen vector of inputs
 * @param[in] y eigen vector of "f(x)" and shall have the same size as "x"
 * @param[in] deg order of polynomial.
 * @param[in] center_scale if true will centerize and rescale "x" by its
 * mean and std, respectively, prior to solving for coeffs.
 * @return isce3 Poly1d object
 * @exception LengthError, InvalidArgument
 * @see polyfit()
 */
isce3::core::Poly1d polyfitObj(const Eigen::Ref<const Eigen::ArrayXd>& x,
        const Eigen::Ref<const Eigen::ArrayXd>& y, int deg,
        bool center_scale = false);

}} // namespace isce3::math
