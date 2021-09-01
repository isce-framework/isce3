/** @file edge_method_cost_func.h
 * Cost functions used in EL antenna pointing estimation via edge method
 */
#pragma once

#include <optional>
#include <tuple>

#include <isce3/core/Linspace.h>
#include <isce3/core/Poly1d.h>

/** @namespace isce3::antenna */
namespace isce3 { namespace antenna {

// Aliases
using poly1d_t = isce3::core::Poly1d;

/**
 * Estimate roll angle offset via edge method from poly-fitted
 * power patterns obtained from echo raw data and antenna pattern.
 * The cost function is solved via Newton method and final solution
 * is the weighted average of individual solution within look
 * (off-nadir) angles [near, far] with desired angle precision all
 * defined by isce3 Linspace.
 * See equations for cost function in section 1.1 of the reference
 * @cite EdgeMethodElPointDoc
 * The only difference is that the look angles are in (rad) rather than in
 * (deg). Note that the respective magnitudes for both echo and antenna can be
 * either 2-way or 1-way power patterns.
 * @param[in] polyfit_echo isce3 Poly1d object for polyfitted magnitude
 * of either  range compressed (preferred) or raw echo data.
 * It must be third-order polynomial of relative magnitude/power in (dB)
 * as a function of look angle in (rad)
 * @param[in] polyfit_ant isce3 Poly1d object for antenna EL power pattern.
 * It must be third-order polynomial of relative magnitude/power in (dB)
 * as a function of look angle in (rad).
 * It must have the same mean and std as that of "polyfit_echo"!
 * @param[in] look_ang isce3 Linspace object to cover desired range of
 * look angles (rad) with desired precision/spacing.
 * @param[in] polyfit_weight (optional) isce3 Poly1d object for weightings used
 * in final weighted averaged of individual solutions over desired look angle
 * coverage. It shall represent relative magnitude/power in (dB) as a function
 * of look angle in (rad).
 * The order of the polynomial must be at least 0 (constant weights).
 * @return roll angle offset (rad)
 * Note that the roll offset shall be added to EL angles in antenna frame
 * to align EL power pattern from antenna to the one extracted from echo given
 * the cost function optimized for offset applied to polyfitted antenna data.
 * @return max cost function value among all iterations
 * @return overall convergence flag (true or false)
 * @return max number of iterations among all iterations
 * @exception InvalidArgument
 */
std::tuple<double, double, bool, int> rollAngleOffsetFromEdge(
        const poly1d_t& polyfit_echo, const poly1d_t& polyfit_ant,
        const isce3::core::Linspace<double>& look_ang,
        std::optional<poly1d_t> polyfit_weight = {});

/**
 * Estimate roll angle offset via edge method from poly-fitted
 * power patterns obtained from echo raw data and antenna pattern.
 * The cost function is solved via Newton method and final solution
 * is the weighted average of individual solution within look
 * (off-nadir) angles [near, far] with desired angle precision "look_ang_prec".
 * See equations for cost function in section 1.1 of the reference
 * @cite EdgeMethodElPointDoc
 * The only difference is that the look angles are in (rad) rather than in
 * (deg). Note that the respective magnitudes for both echo and antenna can be
 * either 2-way or 1-way power patterns.
 * @param[in] polyfit_echo isce3 Poly1d object for polyfitted magnitude
 * of either  range compressed (preferred) or raw echo data.
 * It must be third-order polynomial of relative magnitude/power in (dB)
 * as a function of look angle in (rad)
 * @param[in] polyfit_ant isce3 Poly1d object for antenna EL power pattern.
 * It must be third-order polynomial of relative magnitude/power in (dB)
 * as a function of look angle in (rad).
 * It must have the same mean and std as that of "polyfit_echo"!
 * @param[in] look_ang_near look angle for near range in (rad)
 * @param[in] look_ang_far look angle for far range in (rad)
 * @param[in] look_ang_prec look angle precision/resolution in (rad)
 * @param[in] polyfit_weight (optional) isce3 Poly1d object for weightings used
 * in final weighted averaged of individual solutions over desired look angle
 * coverage. It shall represent relative magnitude/power in (dB) as a function
 * of look angle in (rad).
 * The order of the polynomial must be at least 0 (constant weights).
 * @return roll angle offset (rad)
 * Note that the roll offset shall be added to EL angles in antenna frame
 * to align EL power pattern from antenna to the one extracted from echo given
 * the cost function optimized for offset applied to polyfitted antenna data.
 * @return max cost function value among all iterations
 * @return overall convergence flag (true or false)
 * @return max number of iterations among all iterations
 * @exception InvalidArgument
 */
std::tuple<double, double, bool, int> rollAngleOffsetFromEdge(
        const poly1d_t& polyfit_echo, const poly1d_t& polyfit_ant,
        double look_ang_near, double look_ang_far, double look_ang_prec,
        std::optional<poly1d_t> polyfit_weight = {});

}} // namespace isce3::antenna
