// 1-D Newton Root finding class
#pragma once

#include <functional>
#include <tuple>

#include <isce3/core/Poly1d.h>

#include "detail/RootFind1dBase.h"

namespace isce3 { namespace math {

/**
 * A class with overloaded method to solve 1-D (single variable)
 * equation f(x)=0 with at least one real-value root/solution via
 * Newton approach <a href="https://en.wikipedia.org/wiki/Newton's_method"
 * target="_blank"> See Newton method</a>.
 */
class RootFind1dNewton : public detail::RootFind1dBase {
    // aliases
protected:
    using func_t = std::function<double(double)>;
    using tuple4_t = std::tuple<double, double, bool, int>;
    using poly1d_t = isce3::core::Poly1d;
    using func2_t = std::function<std::tuple<double, double>(double)>;

public:
    // inherit all constructors from base class
    using detail::RootFind1dBase::RootFind1dBase;

    // methods
    /**
     * Find a root of the function "f(x)" closest to its initial value via
     * Newton approach.
     * @param[in] f isce3.core.Poly1d object
     * expressing 1-D function "f(x)" as polynomial.
     * @param[in] x0 (optional) initial guess of the "x". Default is 0.0.
     * @return solution "x"
     * @return function eval "f(x)"
     * @return convergence flag (true or false)
     * @return number of iterations
     * @exception InvalidArgument
     */
    std::tuple<double, double, bool, int> root(
            const isce3::core::Poly1d& f, double x0 = 0) const;

    /**
     * Find a root of the function "f(x)" closest to its initial value via
     * Newton approach.
     * @param[in] f single-variable function object to represent "f(x)".
     * @param[in] f_der single-variable function object to represent derivative
     * of "f(x)".
     * @param[in] x0 (optional) initial guess of the "x". Default is 0.0.
     * @return solution "x"
     * @return function eval "f(x)"
     * @return convergence flag (true or false)
     * @return number of iterations
     */
    std::tuple<double, double, bool, int> root(
            const std::function<double(double)>& f,
            const std::function<double(double)>& f_der, double x0 = 0) const;

    /**
     * Find a root of the function "f(x)" closest to its initial value via
     * Newton approach.
     * @param[in] f single-variable function object to return a tuple of "f(x)"
     * and "f_der(x)", that is both function value and its first derivative for
     * "x".
     * @param[in] x0 (optional) initial guess of the "x". Default is 0.0.
     * @return solution "x"
     * @return function eval "f(x)"
     * @return convergence flag (true or false)
     * @return number of iterations
     */
    std::tuple<double, double, bool, int> root(
            const std::function<std::tuple<double, double>(double)>& f,
            double x0 = 0) const;

private:
    /**
     * @internal
     * Overloaded helper function for Newton approach.
     * @param[in] f single-variable function object to return "f(x)" and
     * "f_der(x)", that is a pair of values of function and its first
     * derivative.
     * @param[in] x0 initial guess of the "x".
     * @return solution "x"
     * @return function eval "f(x)"
     * @return convergence flag (true or false)
     * @return number of iterations
     * @see _newton_strict()
     */
    tuple4_t _newton(const func2_t& f, double x0) const;

    /**
     * @internal
     * Overloaded helper function for Newton approach with extra tolerance
     * "x_tol" for a more strict convergence criteria.
     * @param[in] f single-variable function object to return "f(x)" and
     * "f_der(x)", that is a pair of values of function and its first
     * derivative.
     * @param[in] x0 initial guess of the "x".
     * @return solution "x"
     * @return function eval "f(x)"
     * @return convergence flag (true or false)
     * @return number of iterations
     * @see _newton()
     */
    tuple4_t _newton_strict(const func2_t& f, double x0) const;
};

}} // namespace isce3::math
