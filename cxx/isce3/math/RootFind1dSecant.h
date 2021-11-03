// 1-D Secant Root finding class
#pragma once

#include <functional>
#include <tuple>

#include "detail/RootFind1dBase.h"

namespace isce3 { namespace math {

/**
 * A class with root  method to solve 1-D (single variable)
 * equation f(x)=0 with at least one real-value root/solution via
 * Secant approach <a href="https://en.wikipedia.org/wiki/Secant_method"
 * target="_blank"> See Secant method</a>
 */
class RootFind1dSecant : public detail::RootFind1dBase {
    // aliases
protected:
    using func_t = std::function<double(double)>;
    using tuple4_t = std::tuple<double, double, bool, int>;

public:
    // inherit all constructors from base class
    using detail::RootFind1dBase::RootFind1dBase;

    /**
     * Find a root of the function "f(x)" closest to its initial values via
     * Secant approach.
     * @param[in] f single-variable function object to represent "f(x)".
     * @param[in] x0 first initial guess of the "x".
     * @param[in] x1 second guess of the "x".
     * @return solution "x"
     * @return function eval "f(x)"
     * @return convergence flag (true or false)
     * @return number of iterations
     */
    std::tuple<double, double, bool, int> root(
            const std::function<double(double)>& f, double x0, double x1) const;

private:
    /**
     * @internal
     * Secant approach with two initial values when simply f(x) is known and
     * its derivative is not available.
     * @param[in] f single-variable function object to represent "f(x)".
     * @param[in] x0 first initial guess of the "x".
     * @param[in] x1 second guess of the "x".
     * @return solution "x"
     * @return function eval "f(x)"
     * @return convergence flag (true or false)
     * @return number of iterations
     * @see _secant_strict()
     */
    tuple4_t _secant(const func_t& f, double x0, double x1) const;

    /**
     * @internal
     * A more strict Secant approach with two initial values when simply f(x) is
     * known and its derivative is not available. The second tolerance, "x_tol"
     * will be used on top of "f_tol".
     * @param[in] f single-variable function object to represent "f(x)".
     * @param[in] x0 first initial guess of the "x".
     * @param[in] x1 second guess of the "x".
     * @return solution "x"
     * @return function eval "f(x)"
     * @return convergence flag (true or false)
     * @return number of iterations
     * @see _secant()
     */
    tuple4_t _secant_strict(const func_t& f, double x0, double x1) const;
};

}} // namespace isce3::math
