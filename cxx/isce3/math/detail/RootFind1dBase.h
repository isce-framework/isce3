// Base class for all 1-D Root finding classes
#pragma once

#include <functional>
#include <optional>

#include <isce3/core/Poly1d.h>
#include <isce3/except/Error.h>

namespace isce3 { namespace math { namespace detail {

/**
 * A base class used in other derived classes to solve 1-D (single variable)
 * equation f(x)=0 with at least one real-value root/solution.
 * Each derived class must represent a unique solver/method.
 * All derived class must have a method called "root" returning
 * a tuple of four scalars:
 * {solution, function value, convergence flag, number of iterations}.
 */
class RootFind1dBase {

public:
    // constructors
    /**
     * A default constructor with absolute tolerances for "x" and "f(x)"
     * plus max number of iterations.
     * @param[in] f_tol (optional) absolute tolerance for function eval
     * "f(x)". Default is 1e-5.
     * @param[in] max_iter (optional) max number of iterations.
     * Default is 20.
     * @param[in] x_tol (optional) absolute tolerance for function variable "x".
     * If not specified or set to {} or std::nullopt, it will be ignored,
     * otherwise it will be an extra layer of tolerance checking on top of
     * "f_val"!
     * @exception InvalidArgument
     */
    RootFind1dBase(double f_tol = 1e-5, int max_iter = 20,
            std::optional<double> x_tol = {})
        : f_tol {f_tol}, max_iter {max_iter}, x_tol {x_tol}
    {
        if (max_iter < 1)
            throw isce3::except::InvalidArgument(
                    ISCE_SRCINFO(), "Max number of iterations must be >=1!");
        if (!(f_tol > 0.0))
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                    "Tolerance for function value must be positive!");
        if (x_tol)
            if (!(*x_tol > 0.0))
                throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                        "Tolerance for function variable must be positive!");
    }

    /**
     * A constructor with max number of iterations.
     * @param[in] max_iter max number of iterations.
     * Note that the absolute tolerance for the function value is 1e-5.
     * @exception InvalidArgument
     */
    RootFind1dBase(int max_iter) : RootFind1dBase(1e-5, max_iter, {}) {}

    // Abstract virtual method needed  by all derived class
    // virtual std::tuple<double, double, bool, int> root(arg, ...) const = 0;

    // non-virtual public methods

    /**
     * Get the absolute tolerance for function value
     * @return tolerance
     */
    double func_tol() const { return f_tol; }

    /**
     * Get max number of iteration being set
     * @return number of iterations
     */
    int max_num_iter() const { return max_iter; }

    /**
     * Get the absolute tolerance for function variable "x" if set
     * It returns std::nullopt if "x_tol" is not specified at object creation.
     * @return optional double precision variable tolerance.
     */
    std::optional<double> var_tol() const
    {
        return x_tol ? *x_tol : std::optional<double> {};
    }

    /**
     * Convert isce3 Poly1d object into a single-variavle function object "f(x)"
     * @param[in] f isce3 Poly1d object
     * @return single-variable function object
     */
    static std::function<double(double)> poly2func(const isce3::core::Poly1d& f)
    {
        return [=](double x) { return f.eval(x); };
    }

    // members
protected:
    double f_tol;
    int max_iter;
    std::optional<double> x_tol;
};

}}} // namespace isce3::math::detail
