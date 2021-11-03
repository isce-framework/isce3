#include "RootFind1dSecant.h"

#include <cmath>

namespace isce3 { namespace math {

// regular methods
typename RootFind1dSecant::tuple4_t RootFind1dSecant::root(
        const func_t& f, double x0, double x1) const
{
    // if optional x_tol exists run stricter Secant approach
    if (x_tol)
        return _secant_strict(f, x0, x1);
    // otherwise run regular Secant
    return _secant(f, x0, x1);
}

// helper functions
typename RootFind1dSecant::tuple4_t RootFind1dSecant::_secant(
        const func_t& f, double x0, double x1) const
{
    // setting initial values and iterations
    double x {0.0};
    int n_itr {0};
    auto f0 = f(x0);
    auto f1 = f(x1);
    do {
        auto df = f1 - f0;
        if (std::abs(df) > 0.0)
            x = (x0 * f1 - x1 * f0) / df;
        else
            x = x1;
        // update two initial guesses
        x0 = x1;
        x1 = x;
        f0 = f1;
        f1 = f(x1);
        ++n_itr;
    } while (std::abs(f1) > f_tol && n_itr < max_iter);
    bool flag {(std::abs(f1) <= f_tol && n_itr <= max_iter)};
    return {x1, f1, flag, n_itr};
}

typename RootFind1dSecant::tuple4_t RootFind1dSecant::_secant_strict(
        const func_t& f, double x0, double x1) const
{
    // setting initial values and iterations
    double x {0.0};
    int n_itr {0};
    double dx {};
    auto f0 = f(x0);
    auto f1 = f(x1);
    do {
        auto df = f1 - f0;
        if (std::abs(df) > 0.0)
            x = (x0 * f1 - x1 * f0) / df;
        else
            x = x1;
        dx = std::abs(x - x1);
        // update two initial guesses
        x0 = x1;
        x1 = x;
        f0 = f1;
        f1 = f(x1);
        ++n_itr;
    } while ((std::abs(f1) > f_tol || dx > *x_tol) && n_itr < max_iter);
    bool flag {(std::abs(f1) <= f_tol && dx <= *x_tol && n_itr <= max_iter)};
    return {x1, f1, flag, n_itr};
}

}} // namespace isce3::math
