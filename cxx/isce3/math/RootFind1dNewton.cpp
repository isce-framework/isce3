#include "RootFind1dNewton.h"

#include <cmath>

#include <isce3/except/Error.h>

namespace isce3 { namespace math {

// regular methods
typename RootFind1dNewton::tuple4_t RootFind1dNewton::root(
        const poly1d_t& f, double x0) const
{
    // check the order/degree of the polynomial
    if (f.order <= 1)
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Polynomial degree must be at least 1!");
    return root(poly2func(f), poly2func(f.derivative()), x0);
}

typename RootFind1dNewton::tuple4_t RootFind1dNewton::root(
        const func2_t& f, double x0) const
{
    // if optional x_tol exists run stricter Newton approach
    if (x_tol)
        return _newton_strict(f, x0);
    // otherwise run regular Newton
    return _newton(f, x0);
}

typename RootFind1dNewton::tuple4_t RootFind1dNewton::root(
        const func_t& f, const func_t& f_der, double x0) const
{
    // create a single function from two functions (f, f_der)
    func2_t f2 = [=](double x) { return std::make_tuple(f(x), f_der(x)); };
    // call the other overloaded root method
    return root(f2, x0);
}

// helper functions
typename RootFind1dNewton::tuple4_t RootFind1dNewton::_newton(
        const func2_t& f, double x0) const
{
    // setting initial values and iterations
    auto x1 = x0;
    int n_itr {0};
    auto [f_val, fp_eval] = f(x1);
    do {
        if (std::abs(fp_eval) > 0.0)
            x1 -= f_val / fp_eval;
        std::tie(f_val, fp_eval) = f(x1);
        ++n_itr;
    } while (std::abs(f_val) > f_tol && n_itr < max_iter);
    bool flag {(std::abs(f_val) <= f_tol && n_itr <= max_iter)};
    return {x1, f_val, flag, n_itr};
}

typename RootFind1dNewton::tuple4_t RootFind1dNewton::_newton_strict(
        const func2_t& f, double x0) const
{
    int n_itr {0};
    auto x = x0;
    auto [f_val, fp_eval] = f(x);
    double dx;
    do {
        dx = (std::abs(fp_eval) > 0.0) ? -f_val / fp_eval : 0.0;
        x += dx;
        std::tie(f_val, fp_eval) = f(x);
        ++n_itr;
    } while ((std::abs(f_val) > f_tol || dx > *x_tol) && n_itr < max_iter);
    bool flag {(std::abs(f_val) <= f_tol && dx <= *x_tol && n_itr <= max_iter)};
    return {x, f_val, flag, n_itr};
}

}} // namespace isce3::math
