#pragma once
#include <isce3/core/Common.h>

namespace isce3 { namespace math {

/** Find a zero of a function on a bracketed interval using Brent's method.
 *
 * @param[in]  a     One side of interval bracketing the root.
 * @param[in]  b     The other side of interval bracketing the root.
 * @param[in]  f     Callable function f(x).
 * @param[in]  tol   Allowable absolute uncertainty (>= 0) in the solution.
 *                   Solution will be found within `4 * eps * abs(x) + tol`
 *                   of the actual root, where eps is the relative precision.
 * @param[out] root  The solution x where f(x)==0.  The value is only updated
 *                   on success or when the allowed number of iterations has
 *                   been reached.
 *
 * @returns Zero on success, error code otherwise.
 *
 * The interval [a,b] must be chosen such that f(a) and f(b) have opposite sign.
 *
 * Number of iterations is strictly bounded by N^2 where N is the number of
 * bisection steps for a given tolerance, N = log2((b - a) / tol).  But in
 * practice the number of iterations is seldom more than 3*N.
 *
 * For "well behaved" functions (e.g., C1 continuous around a simple root)
 * convergence is superlinear (q > 1.6) and therefore much faster than
 * bisection (q=1).  Assuming computing the derivative f'(x) takes about as much
 * effort as computing f(x), then this is also faster than Newton's method
 * (q=2 but double the work per iteration).
 *
 * @cite brent
 * https://maths-people.anu.edu.au/~brent/pd/rpb011a.pdf
 */
template<typename T, typename Func>
CUDA_HOSTDEV isce3::error::ErrorCode
find_zero_brent(T a, T b, Func f, const T tol, T* root);


/** Find a zero of a function on a bracketed interval using bisection.
 *
 * @param[in]  a     One side of interval bracketing the root.
 * @param[in]  b     The other side of interval bracketing the root.
 * @param[in]  f     Callable function f(x).
 * @param[in]  tol   Allowable absolute uncertainty (> 0) in the solution.
 *                   Solution will be found within `eps * abs(x) + tol`
 *                   of the actual root, where eps is the relative precision.
 * @param[out] root  The solution x where f(x)==0.  The value is only updated
 *                   when the allowed number of iterations has been reached.
 *
 * @returns Zero on success, error code otherwise.
 *
 * The interval [a,b] must be chosen such that f(a) and f(b) have opposite sign.
 */
template<typename T, typename Func>
CUDA_HOSTDEV isce3::error::ErrorCode
find_zero_bisection(T a, T b, Func f, T tol, T* root);


/** Find a zero using a fixed number of bisection steps.
 * @see find_zero_bisection
 */
template<typename T, typename Func>
CUDA_HOSTDEV isce3::error::ErrorCode
find_zero_bisection_iter(T a, T b, Func f, int niter, T* root);


}} // namespace isce3::math

#include "RootFind1dBracket.icc"
