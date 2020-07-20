#pragma once

#include <thrust/complex.h>

#include <isce3/core/Common.h>
#include <isce3/math/ComplexMultiply.h>

/** Promotion rules so T * complex<U> works for {T,U} in {float,double}. */
namespace isce { namespace math { namespace complex_multiply {

CUDA_HOSTDEV inline auto operator*(float t, thrust::complex<double> u)
{
    return double(t) * u;
}

CUDA_HOSTDEV inline auto operator*(thrust::complex<double> t, float u)
{
    return t * double(u);
}

}}} // namespace isce::math::complex_multiply
