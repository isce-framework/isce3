#pragma once

#include <thrust/complex.h>

#include <isce3/core/Common.h>
#include <isce3/math/complexOperations.h>

/** Promotion rules so T * complex<U> works for {T,U} in {float,double}. */
namespace isce3 { namespace cuda { namespace math { namespace complex_operations {

template<typename T1, typename T2>
CUDA_HOSTDEV inline auto operator*(T1 t, thrust::complex<T2> u)
{
    using U = typename std::common_type_t<T1, T2>;
    return U(t) * thrust::complex<U>(u);
}

template<typename T1, typename T2>
CUDA_HOSTDEV inline auto operator*(thrust::complex<T1> t, T2 u)
{
    using U = typename std::common_type_t<T1, T2>;
    return thrust::complex<U>(t) * U(u);
}

template<typename T1, typename T2>
CUDA_HOSTDEV inline auto operator/(T1 t, thrust::complex<T2> u)
{
    using U = typename std::common_type_t<T1, T2>;
    return U(t) / thrust::complex<U>(u);
}

template<typename T1, typename T2>
CUDA_HOSTDEV inline auto operator/(thrust::complex<T1> t, T2 u)
{
    using U = typename std::common_type_t<T1, T2>;
    return thrust::complex<U>(t) / U(u);
}

}}}} // namespace isce3::cuda::math::complex_operations
