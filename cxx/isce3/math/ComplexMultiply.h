#pragma once

#include <complex>

namespace isce3 { namespace math { namespace complex_multiply {

template<typename T1, typename T2>
auto operator*(const std::complex<T1>& lhs, const T2& rhs)
{
    using U = typename std::common_type_t<T1, T2>;
    return std::complex<U>(lhs) * U(rhs);
}

template<typename T1, typename T2>
auto operator*(const T1& lhs, const std::complex<T2>& rhs)
{
    using U = typename std::common_type_t<T1, T2>;
    return U(lhs) * std::complex<U>(rhs);
}

}}}
