#pragma once

#include <complex>

namespace isce3 { namespace math { namespace complex_operations {

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

template<typename T1, typename T2>
auto operator/(const std::complex<T1>& lhs, const T2& rhs)
{
    using U = typename std::common_type_t<T1, T2>;
    return std::complex<U>(lhs) / U(rhs);
}

template<typename T1, typename T2>
auto operator/(const T1& lhs, const std::complex<T2>& rhs)
{
    using U = typename std::common_type_t<T1, T2>;
    return U(lhs) / std::complex<U>(rhs);
}

/** Calculate exp(1j * phase) */
template <typename T>
inline constexpr std::complex<T> unitPhasor(const T phase)
{
    // With gcc -O2 this compiles to sincos or sincosf, see
    // https://godbolt.org/z/as6jz3nxT
    return std::complex<T>(std::cos(phase), std::sin(phase));
}

}}}
