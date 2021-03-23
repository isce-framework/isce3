#pragma once

namespace isce3 { namespace geocode {


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

template<typename T, typename T_out>
void _convertToOutputType(T a, T_out& b)
{
    b = a;
}

template<typename T, typename T_out>
void _convertToOutputType(std::complex<T> a, T_out& b)
{
    b = std::norm(a);
}

template<typename T, typename T_out>
void _convertToOutputType(std::complex<T> a, std::complex<T_out>& b)
{
    b = a;
}

template<typename T, typename T_out>
void _accumulate(T_out& band_value, T a, double b)
{
    if (b == 0)
        return;
    T_out a2;
    _convertToOutputType(a, a2);
    band_value += a2 * b;
}

}}
