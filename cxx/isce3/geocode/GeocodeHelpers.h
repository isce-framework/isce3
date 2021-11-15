#pragma once

#include <isce3/math/complexOperations.h>

namespace isce3 { namespace geocode {

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

    using isce3::math::complex_operations::operator*;

    T_out a2;
    _convertToOutputType(a, a2);
    band_value += a2 * b;
}

}}
