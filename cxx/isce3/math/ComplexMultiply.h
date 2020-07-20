#pragma once

#include <complex>

/** Promotion rules so T * complex<U> works for {T,U} in {float,double}. */
namespace isce3 { namespace math { namespace complex_multiply {

inline auto operator*(float t, std::complex<double> u) {
    return double(t) * u;
}

inline auto operator*(std::complex<double> t, float u) {
    return t * double(u);
}

}}}
