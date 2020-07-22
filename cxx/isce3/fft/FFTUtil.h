#pragma once

#include <cstdint>
#include <type_traits>

namespace isce3 { namespace fft {

/**
 * Return the smallest power of two greater than or equal to the input.
 *
 * The argument is expected to be integral and non-negative.
 */
template<typename T, typename std::enable_if<std::is_integral<T>::value>::type * = nullptr>
T nextPowerOfTwo(T);

/** Return an integer m >= n well suited to FFT sizes.
 *
 * Specifically, return the smallest integer
 * \f$ m = 2^a \cdot 3^b \cdot 5^c \geq n \f$
 * where (a,b,c) are all non-negative integers.
 */
std::int32_t nextFastPower(std::int32_t n);

}}

#define ISCE_FFT_FFTUTIL_ICC
#include "FFTUtil.icc"
#undef ISCE_FFT_FFTUTIL_ICC
