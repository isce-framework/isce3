#pragma once

#include <complex>
#include <type_traits>

namespace isce {

template<class> struct is_floating_or_complex                  : public std::false_type {};
template<> struct is_floating_or_complex<float>                : public std::true_type {};
template<> struct is_floating_or_complex<double>               : public std::true_type {};
template<> struct is_floating_or_complex<std::complex<float>>  : public std::true_type {};
template<> struct is_floating_or_complex<std::complex<double>> : public std::true_type {};
// (we currently don't allow long-doubles)

template<class T>
inline constexpr bool is_floating_or_complex_v = is_floating_or_complex<T>::value;

template<class T> using enable_if_floating_or_complex =
        std::enable_if<is_floating_or_complex_v<T>, T>;
template<class T> using enable_if_floating_or_complex_t =
        typename enable_if_floating_or_complex<T>::type;

} // namespace isce
