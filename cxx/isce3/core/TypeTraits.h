#pragma once

#include <complex>
#include <type_traits>

namespace isce3 {

template<class> struct is_floating_or_complex                  : public std::false_type {};
template<> struct is_floating_or_complex<float>                : public std::true_type {};
template<> struct is_floating_or_complex<double>               : public std::true_type {};
template<> struct is_floating_or_complex<std::complex<float>>  : public std::true_type {};
template<> struct is_floating_or_complex<std::complex<double>> : public std::true_type {};
// (we currently don't allow long-doubles)

template<class T>
static constexpr bool is_floating_or_complex_v = is_floating_or_complex<T>::value;


template<class T> using enable_if_floating_or_complex =
        std::enable_if<is_floating_or_complex_v<T>, T>;
template<class T> using enable_if_floating_or_complex_t =
        typename enable_if_floating_or_complex<T>::type;

template <typename T> struct real { using type = T; };
template <typename T> struct real <std::complex<T>> { using type = T; };

template <typename T> struct complx { using type = std::complex<T>; };
template <typename T> struct complx <std::complex<T>> { using type = std::complex<T>; };

template<typename T>
struct is_complex : std::false_type {};
template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};
template<typename T>
static constexpr bool is_complex_v()
{
    return is_complex<T>::value;
}

} // namespace isce3
