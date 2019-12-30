#pragma once

#include <cufft.h>
#include <thrust/complex.h>

namespace isce { namespace cuda { namespace fft { namespace detail {

template<typename T> struct CufftC2CType {};
template<>           struct CufftC2CType<float>  { constexpr static cufftType type = CUFFT_C2C; };
template<>           struct CufftC2CType<double> { constexpr static cufftType type = CUFFT_Z2Z; };

template<typename T> struct CufftR2CType {};
template<>           struct CufftR2CType<float>  { constexpr static cufftType type = CUFFT_R2C; };
template<>           struct CufftR2CType<double> { constexpr static cufftType type = CUFFT_D2Z; };

template<typename T> struct CufftC2RType {};
template<>           struct CufftC2RType<float>  { constexpr static cufftType type = CUFFT_C2R; };
template<>           struct CufftC2RType<double> { constexpr static cufftType type = CUFFT_Z2D; };

template<int Sign, typename T>
void executePlan(cufftHandle plan, void * in, void * out, cufftType type);

}}}}
