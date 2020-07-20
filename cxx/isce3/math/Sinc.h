#pragma once

#include <isce3/core/Common.h>

namespace isce { namespace math {

/** sinc function defined as \f$ \frac{\sin(\pi x)}{\pi x} \f$ */
template<typename T>
CUDA_HOSTDEV
T sinc(T t);

}}

#include "Sinc.icc"
