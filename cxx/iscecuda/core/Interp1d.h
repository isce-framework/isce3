#pragma once

#include <isce/core/Common.h>

namespace isce { namespace cuda { namespace core {

/**
 * Interpolate sequence x at point t
 *
 * Sequence x will be addressed as x[i*stride] for 0 <= i < length.
 *
 * @param[in] kernel    Kernel function to use for interpolation.
 * @param[in] x         Sequence to interpolate.
 * @param[in] length    Length of sequence.
 * @param[in] stride    Stride between elements of sequence.
 * @param[in] t         Desired time sample (0 <= t <= x.size()-1).
 * @param[in] periodic  Use periodic boundary condition.  Default = false.
 * @returns Interpolated value or 0 if kernel would run off array.
 */
template<class Kernel, typename T>
CUDA_HOSTDEV T interp1d(const Kernel& kernel, const T* x, size_t length,
                        size_t stride, double t, bool periodic = false);

}}} // namespace isce::cuda::core

#include "Interp1d.icc"
