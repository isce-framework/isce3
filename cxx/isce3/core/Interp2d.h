#pragma once

#include "forward.h"

namespace isce3 { namespace core {

/** Interpolate Matrix z at point (x,y)
 *
 * @tparam KernelType kernel element type
 * @tparam DataType data element type
 *
 * @param[in] kernelx   Kernel function to use for interpolation in x direction
 * @param[in] kernely   Kernel function to use for interpolation in y direction
 * @param[in] z         Matrix to interpolate.
 * @param[in] nx        Number of x samples.
 * @param[in] stridex   Stride between x samples.
 * @param[in] ny        Number of y samples.
 * @param[in] stridex   Stride between y samples.
 * @param[in] x         Desired sample (0 <= x < nx).
 * @param[in] y         Desired sample (0 <= y < ny).
 * @param[in] periodic  Use periodic boundary condition.  Default = false.
 * @returns Interpolated value or 0 if kernel would run off array.
 *
 * Matrix z will be addressed as z[ix * stridex + iy * stridey] for
 * 0 <= ix < nx and 0 <= iy < ny.
 */
template<typename KernelType, typename DataType>
DataType interp2d(const Kernel<KernelType>& kernelx,
        const Kernel<KernelType>& kernely, const DataType* z, size_t nx,
        size_t stridex, size_t ny, size_t stridey, double x, double y,
        bool periodic = false);

}} // namespace isce3::core

#include "Interp2d.icc"
