#pragma once

#include <isce3/math/complexOperations.h>

#include "../Kernels.h"

namespace isce3::core::detail {

/** Get interpolator coefficents for a given offset.
 *
 * @param[in]  kernel Kernel function to use for interpolation.
 * @param[in]  t      Desired time sample (0 <= t < array_size).
 * @param[out] low    Offset in input array where to apply coeffs.
 * @param[out] coeffs Interpolator coeffs, size >= ceil(kernel.width())
 *
 * Beware! This is a low-level and unsafe interface mostly intended to help
 * implement higher-dimensional interpolation.  Behavior is undefined if low is
 * nullptr or coeffs isn't long enough.
 *
 * Interpolated value can be calculated like x[low:low + N].dot(coeffs).
 */
template<typename KernelType>
void interp1d_coeffs(const Kernel<KernelType>& kernel, const double t,
        long* low, KernelType coeffs[])
{
    int width = int(ceil(kernel.width()));
    long i0 = 0;
    if (width % 2 == 0) {
        i0 = static_cast<long>(ceil(t));
    } else {
        i0 = static_cast<long>(round(t));
    }
    *low = i0 - width / 2; // integer division implicit floor()
    for (int i = 0; i < width; ++i) {
        double ti = i + (*low) - t;
        coeffs[i] = kernel(ti);
    }
}

/** Return a pointer to a contiguous block of memory for a given selection,
 *  making a copy only if necessary.
 *
 * @param[out] block    Buffer used for storage if copy is required, must
 *                      be large enough to hold `width` elements.
 * @param[in]  width    Size of selection.
 * @param[in]  low      Index to start of selection.
 * @param[in]  data     Buffer to take selection from.
 * @param[in]  size     Size of buffer (number of elements).
 * @param[in]  stride   Stride betwen elements in data buffer.
 * @param[in]  periodic Whether to use periodic boundary condition.
 *
 * @returns Pointer to contiguous selection in `data` or to `block` if copied.
 */
template<typename DataType>
const DataType* get_contiguous_view_or_copy(DataType block[], int width,
        long low, const DataType* data, size_t size, size_t stride,
        bool periodic)
{
    const long high = low + width;
    if ((stride == 1) and (low >= 0) and (high < size)) {
        return &data[low];
    }
    // else
    if (periodic) {
        for (int i = 0; i < width; ++i) {
            long j = ((low + i) % size) * stride;
            block[i] = data[j];
        }
    } else {
        for (int i = 0; i < width; ++i) {
            long j = (low + i) * stride;
            if ((j >= 0) and (j < size)) {
                block[i] = data[j];
            } else {
                block[i] = static_cast<DataType>(0);
            }
        }
    }
    return block;
}

/** Compute inner product of two contiguous buffers of different element types.
 *
 * @param[in] width     Number of elements in each buffer.
 * @param[in] x         First buffer.
 * @param[in] y         Second buffer.
 *
 * @returns x.dot(y)
 */
template<typename TX, typename TY>
auto inner_product(const int width, const TX x[], const TY y[])
{
    using namespace isce3::math::complex_operations;
    using TO = typename std::common_type<TX, TY>::type;
    TO sum = 0;

    // use SIMD instructions for the loop if possible
    // seems dumb, but this custom reduction is required for complex
    #pragma omp declare reduction(cpxsum:TO : omp_out += omp_in) \
        initializer(omp_priv = 0)

    #pragma omp simd reduction(cpxsum : sum)
    for (int i = 0; i < width; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

} // namespace isce3::core::detail
