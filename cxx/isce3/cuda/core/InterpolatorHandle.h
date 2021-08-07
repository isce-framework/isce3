#pragma once

#include <isce3/core/Constants.h>
#include <isce3/cuda/core/gpuInterpolator.h>

namespace isce3::cuda::core {

/** Class that handles device gpuInterpolator double pointers on device.
 *
 * This handle class ensures that pointers are properly allocated
 * and deallocated.
 *
 */
template<class T>
class InterpolatorHandle {
private:
    // double pointer to gpuInterpolator on device
    // 1st pointer is the gpuInterpolator location on device
    // 2nd pointer is the gpuInterpolator object on device
    isce3::cuda::core::gpuInterpolator<T>** _interp = nullptr;

public:
    /** Class constructor. Mallocs 1st pointer and creates gpuInterpolator
     * object on device.
     */
    InterpolatorHandle(isce3::core::dataInterpMethod interp_method);

    /** Destructor that frees and deletes pointers accordingly. */
    ~InterpolatorHandle();

    /** Disabling copy constructor and assignment operator to prever misuse */
    InterpolatorHandle(const InterpolatorHandle&) = delete;
    InterpolatorHandle& operator=(const InterpolatorHandle&) = delete;

    isce3::cuda::core::gpuInterpolator<T>** getInterp() const
    {
        return _interp;
    };
};
} // namespace isce3::cuda::core
