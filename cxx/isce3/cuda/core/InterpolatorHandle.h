#pragma once

#include <thrust/device_vector.h>

#include <isce3/cuda/core/gpuInterpolator.h>

namespace isce3::cuda::core {

// XXX A virtual base class for `InterpolatorHandle` that does not depend on
// the template parameter `T`. This hack is needed by the CUDA `Geocode` class
// in order to pass `InterpolatorHandle` objects (via pointer to the base
// class) across interfaces where the template parameter is not known at
// compile time.
class InterpolatorHandleVirtual{
public:
    virtual ~InterpolatorHandleVirtual() = default;
};

/** Class that handles device gpuInterpolator double pointers on device.
 *
 * This handle class ensures that pointers are properly allocated
 * and deallocated.
 *
 */
template<class T>
class InterpolatorHandle : public InterpolatorHandleVirtual{
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
    ~InterpolatorHandle() override;

    /** Disabling copy constructor and assignment operator to prevent misuse */
    InterpolatorHandle(const InterpolatorHandle&) = delete;
    InterpolatorHandle& operator=(const InterpolatorHandle&) = delete;

    isce3::cuda::core::gpuInterpolator<T>** getInterp() const
    {
        return _interp;
    };

    // Member specifically for sinc interpolator filter. Allows filter on
    // device memory to persist when the sinc interpolator constructor is
    // called on device.
    thrust::device_vector<double> d_sinc_filter;
};
} // namespace isce3::cuda::core
