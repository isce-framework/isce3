#pragma once

#include <isce3/cuda/core/gpuProjections.h>

namespace isce3::cuda::core {

/** Class that handles device ProjectionBase double pointers on device.
 *
 * This handle class ensures that pointers are properly allocated
 * and deallocated.
 *
 */
class ProjectionBaseHandle {
private:
    // double pointer to Projection Base on device
    // 1st pointer is the ProjectionBase location on device
    // 2nd pointer is the ProjectionBase object on device
    isce3::cuda::core::ProjectionBase** _proj = nullptr;

public:
    /** Class constructor. Mallocs 1st pointer and creates ProjectionBase
     *  object on device.
     *
     * \param[in] epsg  EPSG of ProjectionBase to be created
     *
     * */
    ProjectionBaseHandle(int epsg);

    /** Destructor that frees and deletes pointers accordingly. */
    ~ProjectionBaseHandle();

    /** Disabling copy constructor and assignment operator to prever misuse */
    ProjectionBaseHandle(const ProjectionBaseHandle&) = delete;
    ProjectionBaseHandle& operator=(const ProjectionBaseHandle&) = delete;

    isce3::cuda::core::ProjectionBase** get_proj() const { return _proj; }
};
} // namespace isce3::cuda::core
