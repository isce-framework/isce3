#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <pyre/journal.h>

#include <isce3/cuda/except/Error.h>
#include <isce3/except/Error.h>

#include "ProjectionBaseHandle.h"

namespace isce3::cuda::core {

using isce3::cuda::core::ProjectionBase;

__global__ void init_proj(
        ProjectionBase** proj, int epsg_code, bool* proj_invalid)
{

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*proj) = isce3::cuda::core::createProj(epsg_code);
        if (!*proj)
            *proj_invalid = true;
    }
}

__global__ void finalize_proj(ProjectionBase** proj)
{

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *proj;
    }
}

ProjectionBaseHandle::ProjectionBaseHandle(int epsg)
{
    checkCudaErrors(cudaMalloc(&_proj, sizeof(ProjectionBase**)));

    thrust::device_vector<bool> d_proj_invalid(1, false);
    init_proj<<<1, 1>>>(_proj, epsg, d_proj_invalid.data().get());
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    bool proj_invalid = d_proj_invalid[0];
    if (proj_invalid) {
        pyre::journal::error_t error(
                "isce.cuda.core.ProjectionBaseHandle.ProjectionBaseHandle");
        error << "Unsupported EPSG provided." << pyre::journal::endl;
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Unsupported ESPG provided.");
    }
}

ProjectionBaseHandle::~ProjectionBaseHandle()
{
    finalize_proj<<<1, 1>>>(_proj);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(_proj));
}

} // namespace isce3::cuda::core
