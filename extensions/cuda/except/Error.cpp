#include <cuda_runtime.h>
#include <isce/cuda/except/Error.h>

using namespace isce::cuda::except;

template<>
CudaError<cudaError_t>::CudaError(const SrcInfo& info, const cudaError_t err) :
        err(err),
        Error(info, std::string("cudaError " +
                                std::to_string(err) + " (" +
                                cudaGetErrorString(err) + ")")) {}
