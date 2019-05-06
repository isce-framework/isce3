#pragma once

#include <isce/except/Error.h>

namespace isce { namespace cuda { namespace except {

    using namespace isce::except;

    /** CudaError provide the same information as BaseError,
     *  and also retains the original error code.*/
    template<class T>
    struct CudaError : RuntimeError {
        const T err;
        CudaError(const SrcInfo& info, const T err);
    };

    // Instantiate the for when the error code is a cudaError_t
    // In this case can get a useful error message via cudaGetErrorString
    template<>
    CudaError<cudaError_t>::CudaError(const SrcInfo& info, const cudaError_t err);

    template<class T>
    CudaError<T>::CudaError(const SrcInfo& info, const T err) :
            err(err),
            RuntimeError(info, std::string("cudaError ") + std::to_string(err)) {}

}}}

template<class T>
static inline void checkCudaErrorsImpl(const isce::except::SrcInfo& info, T err) {
    if (err)
        throw isce::cuda::except::CudaError<T>(info, err);
}


// Wrapper to pass file name, line number, and function name
#define checkCudaErrors(val) checkCudaErrorsImpl(ISCE_SRCINFO(), val)
