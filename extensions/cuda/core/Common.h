#pragma once

/*
 * Try to hide CUDA-specific declarations from the host compiler by
 * recognizing device-only specifiers and disabling them with attributes
 */

// Clang supports the enable_if attribute, so we
// can hard-code false and provide an error message.
#if defined(__clang__)
#define ATTRIB_DISABLE(msg) __attribute(( enable_if(0, msg) ))
// Otherwise, fall back to the error attribute.
// GCC uses this attribute to abort compilation with an error message.
// Compilers without support should nonetheless display a warning.
#else
#define ATTRIB_DISABLE(msg) __attribute(( error(msg) ))
#endif

// Error message for specifiers not linkable against host code
#define SPEC_MSG(spec) ("cannot call a " spec " from host code")

#ifdef __CUDACC__
#define CUDA_HOST    __host__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV              __device__
#define CUDA_GLOBAL  __global__
#else
// Pass valid host functions through
#define CUDA_HOST
#define CUDA_HOSTDEV
// Prevent linking to device-only functions
#define CUDA_DEV    ATTRIB_DISABLE(SPEC_MSG("__device__ function"))
#define CUDA_GLOBAL ATTRIB_DISABLE(SPEC_MSG("__global__ function (i.e. kernel)"))
#endif
