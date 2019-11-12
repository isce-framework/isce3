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
#define CUDA_DEV    ATTRIB_DISABLE( \
    "cannot call a __device__ function directly from CPU code")
#define CUDA_GLOBAL ATTRIB_DISABLE( \
    "calling a __global__ function (i.e. kernel) " \
    "is not supported by non-CUDA compilers")
#endif

// Suppress nvcc warnings about calling a __host__ function
// from a __host__ __device__ function
#ifdef __NVCC__
#define NVCC_HD_WARNING_DISABLE _Pragma("hd_warning_disable")
#else
#define NVCC_HD_WARNING_DISABLE
#endif

// Convert s into a string literal (s is not macro-expanded first)
#define STRINGIFY(s) #s

// Hint to the compiler that the subsequent loop should be unrolled (if
// supported by the compiler). The unroll count can be optionally specified.
#if defined(__clang__) || defined(__NVCC__)
#define PRAGMA_UNROLL              _Pragma("unroll")
#define PRAGMA_UNROLL_COUNT(count) _Pragma(STRINGIFY(unroll count))
#else
#define PRAGMA_UNROLL
#define PRAGMA_UNROLL_COUNT(count)
#endif
