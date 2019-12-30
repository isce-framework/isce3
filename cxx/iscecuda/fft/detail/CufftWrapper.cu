#include "CufftWrapper.h"

#include <isce/cuda/except/Error.h>

namespace isce { namespace cuda { namespace fft { namespace detail {

template<>
void executePlan<CUFFT_FORWARD, float>(cufftHandle plan, void * in, void * out, cufftType type)
{
    if (type == CUFFT_C2C) {
        checkCudaErrors( cufftExecC2C(
                plan,
                reinterpret_cast<cufftComplex *>(in),
                reinterpret_cast<cufftComplex *>(out),
                CUFFT_FORWARD) );
    }
    else { // CUFFT_R2C
        checkCudaErrors( cufftExecR2C(
                plan,
                reinterpret_cast<cufftReal *>(in),
                reinterpret_cast<cufftComplex *>(out)) );
    }
}

template<>
void executePlan<CUFFT_FORWARD, double>(cufftHandle plan, void * in, void * out, cufftType type)
{
    if (type == CUFFT_Z2Z) {
        checkCudaErrors( cufftExecZ2Z(
                plan,
                reinterpret_cast<cufftDoubleComplex *>(in),
                reinterpret_cast<cufftDoubleComplex *>(out),
                CUFFT_FORWARD) );
    }
    else { // CUFFT_D2Z
        checkCudaErrors( cufftExecD2Z(
                plan,
                reinterpret_cast<cufftDoubleReal *>(in),
                reinterpret_cast<cufftDoubleComplex *>(out)) );
    }
}

template<>
void executePlan<CUFFT_INVERSE, float>(cufftHandle plan, void * in, void * out, cufftType type)
{
    if (type == CUFFT_C2C) {
        checkCudaErrors( cufftExecC2C(
                plan,
                reinterpret_cast<cufftComplex *>(in),
                reinterpret_cast<cufftComplex *>(out),
                CUFFT_INVERSE) );
    }
    else { // CUFFT_C2R
        checkCudaErrors( cufftExecC2R(
                plan,
                reinterpret_cast<cufftComplex *>(in),
                reinterpret_cast<cufftReal *>(out)) );
    }
}

template<>
void executePlan<CUFFT_INVERSE, double>(cufftHandle plan, void * in, void * out, cufftType type)
{
    if (type == CUFFT_Z2Z) {
        checkCudaErrors( cufftExecZ2Z(
                plan,
                reinterpret_cast<cufftDoubleComplex *>(in),
                reinterpret_cast<cufftDoubleComplex *>(out),
                CUFFT_INVERSE) );
    }
    else { // CUFFT_Z2D
        checkCudaErrors( cufftExecZ2D(
                plan,
                reinterpret_cast<cufftDoubleComplex *>(in),
                reinterpret_cast<cufftDoubleReal *>(out)) );
    }
}

}}}}
