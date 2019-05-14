//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CUDA_CORE_GPULINALG_H
#define ISCE_CUDA_CORE_GPULINALG_H

#include <isce/core/Common.h>

// Declaration
namespace isce {
    namespace cuda {
        namespace core {
            struct gpuLinAlg;
        }
    }
}

/** Simple linear algebra operations for triplets of double precision numbers*/
struct isce::cuda::core::gpuLinAlg {
    CUDA_HOSTDEV gpuLinAlg() = delete;

    /** Cross product*/  
    CUDA_DEV static void cross(const double *, const double *, double *);

    /** Dot product*/
    CUDA_DEV static double dot(const double *, const double *);

    /** Linear combination of vectors*/
    CUDA_DEV static void linComb(double, const double *, double, const double *, double *);

    /** Unit vector */
    CUDA_DEV static void unitVec(const double *, double *);

    /** Norm of vector */
    CUDA_DEV static double norm(const double *);

    /** Multiply all elements by a scalar value*/
    CUDA_DEV static void scale(double *, double);

    /** Matrix-vector multiplication */
    CUDA_DEV static void matVec(const double *, const double *, double *);

    /** Transpose matrix */
    CUDA_DEV static void tranMat(const double *, double *);

    /** Compute ENU basis*/
    CUDA_DEV static void enuBasis(double, double, double *);

};

#endif

// end of file
