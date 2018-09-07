//
// Author: Bryan Riel
// Copyright 2018
//

#ifndef ISCE_CUDA_GEOMETRY_GPUGEOMETRY_H
#define ISCE_CUDA_GEOMETRY_GPUGEOMETRY_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#endif

#include <cmath>

// isce::core
#include "isce/core/Ellipsoid.h"
#include "isce/core/Orbit.h"
#include "isce/core/Poly2d.h"

// isce::product
#include "isce/product/ImageMode.h"

// isce::cuda::core
#include "isce/cuda/core/gpuEllipsoid.h"
#include "isce/cuda/core/gpuOrbit.h"
#include "isce/cuda/core/gpuPoly2d.h"
#include "isce/cuda/core/gpuLinAlg.h"

// isce::cuda::product
#include "isce/cuda/product/gpuImageMode.h"

// Declaration
namespace isce {
namespace cuda {
namespace geometry {

    // geo->radar
    CUDA_DEV int geo2rdr(double *,
                         const isce::cuda::core::gpuEllipsoid &,
                         const isce::cuda::core::gpuOrbit &,
                         const isce::cuda::core::gpuPoly2d &,
                         const isce::cuda::product::gpuImageMode &,
                         double *, double *,
                         double, int, double);

    // Host geo->radar to test underlying functions in a single-threaded context
    CUDA_HOST int geo2rdr_h(const cartesian_t &,
                            const isce::core::Ellipsoid &,
                            const isce::core::Orbit &,
                            const isce::core::Poly2d &,
                            const isce::product::ImageMode &,
                            double &, double &,
                            double, int, double);

} // namespace geometry
} // namespace cuda
} // namespace isce

#endif

// end of file
