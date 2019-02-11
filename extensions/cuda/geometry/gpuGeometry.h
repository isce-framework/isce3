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
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#define CUDA_GLOBAL
#endif

#include <cmath>

// isce::core
#include "isce/core/Ellipsoid.h"
#include "isce/core/Orbit.h"
#include "isce/core/LUT1d.h"

// isce::product
#include "isce/product/ImageMode.h"

// isce::geometry
#include "isce/geometry/DEMInterpolator.h"

// isce::cuda::core
#include "isce/cuda/core/gpuBasis.h"
#include "isce/cuda/core/gpuEllipsoid.h"
#include "isce/cuda/core/gpuOrbit.h"
#include "isce/cuda/core/gpuPixel.h"
#include "isce/cuda/core/gpuLUT1d.h"
#include "isce/cuda/core/gpuLinAlg.h"
#include "isce/cuda/core/gpuStateVector.h"

// isce::cuda::product
#include "isce/cuda/product/gpuImageMode.h"

// isce::cuda::geometry
#include "isce/cuda/geometry/gpuDEMInterpolator.h"

// Declaration
namespace isce {
namespace cuda {
namespace geometry {

    /** Radar geometry coordinates to map coordinates transformer*/
    CUDA_DEV int rdr2geo(double, double, double,
                         const isce::cuda::core::gpuOrbit &,
                         const isce::cuda::core::gpuEllipsoid &,
                         const gpuDEMInterpolator &,
                         double *,
                         double, int, double, int, int);

    /** Radar geometry coordinates to map coordinates transformer*/
    CUDA_DEV int rdr2geo(const isce::cuda::core::gpuPixel &,
                         const isce::cuda::core::gpuBasis &,
                         const isce::cuda::core::gpuStateVector &,
                         const isce::cuda::core::gpuEllipsoid &,
                         const gpuDEMInterpolator &,
                         double *,
                         int, double, int, int);

    /** Map coordinates to radar geometry coordinates transformer*/
    CUDA_DEV int geo2rdr(double *,
                         const isce::cuda::core::gpuEllipsoid &,
                         const isce::cuda::core::gpuOrbit &,
                         const isce::cuda::core::gpuLUT1d<double> &,
                         const isce::cuda::product::gpuImageMode &,
                         double *, double *,
                         double, int, double);

    /** Radar geometry coordinates to map coordinates transformer (host testing) */
    CUDA_HOST int rdr2geo_h(const isce::core::Pixel &,
                            const isce::core::Basis &,
                            const isce::core::StateVector &,
                            const isce::core::Ellipsoid &,
                            isce::geometry::DEMInterpolator &,
                            cartesian_t &,
                            int, double, int, int);

    /** Map coordinates to radar geometry coordinates transformer (host testing) */
    CUDA_HOST int geo2rdr_h(const cartesian_t &,
                            const isce::core::Ellipsoid &,
                            const isce::core::Orbit &,
                            const isce::core::LUT1d<double> &,
                            const isce::product::ImageMode &,
                            double &, double &,
                            double, int, double);

} // namespace geometry
} // namespace cuda
} // namespace isce

/** Create ProjectionBase pointer on the device (meant to be run by a single thread) */
CUDA_GLOBAL void createProjection(isce::cuda::core::ProjectionBase **, int); 

/** Delete ProjectionBase pointer on the device (meant to be run by a single thread) */
CUDA_GLOBAL void deleteProjection(isce::cuda::core::ProjectionBase **);

#endif

// end of file
