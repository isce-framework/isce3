//
// Author: Bryan Riel
// Copyright 2018
//

#ifndef ISCE_CUDA_GEOMETRY_GPUGEOMETRY_H
#define ISCE_CUDA_GEOMETRY_GPUGEOMETRY_H

#include <cmath>

// isce::core
#include <isce/core/Ellipsoid.h>
#include <isce/core/Orbit.h>
#include <isce/core/Pixel.h>
#include <isce/core/LUT1d.h>

// isce::geometry
#include <isce/geometry/DEMInterpolator.h>

// isce::cuda::core
#include <isce/cuda/core/gpuOrbit.h>
#include <isce/cuda/core/gpuLUT1d.h>

// isce::cuda::geometry
#include <isce/cuda/geometry/gpuDEMInterpolator.h>

// Declaration
namespace isce {
namespace cuda {
namespace geometry {

    /** Radar geometry coordinates to map coordinates transformer*/
    CUDA_DEV int rdr2geo(double, double, double,
                         const isce::cuda::core::gpuOrbit &,
                         const isce::core::Ellipsoid &,
                         const gpuDEMInterpolator &,
                         isce::core::Vec3&, double, int, double, int, int);

    /** Radar geometry coordinates to map coordinates transformer*/
    CUDA_DEV int rdr2geo(const isce::core::Pixel &,
                         const isce::core::Basis &,
                         const isce::core::Vec3& pos,
                         const isce::core::Vec3& vel,
                         const isce::core::Ellipsoid &,
                         const gpuDEMInterpolator &,
                         isce::core::Vec3&, int, double, int, int);

    /** Map coordinates to radar geometry coordinates transformer*/
    CUDA_DEV int geo2rdr(const isce::core::Vec3&,
                         const isce::core::Ellipsoid&,
                         const isce::cuda::core::gpuOrbit &,
                         const isce::cuda::core::gpuLUT1d<double> &,
                         double *, double *,
                         double, double, int, double);

    /** Radar geometry coordinates to map coordinates transformer (host testing) */
    CUDA_HOST int rdr2geo_h(const isce::core::Pixel &,
                            const isce::core::Basis &,
                            const isce::core::Vec3& pos,
                            const isce::core::Vec3& vel,
                            const isce::core::Ellipsoid &,
                            isce::geometry::DEMInterpolator &,
                            cartesian_t &,
                            int, double, int, int);

    /** Map coordinates to radar geometry coordinates transformer (host testing) */
    CUDA_HOST int geo2rdr_h(const cartesian_t&,
                            const isce::core::Ellipsoid &,
                            const isce::core::Orbit &,
                            const isce::core::LUT1d<double> &,
                            double &, double &,
                            double, double, int, double);

} // namespace geometry
} // namespace cuda
} // namespace isce

/** Create ProjectionBase pointer on the device (meant to be run by a single thread) */
CUDA_GLOBAL void createProjection(isce::cuda::core::ProjectionBase **, int); 

/** Delete ProjectionBase pointer on the device (meant to be run by a single thread) */
CUDA_GLOBAL void deleteProjection(isce::cuda::core::ProjectionBase **);

#endif

// end of file
