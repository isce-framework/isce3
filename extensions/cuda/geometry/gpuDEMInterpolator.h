//
// Author: Bryan Riel
// Copyright 2018
//

#ifndef ISCE_CUDA_GEOMETRY_GPUDEMINTERPOLATOR_H
#define ISCE_CUDA_GEOMETRY_GPUDEMINTERPOLATOR_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#endif

// isce::geometry
#include "isce/geometry/DEMInterpolator.h"

// Declaration
namespace isce {
    namespace cuda {
        namespace geometry {
            class gpuDEMInterpolator;
        }
    }
}

// DEMInterpolator declaration
class isce::cuda::geometry::gpuDEMInterpolator {

    public:
        /** Default constructor .*/
        CUDA_HOSTDEV inline gpuDEMInterpolator() : _haveRaster(false), _refHeight(0.0),
            _xstart(0.0), _ystart(0.0), _deltax(1.0), _deltay(1.0) {}

        /** Constructor with a constant height .*/
        CUDA_HOSTDEV inline gpuDEMInterpolator(float height) : 
            _haveRaster(false), _refHeight(height),
            _xstart(0.0), _ystart(0.0), _deltax(1.0), _deltay(1.0) {}

        /** Copy constructor from CPU DEMInterpolator. */
        CUDA_HOST gpuDEMInterpolator(const isce::geometry::DEMInterpolator &);

        /** Interpolate at a given longitude and latitude. */
        CUDA_DEV double interpolateLonLat(double lon, double lat) const;

        /** Interpolate at native XY coordinates of DEM. */
        CUDA_DEV double interpolateXY(double x, double y) const;

    private:
        // Flag indicating whether we have access to a DEM raster
        bool _haveRaster;
        // Constant value if no raster is provided
        float _refHeight;
        //// Pointer to a ProjectionBase
        //int _epsgcode;
        //isce::core::ProjectionBase * _proj;
        //// 2D array for storing DEM subset
        //isce::core::Matrix<float> _dem;
        // Starting x/y for DEM subset and spacing
        double _xstart, _ystart, _deltax, _deltay;
        // Interpolation method
        //isce::core::dataInterpMethod _interpMethod;
};

#endif

// end of file
