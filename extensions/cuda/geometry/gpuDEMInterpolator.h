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

// isce::cuda::core
#include "isce/cuda/core/gpuProjections.h"

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
            _xstart(0.0), _ystart(0.0), _deltax(1.0), _deltay(1.0), _owner(false) {}

        /** Constructor with a constant height .*/
        CUDA_HOSTDEV inline gpuDEMInterpolator(float height) : 
            _haveRaster(false), _refHeight(height),
            _xstart(0.0), _ystart(0.0), _deltax(1.0), _deltay(1.0), _owner(false) {}

        /** Destructor. */
        ~gpuDEMInterpolator();

        /** Copy constructor from CPU DEMInterpolator. */
        CUDA_HOST gpuDEMInterpolator(isce::geometry::DEMInterpolator &);

        /** Copy constructor on device. */
        CUDA_HOSTDEV gpuDEMInterpolator(gpuDEMInterpolator &);

        /** Interpolate at a given longitude and latitude. */
        CUDA_DEV double interpolateLonLat(double lon, double lat) const;

        /** Interpolate at native XY coordinates of DEM. */
        CUDA_DEV double interpolateXY(double x, double y) const;

        /** Middle X and Y coordinates. */
        CUDA_DEV inline double midX() const { return _xstart + 0.5*_width*_deltax; }
        CUDA_DEV inline double midY() const { return _ystart + 0.5*_length*_deltay; }

        /** Middle lat/lon/refHeight. */
        CUDA_DEV void midLonLat(double *) const; 

        /** Get upper left X. */
        CUDA_HOSTDEV inline double xStart() const { return _xstart; }

        /** Get upper left Y. */
        CUDA_HOSTDEV inline double yStart() const { return _ystart; }

        /** Get X spacing. */
        CUDA_HOSTDEV inline double deltaX() const { return _deltax; }

        /** Get Y spacing. */
        CUDA_HOSTDEV inline double deltaY() const { return _deltay; }

        /** Flag indicating whether we have a raster. */
        CUDA_HOSTDEV inline bool haveRaster() const { return _haveRaster; }

        /** Reference height. */
        CUDA_HOSTDEV inline float refHeight() const { return _refHeight; }

        /** DEM length. */
        CUDA_HOSTDEV inline size_t length() const { return _length; }

        /** DEM width. */
        CUDA_HOSTDEV inline size_t width() const { return _width; }

        /** EPSG code. */
        CUDA_HOSTDEV inline int epsgCode() const { return _epsgcode; }

        // Make DEM pointer data public for now
        float * _dem;

        /** Set pointer to ProjectionBase pointer. */
        CUDA_HOSTDEV inline void proj(isce::cuda::core::ProjectionBase ** inputProj) {
            _proj = inputProj;
        }

    private:
        // Flag indicating whether we have access to a DEM raster
        bool _haveRaster;
        // Constant value if no raster is provided
        float _refHeight;
        // Pointer to a ProjectionBase
        int _epsgcode;
        isce::cuda::core::ProjectionBase ** _proj;
        //// 2D array for storing DEM subset
        //isce::core::Matrix<float> _dem;
        //float * _dem;
        // DEM dimensions
        size_t _length, _width;
        // Starting x/y for DEM subset and spacing
        double _xstart, _ystart, _deltax, _deltay;
        // Interpolation method
        //isce::core::dataInterpMethod _interpMethod;
        // Boolean for owning memory
        bool _owner;
};

#endif

// end of file
