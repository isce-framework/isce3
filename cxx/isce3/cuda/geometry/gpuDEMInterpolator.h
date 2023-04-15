//
// Author: Bryan Riel
// Copyright 2018
//

#pragma once

#include "forward.h"
#include <isce3/geometry/forward.h>
#include <isce3/cuda/core/forward.h>

#include <isce3/cuda/core/gpuProjections.h>

// DEMInterpolator declaration
class isce3::cuda::geometry::gpuDEMInterpolator {

    typedef isce3::core::Vec3 Vec3;

    public:
        /** Default constructor .*/
        CUDA_HOSTDEV inline gpuDEMInterpolator() :
            _haveRaster(false), _refHeight(0.0), _interpMethod(isce3::core::BILINEAR_METHOD),
            _xstart(0.0), _ystart(0.0), _deltax(1.0), _deltay(1.0), _owner(false) {}

        /** Constructor with a constant height .*/
        CUDA_HOSTDEV inline gpuDEMInterpolator(float height) :
            _haveRaster(false), _refHeight(height), _interpMethod(isce3::core::BILINEAR_METHOD),
            _xstart(0.0), _ystart(0.0), _deltax(1.0), _deltay(1.0), _owner(false) {}

        /** Destructor. */
        ~gpuDEMInterpolator();

        /** Copy constructor from CPU DEMInterpolator. */
        CUDA_HOST gpuDEMInterpolator(const isce3::geometry::DEMInterpolator&);

        /** Copy constructor on device. */
        CUDA_HOSTDEV gpuDEMInterpolator(gpuDEMInterpolator &);

        /** Interpolate at a given longitude and latitude. */
        CUDA_DEV float interpolateLonLat(double lon, double lat) const;

        /** Interpolate at native XY coordinates of DEM. */
        CUDA_DEV float interpolateXY(double x, double y) const;

        /** Middle X and Y coordinates. */
        CUDA_DEV inline double midX() const { return _xstart + 0.5*_width*_deltax; }
        CUDA_DEV inline double midY() const { return _ystart + 0.5*_length*_deltay; }

        /** Middle lat/lon/refHeight. */
        CUDA_DEV Vec3 midLonLat() const;

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

        /** Interpolator method. */
        CUDA_HOSTDEV inline isce3::core::dataInterpMethod interpMethod() const {
            return _interpMethod;
        }

        /** Pointer to ProjectionBase pointer. */
        CUDA_HOSTDEV inline isce3::cuda::core::ProjectionBase ** proj() const {
            return _proj;
        }

        /** Pointer to gpuInterpolator pointer. */
        CUDA_HOSTDEV inline isce3::cuda::core::gpuInterpolator<float> ** interp() const {
            return _interp;
        }

        // Make DEM pointer data public for now
        float * _dem;

    private:
        // Flag indicating whether we have access to a DEM raster
        bool _haveRaster;
        // Constant value if no raster is provided
        float _refHeight;
        // Pointers to ProjectionBase
        int _epsgcode;
        isce3::cuda::core::ProjectionBase ** _proj;
        // Pointer to an Interpolator
        isce3::core::dataInterpMethod _interpMethod;
        isce3::cuda::core::gpuInterpolator<float> ** _interp;
        //// 2D array for storing DEM subset
        //isce3::core::Matrix<float> _dem;
        // DEM dimensions
        size_t _length, _width;
        // Starting x/y for DEM subset and spacing
        double _xstart, _ystart, _deltax, _deltay;
        // Boolean for owning memory
        bool _owner;

        /** Initialize projection and interpolation objects on device. */
        CUDA_HOST void initProjInterp();

        /** Finalize/delete projection and interpolation objects on device. */
        CUDA_HOST void finalizeProjInterp();
};
