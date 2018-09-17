// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#include "gpuDEMInterpolator.h"
#include "../helper_cuda.h"

using isce::cuda::core::ProjectionBase;

// Temporary bilinear interpolation definition until Interpolator class implemented
__device__
double
bilinear(double x, double y, float * z, size_t nx) {

    int x1 = std::floor(x);
    int x2 = std::ceil(x);
    int y1 = std::floor(y);
    int y2 = std::ceil(y);
    double q11 = z[y1*nx + x1];
    double q12 = z[y2*nx + x1];
    double q21 = z[y1*nx + x2];
    double q22 = z[y2*nx + x2];

    if ((y1 == y2) && (x1 == x2)) {
        return q11;
    } else if (y1 == y2) {
        return ((x2 - x) / (x2 - x1)) * q11 +
               ((x - x1) / (x2 - x1)) * q21;
    } else if (x1 == x2) {
        return ((y2 - y) / (y2 - y1)) * q11 +
               ((y - y1) / (y2 - y1)) * q12;
    } else {
        return  ((q11 * (x2 - x) * (y2 - y)) /
                 (x2 - x1) * (y2 - y1)) +
                ((q21 * (x - x1) * (y2 - y)) /
                 (x2 - x1) * (y2 - y1)) +
                ((q12 * (x2 - x) * (y - y1)) /
                 (x2 - x1) * (y2 - y1)) +
                ((q22 * (x - x1) * (y - y1)) /
                 (x2 - x1) * (y2 - y1));
    }
}

/** @param[in] demInterp DEMInterpolator object.
  *
  * Copy DEM interpolator data to GPU device. */
__host__
isce::cuda::geometry::gpuDEMInterpolator::
gpuDEMInterpolator(isce::geometry::DEMInterpolator & demInterp) :
    _haveRaster(demInterp.haveRaster()), _refHeight(demInterp.refHeight()),
    _length(demInterp.length()), _width(demInterp.width()),
    _xstart(demInterp.xStart()), _ystart(demInterp.yStart()),
    _deltax(demInterp.deltaX()), _deltay(demInterp.deltaY()),
    _epsgcode(demInterp.epsgCode()), _owner(true) {

    // Set the device
    //cudaSetDevice(0);

    // Allocate memory on device for DEM data
    size_t npix = _length * _width;
    checkCudaErrors(cudaMalloc((float **) &_dem, npix*sizeof(float)));

    // Copy DEM data
    checkCudaErrors(cudaMemcpy(_dem, &demInterp.data()[0], npix*sizeof(float),
                               cudaMemcpyHostToDevice));
}

/** @param[in] demInterp gpuDEMInterpolator object.
  *
  * Copy DEM interpolator data within GPU device. */
__host__ __device__
isce::cuda::geometry::gpuDEMInterpolator::
gpuDEMInterpolator(isce::cuda::geometry::gpuDEMInterpolator & demInterp) :
    _haveRaster(demInterp.haveRaster()), _refHeight(demInterp.refHeight()),
    _length(demInterp.length()), _width(demInterp.width()),
    _xstart(demInterp.xStart()), _ystart(demInterp.yStart()),
    _deltax(demInterp.deltaX()), _deltay(demInterp.deltaY()),
    _dem(demInterp._dem), _epsgcode(demInterp.epsgCode()), 
    _owner(false) {}

/** Destructor. */
isce::cuda::geometry::gpuDEMInterpolator::
~gpuDEMInterpolator() {
    // Only owner of DEM memory clears it
    if (_owner && _dem) {
        checkCudaErrors(cudaFree(_dem));
    }
}

/** @param[out] double* Array of lon-lat-h at middle of DEM. */
__device__
void
isce::cuda::geometry::gpuDEMInterpolator::
midLonLat(double * llh) const {
    // Create coordinates for middle X/Y
    double xyz[3] = {midX(), midY(), _refHeight};
    // Call projection inverse
    (*_proj)->inverse(xyz, llh);
} 

/** @param[in] lon longitude of interpolation point.
  * @param[in] lat latitude of interpolation point.
  * @param[out] value Interpolated DEM height. */
__device__
double
isce::cuda::geometry::gpuDEMInterpolator::
interpolateLonLat(double lon, double lat) const {

    // If we don't have a DEM, just return reference height
    double value = _refHeight;
    if (!_haveRaster) {
        return value;
    }

    // Pass latitude and longitude through projection
    double xyz[3];
    double llh[3] = {lon, lat, 0.0};
    (*_proj)->forward(llh, xyz);

    // Interpolate DEM at its native coordinates
    value = interpolateXY(xyz[0], xyz[1]);

    // Done
    return value;
}

/** @param[in] x X-coordinate of interpolation point.
  * @param[in] y Y-coordinate of interpolation point.
  * @param[out] value Interpolated DEM height. */
__device__
double
isce::cuda::geometry::gpuDEMInterpolator::
interpolateXY(double x, double y) const {

    // If we don't have a DEM, just return reference height
    double value = _refHeight;
    if (!_haveRaster) {
        return value;
    }

    // Compute the row and column for requested lat and lon
    const double row = (y - _ystart) / _deltay;
    const double col = (x - _xstart) / _deltax;

    // Check validity of interpolation coordinates
    const int irow = int(std::floor(row));
    const int icol = int(std::floor(col));
    // If outside bounds, return reference height
    if (irow < 2 || irow >= int(_length - 1))
        return _refHeight;
    if (icol < 2 || icol >= int(_width - 1))
        return _refHeight;

    //// Choose correct interpolation routine
    //if (_interpMethod == isce::core::BILINEAR_METHOD) {
    //    value = isce::core::Interpolator::bilinear(col, row, _dem);
    //} else if (_interpMethod == isce::core::BICUBIC_METHOD) {
    //    value = isce::core::Interpolator::bicubic(col, row, _dem);
    //} else if (_interpMethod == isce::core::AKIMA_METHOD) {
    //    value = isce::core::Interpolator::akima(col, row, _dem);
    //} else if (_interpMethod == isce::core::BIQUINTIC_METHOD) {
    //    value = isce::core::Interpolator::interp_2d_spline(6, _dem, col, row);
    //} else if (_interpMethod == isce::core::NEAREST_METHOD) {
    //    value = _dem(int(std::round(row)), int(std::round(col)));
    //}

    // TEMP: bilinear only
    value = bilinear(col, row, _dem, _width); 

    // Done
    return value;
}

// end of file
