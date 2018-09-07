// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#include "gpuDEMInterpolator.h"

/** @param[in] demInterp DEMInterpolator object.
  *
  * Copy DEM interpolator data to GPU device. */
__host__
isce::cuda::geometry::gpuDEMInterpolator::
gpuDEMInterpolator(const isce::geometry::DEMInterpolator & demInterp) :
    _haveRaster(demInterp.haveRaster()), _refHeight(demInterp.refHeight()) {

    // Set the device
    cudaSetDevice(0);

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
    //isce::core::cartesian_t xyz;
    //const isce::core::cartesian_t llh{lon, lat, 0.0};
    //_proj->forward(llh, xyz);

    // Interpolate DEM at its native coordinates
    value = interpolateXY(lon, lat);

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
    //// If outside bounds, return reference height
    //if (irow < 2 || irow >= int(_dem.length() - 1))
    //    return _refHeight;
    //if (icol < 2 || icol >= int(_dem.width() - 1))
    //    return _refHeight;

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

    // TEMP STATEMENT
    value = row + col;

    // Done
    return value;
}

// end of file
