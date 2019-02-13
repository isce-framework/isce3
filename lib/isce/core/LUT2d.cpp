//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017

#include <complex>
#include "LUT2d.h"

// Constructor with coordinate starting values and spacing
/** @param[in] xstart Starting X-coordinate
  * @param[in] ystart Starting Y-coordinate
  * @param[in] dx X-spacing
  * @param[in] dy Y-spacing
  * @param[in] method Interpolation method */
template <typename T>
isce::core::LUT2d<T>::
LUT2d(double xstart, double ystart, double dx, double dy, const isce::core::Matrix<T> & data,
      isce::core::dataInterpMethod method) : _xstart(xstart), _ystart(ystart),
      _dx(dx), _dy(dy), _data(data) {
    _setInterpolator(method);
}


// Constructor with valarrays of X and Y coordinates
/** @param[in] xcoord X-coordinates
  * @param[in] ycoord Y-coordinates
  * @param[in] data Matrix of LUT data
  * @param[in] method Interpolation method */
template <typename T>
isce::core::LUT2d<T>::
LUT2d(const std::valarray<double> & xcoord, const std::valarray<double> & ycoord,
      const isce::core::Matrix<T> & data, isce::core::dataInterpMethod method) {
    // Set the data
    setFromData(xcoord, ycoord, data);
    // Save interpolation data 
    _setInterpolator(method);
}

// Set from external data
/** @param[in] xcoord X-coordinates
  * @param[in] ycoord Y-coordinates
  * @param[in] data Matrix of LUT data */
template <typename T>
isce::core::LUT2d<T>::
setFromData(const std::valarray<double> & xcoord, const std::valarray<double> & ycoord,
            const isce::core::Matrix<T> & data) {

    // Consistency check for sizes
    if (xcoord.size() != data.width() || ycoord.size() != data.length()) {
        pyre::journal::error_t errorChannel("isce.core.LUT2d");
        errorChannel
            << pyre::journal::at(__HERE__)
            << "Inconsistent shapes between data and coordinates"
            << pyre::journal::endl;
    }

    // Check Y-coordinates are on a regular grid
    const double dy = ycoord[1] - ycoord[0];
    for (size_t i = 1; i < (ycoord.size() - 1); ++i) {
        const double d = ycoord[i+1] - ycoord[i];
        if (std::abs(d - dy) > 1.0e-8) {
            pyre::journal::error_t errorChannel("isce.core.LUT2d");
            errorChannel
                << pyre::journal::at(__HERE__)
                << "Detected non-regular Y-coordinates for LUT2d grid."
                << pyre::journal::endl;
        }
    }

    // Do the same for X-coordinates
    const double dx = xcoord[1] - xcoord[0];
    for (size_t i = 1; i < (xcoord.size() - 1); ++i) {
        const double d = xcoord[i+1] - xcoord[i];
        if (std::abs(d - dx) > 1.0e-8) {
            pyre::journal::error_t errorChannel("isce.core.LUT2d");
            errorChannel
                << pyre::journal::at(__HERE__)
                << "Detected non-regular X-coordinates for LUT2d grid."
                << pyre::journal::endl;
        }
    }

    // Set start and spacing
    _xstart = xcoord[0];
    _ystart = ycoord[0];
    _dx = dx;
    _dy = dy;

    // Copy data
    _data = data;
} 

// Evaluate LUT at coordinate
/** @param[in] x X-coordinate for evaluation
  * @param[in] y Y-coordinate for evaluation
  * @param[out] value Interpolated value */
template <typename T>
T isce::core::LUT2d<T>::
eval(double y, double x) const {
    /*
     * Evaluate the LUT at the given coordinates.
     */

    // Get matrix indices corresponding to requested coordinates
    const double x_idx = (x - _xstart) / _dx;
    const double y_idx = (y - _ystart) / _dy;

    // Call interpolator
    T value = _interp->interpolate(x_idx, y_idx, _data);
    return value;
}

// Forward declaration of classes
template class isce::core::LUT2d<double>;
template class isce::core::LUT2d<float>;
template class isce::core::LUT2d<std::complex<double>>;
template class isce::core::LUT2d<std::complex<float>>;

// end of file
