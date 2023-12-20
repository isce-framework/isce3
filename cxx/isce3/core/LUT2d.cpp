//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017

#include "LUT2d.h"

#include <complex>
#include <pyre/journal.h>

#include "Interpolator.h"

// Constructor with coordinate starting values and spacing
/** @param[in] xstart Starting X-coordinate
  * @param[in] ystart Starting Y-coordinate
  * @param[in] dx X-spacing
  * @param[in] dy Y-spacing
  * @param[in] method Interpolation method */
template <typename T>
isce3::core::LUT2d<T>::
LUT2d(double xstart, double ystart, double dx, double dy, const isce3::core::Matrix<T> & data,
      isce3::core::dataInterpMethod method, bool boundsError) :
          _haveData(true), _boundsError(boundsError), _refValue(data(0,0)),
          _xstart(xstart), _ystart(ystart), _dx(dx), _dy(dy), _data(data) {
    _setInterpolator(method);
}

// Constructor with valarrays of X and Y coordinates
/** @param[in] xcoord X-coordinates
  * @param[in] ycoord Y-coordinates
  * @param[in] data Matrix of LUT data
  * @param[in] method Interpolation method */
template <typename T>
isce3::core::LUT2d<T>::
LUT2d(const std::valarray<double> & xcoord, const std::valarray<double> & ycoord,
      const isce3::core::Matrix<T> & data, isce3::core::dataInterpMethod method,
      bool boundsError) :
          _haveData(true), _boundsError(boundsError), _refValue(data(0,0)) {
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
void
isce3::core::LUT2d<T>::
setFromData(const std::valarray<double> & xcoord, const std::valarray<double> & ycoord,
            const isce3::core::Matrix<T> & data) {

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
    _haveData = true;
    _refValue = data(0,0);
}

// Evaluate LUT at coordinate
/** @param[in] y Y-coordinate for evaluation
  * @param[in] x X-coordinate for evaluation
  * @param[out] value Interpolated value */
template <typename T>
T isce3::core::LUT2d<T>::
eval(const double y, const double x) const {
    /*
     * Evaluate the LUT at the given coordinates.
     */

    // Check if data are available; if not, return ref value
    T value = _refValue;
    if (!_haveData) {
        return value;
    }

    // Get matrix indices corresponding to requested coordinates
    double x_idx = (x - _xstart) / _dx;
    double y_idx = (y - _ystart) / _dy;

    // Check bounds or clamp indices to valid values
    if (_boundsError && not contains(y, x)) {
        pyre::journal::error_t errorChannel("isce.core.LUT2d");
        errorChannel
            << "Out of bounds LUT2d evaluation at " << y << " " << x
            << pyre::journal::newline
            << " - bounds are " << _ystart << " "
            << _ystart + _dy * (_data.length() - 1.0) << " "
            << _xstart << " " << _xstart + _dx * (_data.width() - 1.0)
            << pyre::journal::endl;
    }
    x_idx = isce3::core::clamp(x_idx, 0.0, _data.width() - 1.0);
    y_idx = isce3::core::clamp(y_idx, 0.0, _data.length() - 1.0);

    // Call interpolator
    value = _interp->interpolate(x_idx, y_idx, _data);
    return value;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> isce3::core::LUT2d<T>::
eval(double y, const Eigen::Ref<const Eigen::VectorXd>& x) const
{
    const auto n = x.size();
    Eigen::Matrix<T, Eigen::Dynamic, 1> out(n);
    _Pragma("omp parallel for")
    for (long i = 0; i < n; ++i) {
        out(i) = eval(y, x(i));
    }
    return out;
}

template <typename T>
void
isce3::core::LUT2d<T>::
_setInterpolator(isce3::core::dataInterpMethod method)
{
    // If biquintic, set the order
    if (method == isce3::core::BIQUINTIC_METHOD) {
        _interp = isce3::core::createInterpolator<T>(isce3::core::BIQUINTIC_METHOD, 6);

    // If sinc, set the window sizes
    } else if (method == isce3::core::SINC_METHOD) {
        _interp = isce3::core::createInterpolator<T>(
            isce3::core::SINC_METHOD,
            6, isce3::core::SINC_LEN, isce3::core::SINC_SUB
        );

    // Otherwise, just pass the interpolation method
    } else {
        _interp = isce3::core::createInterpolator<T>(method);
    }
}

template<typename T>
isce3::core::dataInterpMethod isce3::core::LUT2d<T>::interpMethod() const
{
    return _interp->method();
}

template<typename T>
void isce3::core::LUT2d<T>::interpMethod(dataInterpMethod method)
{
    _setInterpolator(method);
}

// Forward declaration of classes
template class isce3::core::LUT2d<double>;
template class isce3::core::LUT2d<float>;
template class isce3::core::LUT2d<std::complex<double>>;
template class isce3::core::LUT2d<std::complex<float>>;

// end of file
