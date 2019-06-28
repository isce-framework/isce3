// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_LUT2D_H
#define ISCE_CORE_LUT2D_H
#pragma once

#include "forward.h"

#include <valarray>
#include "Constants.h"
#include "Interpolator.h"
#include "Utilities.h"

/** Data structure to store 2D Lookup table.
 *  Suitable for storing data of the form z = f(x,y)*/
template <typename T>
class isce::core::LUT2d {

    public:
        // Constructors
        inline LUT2d();
        inline LUT2d(isce::core::dataInterpMethod method);
        LUT2d(double xstart, double ystart, double dx, double dy,
              const isce::core::Matrix<T> & data,
              isce::core::dataInterpMethod method = isce::core::BILINEAR_METHOD,
              bool boundsError = true);
        LUT2d(const std::valarray<double> & xcoord,
              const std::valarray<double> & ycoord,
              const isce::core::Matrix<T> & data,
              isce::core::dataInterpMethod method = isce::core::BILINEAR_METHOD,
              bool boundsError = true);

        // Deep copy constructor
        inline LUT2d(const LUT2d<T> & lut);
        
        // Deep assignment operator
        inline LUT2d & operator=(const LUT2d<T> & lut);

        // Set data from external data
        void setFromData(const std::valarray<double> & xcoord,
                         const std::valarray<double> & ycoord,
                         const isce::core::Matrix<T> & data);

        // Get interpolator method
        inline isce::core::dataInterpMethod interpMethod() const {
            return _interp->method();
        }
    
        // Set interpolator method
        inline void interpMethod(isce::core::dataInterpMethod method) {
            _setInterpolator(method);
        }

        // Get starting X-coordinate
        inline double xStart() const { return _xstart; }
        // Get starting Y-coordinate
        inline double yStart() const { return _ystart; }
        // Get X-spacing
        inline double xSpacing() const { return _dx; }
        // Get Y-spacing
        inline double ySpacing() const { return _dy; }
        // Get LUT length (number of lines)
        inline size_t length() const { return _data.length(); }
        // Get LUT width (number of samples)
        inline size_t width() const { return _data.width(); }
        // Get the reference value
        inline T refValue() const { return _refValue; }
        // Get flag for having data
        inline bool haveData() const { return _haveData; }
        // Get bounds error flag
        inline bool boundsError() const { return _boundsError; }
        // Get read-only reference to data
        inline const isce::core::Matrix<T> & data() const { return _data; }

        // Set bounds error floag
        inline void boundsError(bool flag) { _boundsError = flag; }
              
        // Evaluate LUT    
        T eval(double y, double x) const;

    private:
        // Flags
        bool _haveData, _boundsError;
        T _refValue;
        // Data
        double _xstart, _ystart, _dx, _dy;
        isce::core::Matrix<T> _data;
        // Interpolation method
        isce::core::Interpolator<T> * _interp;

    private:
        inline void _setInterpolator(isce::core::dataInterpMethod method);

    // BVR: I'm placing the comparison operator implementations inline here because
    // it wasn't clear to me how to handle the template arguments out-of-line
    public:

        // Comparison operator for floating point
        template <typename U = T, std::enable_if_t<!std::is_compound<U>::value, int> = 0>
        inline bool operator==(const isce::core::LUT2d<T> & other) const {
            // Check coordinates and dimensions first
            bool equal = _data.width() == other.width();
            equal *= _data.length() == other.length();
            if (!equal) {
                return false;
            }
            equal *= isce::core::compareFloatingPoint(_xstart, other.xStart());
            equal *= isce::core::compareFloatingPoint(_ystart, other.yStart());
            equal *= isce::core::compareFloatingPoint(_dx, other.xSpacing());
            equal *= isce::core::compareFloatingPoint(_dy, other.ySpacing());
            if (!equal) {
                return false;
            }
            // If we made it this far, check contents
            const isce::core::Matrix<T> & otherMat = other.data();
            for (size_t i = 0; i < _data.length(); ++i) {
                for (size_t j = 0; j < _data.width(); ++j) {
                    equal *= isce::core::compareFloatingPoint(_data(i,j), otherMat(i,j));
                }
            }
            return equal;
        }
        
        // Comparison operator for complex
        template <typename U = T, std::enable_if_t<std::is_compound<U>::value, int> = 0>
        inline bool operator==(const isce::core::LUT2d<T> & other) const {
            // Check coordinates and dimensions first
            bool equal = _data.width() == other.width();
            equal *= _data.length() == other.length();
            if (!equal) {
                return false;
            }
            equal *= isce::core::compareFloatingPoint(_xstart, other.xStart());
            equal *= isce::core::compareFloatingPoint(_ystart, other.yStart());
            equal *= isce::core::compareFloatingPoint(_dx, other.xSpacing());
            equal *= isce::core::compareFloatingPoint(_dy, other.ySpacing());
            if (!equal) {
                return false;
            }
            // If we made it this far, check contents
            const isce::core::Matrix<T> & otherMat = other.data();
            for (size_t i = 0; i < _data.length(); ++i) {
                for (size_t j = 0; j < _data.width(); ++j) {
                    equal *= isce::core::compareComplex(_data(i,j), otherMat(i,j));
                }
            }
            return equal;
        }
};

// Default constructor using bilinear interpolator
template <typename T>
isce::core::LUT2d<T>::
LUT2d() : _haveData(false), _boundsError(true), _refValue(0.0) {
    _setInterpolator(isce::core::BILINEAR_METHOD);
}

// Constructor with specified interpolator
template <typename T>
isce::core::LUT2d<T>::
LUT2d(isce::core::dataInterpMethod method) : _haveData(false), _boundsError(true),
                                             _refValue(0.0) {
    _setInterpolator(method);
}

// Deep copy constructor
template <typename T>
isce::core::LUT2d<T>::
LUT2d(const isce::core::LUT2d<T> & lut) : _haveData(lut.haveData()),
                                          _boundsError(lut.boundsError()),
                                          _refValue(lut.refValue()),
                                          _xstart(lut.xStart()), _ystart(lut.yStart()),
                                          _dx(lut.xSpacing()), _dy(lut.ySpacing()),
                                          _data(lut.data()) {
    _setInterpolator(lut.interpMethod());
}

// Deep assignment operator
template <typename T>
isce::core::LUT2d<T> &
isce::core::LUT2d<T>::
operator=(const LUT2d<T> & lut) {
    _refValue = lut.refValue();
    _xstart = lut.xStart();
    _ystart = lut.yStart();
    _dx = lut.xSpacing();
    _dy = lut.ySpacing();
    _data = lut.data();
    _haveData = lut.haveData();
    _boundsError = lut.boundsError();
    _setInterpolator(lut.interpMethod());
    return *this;
}

// Set interpolator method
/** @param[in] method Data interpolation method */
template <typename T>
void
isce::core::LUT2d<T>::
_setInterpolator(isce::core::dataInterpMethod method) {

    // If biquintic, set the order
    if (method == isce::core::BIQUINTIC_METHOD) {
        _interp = isce::core::createInterpolator<T>(isce::core::BIQUINTIC_METHOD, 6);

    // If sinc, set the window sizes
    } else if (method == isce::core::SINC_METHOD) {
        _interp = isce::core::createInterpolator<T>(
            isce::core::SINC_METHOD,
            6, isce::core::SINC_LEN, isce::core::SINC_SUB
        );

    // Otherwise, just pass the interpolation method
    } else {
        _interp = isce::core::createInterpolator<T>(method);
    }
}

#endif
