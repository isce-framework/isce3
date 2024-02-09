// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#pragma once

#include "forward.h"

#include <Eigen/Dense>
#include <valarray>
#include "Constants.h"
#include "Matrix.h"
#include "Utilities.h"

/** Data structure to store 2D Lookup table.
 *  Suitable for storing data of the form z = f(x,y)*/
template <typename T>
class isce3::core::LUT2d {

    public:
        // Constructors
        inline LUT2d();
        inline LUT2d(isce3::core::dataInterpMethod method);
        LUT2d(double xstart, double ystart, double dx, double dy,
              const isce3::core::Matrix<T> & data,
              isce3::core::dataInterpMethod method = isce3::core::BILINEAR_METHOD,
              bool boundsError = true);
        LUT2d(const std::valarray<double> & xcoord,
              const std::valarray<double> & ycoord,
              const isce3::core::Matrix<T> & data,
              isce3::core::dataInterpMethod method = isce3::core::BILINEAR_METHOD,
              bool boundsError = true);

        // Deep copy constructor
        inline LUT2d(const LUT2d<T> & lut);

        // Deep assignment operator
        inline LUT2d & operator=(const LUT2d<T> & lut);

        // Set data from external data.
        // No-op if shape is 0x0, sets refValue if shape is 1x1.
        void setFromData(const std::valarray<double> & xcoord,
                         const std::valarray<double> & ycoord,
                         const isce3::core::Matrix<T> & data);

        // Get interpolator method
        isce3::core::dataInterpMethod interpMethod() const;

        // Set interpolator method
        void interpMethod(isce3::core::dataInterpMethod method);

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
        inline const isce3::core::Matrix<T> & data() const { return _data; }

        // Set bounds error floag
        inline void boundsError(bool flag) { _boundsError = flag; }

        // Evaluate LUT
        T eval(const double y, const double x) const;

        Eigen::Matrix<T, Eigen::Dynamic, 1>
        eval(double y, const Eigen::Ref<const Eigen::VectorXd>& x) const;

        /** Check if point resides in domain of LUT */
        inline bool contains(double y, double x) const
        {
            // Treat default-constructed LUT as having infinite extent.
            if (not _haveData) {
                return true;
            }

            const auto i = (x - xStart()) / xSpacing();
            const auto j = (y - yStart()) / ySpacing();
            return (i >= 0.0 and i <= width() - 1.0) and
                   (j >= 0.0 and j <= length() - 1.0);
        }

    private:
        // Flags
        bool _haveData, _boundsError;
        T _refValue;
        // Data
        double _xstart, _ystart, _dx, _dy;
        isce3::core::Matrix<T> _data;
        // Interpolation method
        isce3::core::Interpolator<T> * _interp;

    private:
        /** @internal
         * Set interpolator method
         * @param[in] method Data interpolation method
         */
        void _setInterpolator(dataInterpMethod method);

    // BVR: I'm placing the comparison operator implementations inline here because
    // it wasn't clear to me how to handle the template arguments out-of-line
    public:

        // Comparison operator for floating point
        template <typename U = T, std::enable_if_t<!std::is_compound<U>::value, int> = 0>
        inline bool operator==(const isce3::core::LUT2d<T> & other) const {
            // Check coordinates and dimensions first
            bool equal = _data.width() == other.width();
            equal *= _data.length() == other.length();
            if (!equal) {
                return false;
            }
            equal *= isce3::core::compareFloatingPoint(_xstart, other.xStart());
            equal *= isce3::core::compareFloatingPoint(_ystart, other.yStart());
            equal *= isce3::core::compareFloatingPoint(_dx, other.xSpacing());
            equal *= isce3::core::compareFloatingPoint(_dy, other.ySpacing());
            if (!equal) {
                return false;
            }
            // If we made it this far, check contents
            const isce3::core::Matrix<T> & otherMat = other.data();
            for (size_t i = 0; i < _data.length(); ++i) {
                for (size_t j = 0; j < _data.width(); ++j) {
                    equal *= isce3::core::compareFloatingPoint(_data(i,j), otherMat(i,j));
                }
            }
            return equal;
        }

        // Comparison operator for complex
        template <typename U = T, std::enable_if_t<std::is_compound<U>::value, int> = 0>
        inline bool operator==(const isce3::core::LUT2d<T> & other) const {
            // Check coordinates and dimensions first
            bool equal = _data.width() == other.width();
            equal *= _data.length() == other.length();
            if (!equal) {
                return false;
            }
            equal *= isce3::core::compareFloatingPoint(_xstart, other.xStart());
            equal *= isce3::core::compareFloatingPoint(_ystart, other.yStart());
            equal *= isce3::core::compareFloatingPoint(_dx, other.xSpacing());
            equal *= isce3::core::compareFloatingPoint(_dy, other.ySpacing());
            if (!equal) {
                return false;
            }
            // If we made it this far, check contents
            const isce3::core::Matrix<T> & otherMat = other.data();
            for (size_t i = 0; i < _data.length(); ++i) {
                for (size_t j = 0; j < _data.width(); ++j) {
                    equal *= isce3::core::compareComplex(_data(i,j), otherMat(i,j));
                }
            }
            return equal;
        }
};

// Default constructor using bilinear interpolator
template <typename T>
isce3::core::LUT2d<T>::
LUT2d() : _haveData(false), _boundsError(true), _refValue(0.0) {
    _setInterpolator(isce3::core::BILINEAR_METHOD);
}

// Constructor with specified interpolator
template <typename T>
isce3::core::LUT2d<T>::
LUT2d(isce3::core::dataInterpMethod method) : _haveData(false), _boundsError(true),
                                             _refValue(0.0) {
    _setInterpolator(method);
}

// Deep copy constructor
template <typename T>
isce3::core::LUT2d<T>::
LUT2d(const isce3::core::LUT2d<T> & lut) : _haveData(lut.haveData()),
                                          _boundsError(lut.boundsError()),
                                          _refValue(lut.refValue()),
                                          _xstart(lut.xStart()), _ystart(lut.yStart()),
                                          _dx(lut.xSpacing()), _dy(lut.ySpacing()),
                                          _data(lut.data()) {
    _setInterpolator(lut.interpMethod());
}

// Deep assignment operator
template <typename T>
isce3::core::LUT2d<T> &
isce3::core::LUT2d<T>::
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
