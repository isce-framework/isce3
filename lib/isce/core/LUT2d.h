//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_LUT2D_H
#define ISCE_CORE_LUT2D_H

// std
#include <complex>
#include <valarray>

// isce::core
#include <isce/core/Interpolator.h>

// Declaration
namespace isce {
    namespace core {
        template <typename T> class LUT2d;
    }
}

// LUT2d declaration
template <typename T>
class isce::core::LUT2d {

    public:
        // Constructors
        inline LUT2d();
        inline LUT2d(isce::core::dataInterpMethod method);
        LUT2d(double xstart, double ystart, double dx, double dy,
              const isce::core::Matrix<T> & data,
              isce::core::dataInterpMethod method = isce::core::BILINEAR_METHOD);
        LUT2d(const std::valarray<double> & xcoord,
              const std::valarray<double> & ycoord,
              const isce::core::Matrix<T> & data,
              isce::core::dataInterpMethod method = isce::core::BILINEAR_METHOD);

        // Deep copy constructor
        inline LUT2d(const LUT2d<T> & lut);

        // Set data from external data
        inline void setFromData(const std::valarray<double> & xcoord,
                                const std::valarray<double> & ycoord,
                                const isce::core::Matrix<T> & data);

        // Get interpolator method
        inline isce::core::dataInterpMethod interpMethod() const {
            return _interp->method();
        }

        // Get starting X-coordinate
        inline double xStart() const { return _xstart; }
        // Get starting Y-coordinate
        inline double yStart() const { return _ystart; }
        // Get X-spacing
        inline double xSpacing() const { return _dx; }
        // Get Y-spacing
        inline double ySpacing() const { return _dy; }
        // Get read-only reference to data
        inline const isce::core::Matrix<T> & data() const { return _data; }
              
        // Evaluate LUT    
        T eval(double, double) const;

    private:
        // Data
        double _xstart, _ystart, _dx, _dy;
        isce::core::Matrix<T> _data;
        // Interpolation method
        isce::core::Interpolator<T> * _interp;

    private:
        inline void _setInterpolator(isce::core::dataInterpMethod method);
};

// Default constructor using bilinear interpolator
template <typename T>
isce::core::LUT2d<T>::
LUT2d() {
    _setInterpolator(isce::core::BILINEAR_METHOD);
}

// Constructor with specified interpolator
template <typename T>
isce::core::LUT2d<T>::
LUT2d(isce::core::dataInterpMethod method) {
    _setInterpolator(method);
}

// Deep copy constructor
template <typename T>
isce::core::LUT2d<T>::
LUT2d(const isce::core::LUT2d<T> & lut) : _xstart(lut.xStart()), _ystart(lut.yStart()),
                                          _dx(lut.xSpacing()), _dy(lut.ySpacing()),
                                          _data(lut.data()) {
    _setInterpolator(lut.interpMethod());
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
