//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_LUT1D_H
#define ISCE_CORE_LUT1D_H

#include <complex>
#include <valarray>

// pyre
#include <portinfo>
#include <pyre/journal.h>

// Declaration
namespace isce {
    namespace core {
        template <typename T> class LUT1d;
    }
}

// LUT1d declaration
template <typename T>
class isce::core::LUT1d {

    public:
        /** Default constructor */
        inline LUT1d() : _extrapolate{true} {
            std::valarray<double> x{0.0, 1.0};
            std::valarray<double> y{0.0, 0.0};
            _coords = x;
            _values = y;
        } 
        
        /** Constructor with coordinates and values */
        inline LUT1d(const std::valarray<double> & coords, const std::valarray<T> & values,
                     bool extrapolate = false) : _coords(coords), _values(values),
                     _extrapolate{extrapolate} {}

        /** Copy constructor. */
        inline LUT1d(const LUT1d<T> & lut) :
            _coords(lut.coords()), _values(lut.values()), _extrapolate(lut.extrapolate()) {}

        /** Assignment operator. */
        inline LUT1d & operator=(const LUT1d<T> & lut) {
            _coords = lut.coords();
            _values = lut.values();
            _extrapolate = lut.extrapolate();
            return *this;
        }

        /** Get the coordinates */
        inline std::valarray<double> coords() const { return _coords; }

        /** Set the coordinates */
        inline void coords(const std::valarray<double> & c) { _coords = c; }

        /** Get the values */
        inline std::valarray<T> values() const { return _values; }

        /** Set the values */
        inline void values(const std::valarray<T> & v) { _values = v; }

        /** Get extrapolate flag */
        inline bool extrapolate() const { return _extrapolate; }

        /** Set extrapolate flag */
        inline void extrapolate(bool flag) { _extrapolate = flag; }

        /** Get size of LUT */
        inline size_t size() const { return _coords.size(); }

        /** Evaluate the LUT */
        inline T eval(double x) const;

    // Data members
    private:
        std::valarray<double> _coords;
        std::valarray<T> _values;
        bool _extrapolate;
};

/** @param[in] x Point to evaluate the LUT
  * @param[out] result Interpolated value */
template <typename T>
T isce::core::LUT1d<T>::
eval(double x) const {

    // Check bounds to see if we need to perform linear extrapolation
    const int n = _coords.size();
    if (x < _coords[0]) {
        if (_extrapolate) {
            const double dx = _coords[0] - _coords[1];
            const double dy = _values[0] - _values[1];
            const double d = x - _coords[1];
            T result = (dy / dx) * d + _values[1];
            return result;
        } else {
            pyre::journal::error_t errorChannel("isce.core.LUT1d");
            errorChannel
                << pyre::journal::at(__HERE__)
                << "Out of bounds evaluation for LUT1d."
                << pyre::journal::newline
                << pyre::journal::endl;
            return 0;
        }
    } else if (x > _coords[n-1]) {
        if (_extrapolate) {
            const double dx = _coords[n-1] - _coords[n-2];
            const double dy = _values[n-1] - _values[n-2];
            const double d = x - _coords[n-2];
            T result = (dy / dx) * d + _values[n-2];
            return result;
        } else {
            pyre::journal::error_t errorChannel("isce.core.LUT1d");
            errorChannel
                << pyre::journal::at(__HERE__)
                << "Out of bounds evaluation for LUT1d."
                << pyre::journal::newline
                << pyre::journal::endl;
            return 0;
        }
    }

    // Otherwise, proceed with interpolation
    // Iterate over coordinates to find x bounds
    double xdiff = -100.0;
    int j;
    for (j = 0; j < n - 1; ++j) {
        // Compute difference with current coordinate
        xdiff = _coords[j] - x;
        // Break if sign has changed
        if (xdiff > 0.0)
            break;
    }
    
    // The indices of the x bounds
    const int j0 = j - 1;
    const int j1 = j;
    
    // Get coordinates at bounds
    double x1 = _coords[j0];
    double x2 = _coords[j1];

    // Interpolate
    T result = (x2 - x) / (x2 - x1) * _values[j0] + (x - x1) / (x2 - x1) * _values[j1];
    return result;
}

#endif

// end of file
