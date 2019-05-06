//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_CORE_PIXEL_H
#define ISCE_CORE_PIXEL_H

// Declaration
namespace isce {
    namespace core {
        class Pixel;
    }
}

/** Helper datastructure to handle slant range information for a pixel */
class isce::core::Pixel {

    public:
        // Constructors
        Pixel() {};
        Pixel(double r, double d, size_t b) : _range(r), _dopfact(d), _bin(b) {}
        // Getters
        double range() const { return _range; }
        double dopfact() const { return _dopfact; }
        size_t bin() const { return _bin; }
        // Setters
        void range(double r) { _range = r; }
        void dopfact(double d) { _dopfact = d; }
        void bin(size_t b) { _bin = b; }

    private:
        double _range;
        double _dopfact;
        size_t _bin;
};

#endif

// end of file
