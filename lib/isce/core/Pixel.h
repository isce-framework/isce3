//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_CORE_PIXEL_H
#define ISCE_CORE_PIXEL_H
#pragma once

#include "forward.h"

/** Helper datastructure to handle slant range information for a pixel */
class isce::core::Pixel {

    public:
        // Constructors
        CUDA_HOSTDEV Pixel() {};
        CUDA_HOSTDEV Pixel(double r, double d, size_t b) : _range(r), _dopfact(d), _bin(b) {}
        // Getters
        CUDA_HOSTDEV double range() const { return _range; }
        CUDA_HOSTDEV double dopfact() const { return _dopfact; }
        CUDA_HOSTDEV size_t bin() const { return _bin; }
        // Setters
        CUDA_HOSTDEV void range(double r) { _range = r; }
        CUDA_HOSTDEV void dopfact(double d) { _dopfact = d; }
        CUDA_HOSTDEV void bin(size_t b) { _bin = b; }

    private:
        double _range;
        double _dopfact;
        size_t _bin;
};

#endif

// end of file
