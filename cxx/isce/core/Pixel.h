//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#pragma once

#include "forward.h"

/** Helper datastructure to handle slant range information for a pixel */
class isce::core::Pixel {

    public:
        // Constructors
        CUDA_HOSTDEV Pixel() {};
        /** Pixel constructor
         *
         * @param[in] r         Range in meters.
         * @param[in] dopfact   r*sin(squint), where squint is the complement of
         *                      the angle between the velocity and look vector.
         * @param[in] bin       Range bin number.
         */
        CUDA_HOSTDEV Pixel(double r, double dopfact, size_t bin) :
            _range(r), _dopfact(dopfact), _bin(bin) {}
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
