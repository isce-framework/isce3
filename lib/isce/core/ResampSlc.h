// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_CORE_RESAMPSLC_H
#define ISCE_CORE_RESAMPSLC_H

#include <cstdint>
#include <cstdio>
#include <complex>
#include <valarray>

// isce::core
#include "Poly2d.h"
#include "Metadata.h"
#include "Tile.h"

// Declarations
namespace isce {
    namespace core {
        class ResampSlc;
    }
}

// Definition
class isce::core::ResampSlc {

    // Public data members
    public:
        Metadata meta;
        Metadata refMeta;

    // Meta-methods
    public:
        // Default constructor
        inline ResampSlc();
        // Destructor
        inline ~ResampSlc();

        // Set the various polynomial attributes
        inline void setRgCarrier(Poly2d*);
        inline void setAzCarrier(Poly2d*);
        inline void setRgOffsets(Poly2d*);
        inline void setAzOffsets(Poly2d*);
        inline void setDoppler(Poly2d*);

        // Geometry getters
        inline size_t inputWidth() const;
        inline size_t inputLength() const;
        inline size_t outputWidth() const;
        inline size_t outputLength() const;

        // Convenience functions
        inline void resetPolys();
        inline void declare() const;

        // Main resamp entry point
        void resamp(const std::string &, const std::string &, const std::string &,
            const std::string &, bool flatten=true, bool isComplex=true, size_t rowBuffer=40);

    // Data members
    private:
        // Polynomials
        Poly2d * _rgCarrier;            // range carrier polynomial
        Poly2d * _azCarrier;            // azimuth carrier polynomial
        Poly2d * _rgOffsetsPoly;        // range offsets polynomial
        Poly2d * _azOffsetsPoly;        // azimuth offsets polynomial
        Poly2d * _dopplerPoly;          // Doppler polynomial

        // Geometry
        size_t _outWidth, _outLength;
        size_t _inWidth, _inLength;

        // Interpolation work data
        std::valarray<float> _fintp;
        float _fDelay;

        // Tile initialization
        void _initializeTile(Tile &, Raster &, size_t);

        // Tile transformation
        void _transformTile(Tile &,
            std::valarray<std::complex<float>> &,
            std::valarray<std::complex<float>> &,
            size_t, size_t);

        // Convenience functions
        inline void _clearPolys();
        inline size_t _computeNumberOfTiles(size_t);

        // Resampling interpolation methods
        void _prepareInterpMethods(int) const;
        inline std::complex<float> _interpolateComplex(std::valarray<std::complex<float>> &,
            int, int, double, double, int, int);
};

// Get inline implementations for ResampSlc
#define ISCE_CORE_RESAMPSLC_ICC
#include "ResampSlc.icc"
#undef ISCE_CORE_RESAMPSLC_ICC

#endif

// end of file
