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
        void resamp(bool flatten=true, bool isComplex=true, size_t rowBuffer=40);

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

        // Tile initialization
        void _initializeTile(Tile<std::complex<float>> &, size_t, size_t &, size_t &);

        // Tile transformation
        void _transformTile(Tile<std::complex<float>> &,
            std::vector<std::complex<float>> &,
            std::vector<std::complex<float>> &,
            size_t, size_t);

        // Convenience functions
        inline void _clearPolys();
        inline size_t _computeNumberOfTiles(size_t);
};

// Get inline implementations for ResampSlc
#define ISCE_CORE_RESAMPSLC_ICC
#include "ResampSlc.icc"
#undef ISCE_CORE_RESAMPSLC_ICC

#endif

// end of file
