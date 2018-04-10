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
#include "Interpolator.h"
#include "Poly2d.h"
#include "Metadata.h"
#include "Raster.h"
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
        typedef isce::core::Tile<std::complex<float>> Tile_t;

    // Meta-methods
    public:
        // Default constructor
        inline ResampSlc();
        // Destructor
        inline ~ResampSlc();

        // Polynomial getters
        inline Poly2d rgCarrier() const;
        inline Poly2d azCarrier() const;
        inline Poly2d doppler() const;
        // Polynomial setters
        inline void rgCarrier(Poly2d &);
        inline void azCarrier(Poly2d &);
        inline void doppler(Poly2d &);

        // Get metadata
        inline Metadata metadata() const;
        inline Metadata refMetadata() const;
        // Set metadata
        inline void metadata(Metadata);
        inline void refMetadata(Metadata);

        // Get/set number of lines per processing tile
        inline size_t linesPerTile() const;
        inline void linesPerTile(size_t);
                
        // Convenience functions
        inline void declare(int, int, int, int) const;

        // Main resamp entry point
        void resamp(const std::string &, const std::string &, const std::string &,
            const std::string &, bool flatten=false, bool isComplex=true, int rowBuffer=40);

    // Data members
    private:
        // Number of lines per tile
        size_t _linesPerTile = 1000;

        // Polynomials
        Poly2d _rgCarrier;            // range carrier polynomial
        Poly2d _azCarrier;            // azimuth carrier polynomial
        Poly2d _dopplerPoly;          // Doppler polynomial

        // Metadata
        Metadata _meta;               // radar metadata for image to be resampled
        Metadata _refMeta;            // radar metadata for reference master image

        // Array of sinc coefficient
        Matrix<float> _fintp;

        // Tile initialization
        void _initializeTile(Tile_t &, Raster &, Raster &, int);

        // Tile transformation
        void _transformTile(Tile_t &, Raster &, Raster &, Raster &, int, bool, int &);

        // Convenience functions
        inline int _computeNumberOfTiles(int, int);

        // Resampling interpolation methods
        void _prepareInterpMethods(int);
        inline std::complex<float> _interpolateComplex(Matrix<std::complex<float>> &,
            int, int, double, double, int, int);
};

// Get inline implementations for ResampSlc
#define ISCE_CORE_RESAMPSLC_ICC
#include "ResampSlc.icc"
#undef ISCE_CORE_RESAMPSLC_ICC

#endif

// end of file
