// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Joshua Cohen, Bryan Riel
// Copyright 2017-2018
//

#ifndef ISCE_IMAGE_RESAMPSLC_H
#define ISCE_IMAGE_RESAMPSLC_H

#include <cstdint>
#include <cstdio>
#include <complex>
#include <valarray>

// isce::core
#include "isce/core/Interpolator.h"
#include "isce/core/Poly2d.h"
#include "isce/core/LUT1d.h"

// isce::io
#include "isce/io/Raster.h"

// isce::product
#include "isce/product/Product.h"

// isce::image
#include "Tile.h"

// Declarations
namespace isce {
    namespace image {
        class ResampSlc;
    }
}

// Definition
class isce::image::ResampSlc {

    // Public data members
    public:
        typedef Tile<std::complex<float>> Tile_t;
        const int SINC_ONE = isce::core::SINC_ONE;
        const int SINC_HALF = isce::core::SINC_HALF;

    // Meta-methods
    public:
        /** Constructor from an isce::product::Product (no flattening) */
        inline ResampSlc(const isce::product::Product & product,
                         char frequency = 'A');

        /** Constructor from an isce::product::Product and reference product (flattening) */
        inline ResampSlc(const isce::product::Product & product,
                         const isce::product::Product & refProduct,
                         char frequency = 'A');

        /** Constructor from an isce::product::Swath (no flattening) */
        inline ResampSlc(const isce::product::Swath & swath);

        /** Constructor from an isce::product::Swath and reference swath (flattening) */
        inline ResampSlc(const isce::product::Swath & swath,
                         const isce::product::Swath & refSwath);

        /** Constructor from individual components (no flattening) */
        inline ResampSlc(const isce::core::LUT2d<double> & doppler,
                         double startingRange, double rangePixelSpacing,
                         double sensingStart, double prf, double wvl);

        /** Constructor from individual components (flattening) */
        inline ResampSlc(const isce::core::LUT2d<double> & doppler,
                         double startingRange, double rangePixelSpacing,
                         double sensingStart, double prf, double wvl,
                         double refStartingRange, double refRangePixelSpacing,
                         double refWvl);

        /** Destructor */
        inline ~ResampSlc() {};

        // Poly2d and LUT getters
        inline isce::core::Poly2d rgCarrier() const;
        inline isce::core::Poly2d azCarrier() const;

        // Poly2d and LUT setters
        inline void rgCarrier(const isce::core::Poly2d &);
        inline void azCarrier(const isce::core::Poly2d &);

        /** Get read-only reference to Doppler LUT2d */
        inline const isce::core::LUT2d<double> & doppler() const;

        /** Get reference to Doppler LUT2d */
        inline isce::core::LUT2d<double> & doppler();

        /** Set Doppler LUT2d */
        inline void doppler(const isce::core::LUT2d<double> &);

        // Set reference product for flattening
        inline void referenceProduct(const isce::product::Product & product,
                                     char frequency = 'A');

        // Get/set number of lines per processing tile
        inline size_t linesPerTile() const;
        inline void linesPerTile(size_t);
                
        // Convenience functions
        inline void declare(int, int, int, int) const;

        // Generic resamp entry point from externally created rasters
        void resamp(isce::io::Raster & inputSlc, isce::io::Raster & outputSlc,
                    isce::io::Raster & rgOffsetRaster, isce::io::Raster & azOffsetRaster,
                    int inputBand=1, bool flatten=false, bool isComplex=true, int rowBuffer=40,
                    int chipSize=isce::core::SINC_ONE);

        // Generic resamp entry point: use filenames to create rasters
        void resamp(const std::string & inputFilename, const std::string & outputFilename,
                    const std::string & rgOffsetFilename, const std::string & azOffsetFilename,
                    int inputBand=1, bool flatten=false, bool isComplex=true, int rowBuffer=40,
                    int chipSize=isce::core::SINC_ONE);
        
    // Data members
    protected:
        // Number of lines per tile
        size_t _linesPerTile = 1000;
        // Band number
        int _inputBand;
        // Filename of the input product
        std::string _filename;
        // Flag indicating if we have a reference data (for flattening)
        bool _haveRefData;
        // Interpolator pointer
        isce::core::Interpolator<std::complex<float>> * _interp;

        // Polynomials and LUTs
        isce::core::Poly2d _rgCarrier;            // range carrier polynomial
        isce::core::Poly2d _azCarrier;            // azimuth carrier polynomial
        isce::core::LUT2d<double> _dopplerLUT;

        // Variables ingested from a Swath
        double _startingRange;
        double _rangePixelSpacing;
        double _sensingStart;
        double _prf;
        double _wavelength;
        double _refStartingRange;
        double _refRangePixelSpacing;
        double _refWavelength;

    // Protected functions
    protected:
        
        // Tile initialization for input offsets
        void _initializeOffsetTiles(Tile_t &, isce::io::Raster &, isce::io::Raster &,
                                    isce::image::Tile<float> &,
                                    isce::image::Tile<float> &, int);

        // Tile initialization for input SLC data
        void _initializeTile(Tile_t &, isce::io::Raster &,
                             const isce::image::Tile<float> &,
                             int, int, int);

        // Tile transformation
        void _transformTile(Tile_t & tile,
                            isce::io::Raster & outputSlc,
                            const isce::image::Tile<float> & rgOffTile,
                            const isce::image::Tile<float> & azOffTile,
                            int inLength, bool flatten,
                            int chipSize);

        // Convenience functions
        inline int _computeNumberOfTiles(int, int);

        // Initialize interpolator pointer
        void _prepareInterpMethods(isce::core::dataInterpMethod, int);

        // Set radar parameters from an isce::product::Swath
        inline void _setDataFromSwath(const isce::product::Swath & swath);

        // Set reference radar parameters from an isce::product::Swath (for flattening)
        inline void _setRefDataFromSwath(const isce::product::Swath & swath);
};

// Get inline implementations for ResampSlc
#define ISCE_IMAGE_RESAMPSLC_ICC
#include "ResampSlc.icc"
#undef ISCE_IMAGE_RESAMPSLC_ICC

#endif

// end of file
