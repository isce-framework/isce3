// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright 2018
//

#ifndef ISCE_CUDA_IMAGE_RESAMPSLC_H
#define ISCE_CUDA_IMAGE_RESAMPSLC_H

// isce::image
#include "isce/image/ResampSlc.h"

// Declaration
namespace isce { 
    namespace cuda { 
        namespace image {
            class ResampSlc;
        }
    }
}

// Definition
class isce::cuda::image::ResampSlc : public isce::image::ResampSlc {

    public:
        // Meta-methods
        // Default constructor
        inline ResampSlc();
        // Constructor from an isce::product::Product
        inline ResampSlc(const isce::product::Product &product) :
            isce::image::ResampSlc(product) {};

        // Constructor from isce::core::Poly2d and isce::product::ImageMode
        inline ResampSlc(const isce::core::Poly2d &poly, const isce::product::ImageMode &imageMode) :
            isce::image::ResampSlc(poly, imageMode) {};

        // Constructor from isce::core objects
        inline ResampSlc(const isce::core::Poly2d &poly, const isce::core::Metadata &metaData) :
            isce::image::ResampSlc(poly, metaData) {};

        // All resamp need? to be redefined to ensure derived functions used
        // Generic resamp entry point from externally created rasters
        void resamp(isce::io::Raster & inputSlc, isce::io::Raster & outputSlc,
                    isce::io::Raster & rgOffsetRaster, isce::io::Raster & azOffsetRaster,
                    int inputBand, bool flatten=false, bool isComplex=true, int rowBuffer=40);

        // Generic resamp entry point: use filenames to create rasters
        void resamp(const std::string & inputFilename, const std::string & outputFilename,
                    const std::string & rgOffsetFilename, const std::string & azOffsetFilename,
                    int inputBand, bool flatten=false, bool isComplex=true, int rowBuffer=40);

        // Main product-based resamp entry point
        void resamp(const std::string & outputFilename, const std::string & polarization,
                    const std::string & rgOffsetFilename, const std::string & azOffsetFilename,
                    bool flatten=false, bool isComplex=true, int rowBuffer=40);

    // Data members
    private:
        // Number of lines per tile
        size_t _linesPerTile = 1000;
};

#endif

// end of file
