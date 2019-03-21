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
        // Constructor from an isce::product::Product
        inline ResampSlc(const isce::product::Product &product, char frequency = 'A') :
            isce::image::ResampSlc(product, frequency) {}

        // Constructor from an isce::product::Product and reference product (flattening) 
        inline ResampSlc(const isce::product::Product & product,
                         const isce::product::Product & refProduct,
                         char frequency = 'A') :
            isce::image::ResampSlc(product, refProduct, frequency) {}

        // Constructor from individual components (no flattening) 
        inline ResampSlc(const isce::core::LUT2d<double> & doppler,
                         double startingRange, double rangePixelSpacing,
                         double sensingStart, double prf, double wvl) :
            isce::image::ResampSlc(doppler, startingRange, rangePixelSpacing, sensingStart,
                                   prf, wvl) {}

        // Constructor from individual components (flattening)
        inline ResampSlc(const isce::core::LUT2d<double> & doppler,
                         double startingRange, double rangePixelSpacing,
                         double sensingStart, double prf, double wvl,
                         double refStartingRange, double refRangePixelSpacing,
                         double refWvl) :
            isce::image::ResampSlc(doppler, startingRange, rangePixelSpacing, sensingStart,
                                   prf, wvl, refStartingRange, refRangePixelSpacing, refWvl) {}

        // All resamp need? to be redefined to ensure derived functions used
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
        
};

#endif

// end of file
