// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2018


#ifndef __ISCE_CUDA_IMAGE_GPUIMAGEMODE_H__
#define __ISCE_CUDA_IMAGE_GPUIMAGEMODE_H__

// isce::product
#include <isce/product/ImageMode.h>

// Declaration
namespace isce {
    namespace cuda {
        namespace image {
            class gpuImageMode;
        }
    }
}


// gpuImageMode struct declaration
// parameters reduced from original ImageMode to what's needed in gpuResamp
struct isce::cuda::image::gpuImageMode {
    
        gpuImageMode() :
            prf(0.0), startingRange(0.0), wavelength(0.0), rangePixelSpacing(0.), isRefMode(false) {}
        gpuImageMode(const isce::product::ImageMode im) : 
            prf(im.prf()), 
            startingRange(im.startingRange()), 
            wavelength(im.wavelength()), 
            rangePixelSpacing(im.rangePixelSpacing()),
            isRefMode(true) {}
        // Instrument related data
        double prf;
        double startingRange;
        double wavelength;
        double rangePixelSpacing;
        bool   isRefMode;
};

#endif
