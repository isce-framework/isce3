//-*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright: 2018

#ifndef ISCE_CUDA_GEOMETRY_GPURESAMPSLC_H
#define ISCE_CUDA_GEOMETRY_GPURESAMPSLC_H

#include <complex>

// isce::core
#include "isce/core/Poly2d.h"
#include "isce/core/LUT1d.h"

// isce:io:
#include  "isce/io/Raster.h"

// isce::image
#include "isce/image/Tile.h"

// isce::product
#include "isce/product/ImageMode.h"

// isce::cuda::core
#include "isce/cuda/core/gpuComplex.h"
#include "isce/cuda/core/gpuInterpolator.h"

using isce::cuda::core::gpuSinc2dInterpolator;
using isce::cuda::core::gpuComplex;

namespace isce {
    namespace cuda {
        namespace image {
            // C++ interface for running topo for a block of data on GPU
            // Tile transformation
            void gpuTransformTile(isce::image::Tile<std::complex<float>> & tile,
                                isce::io::Raster & outputSlc,
                                isce::image::Tile<float> & rgOffTile,
                                isce::image::Tile<float> & azOffTile,
                                const isce::core::Poly2d & rgCarrier,
                                const isce::core::Poly2d & azCarrier,
                                const isce::core::LUT1d<double> & dopplerLUT,
                                isce::product::ImageMode mode,       // image mode for image to be resampled
                                isce::product::ImageMode refMode,    // image mode for reference master image
                                bool haveRefMode,
                                gpuSinc2dInterpolator<gpuComplex<float>> interp,
                                int inWidth, int inLength, bool flatten, 
                                int chipSize);
        }
    }
}


#endif

// end of file
