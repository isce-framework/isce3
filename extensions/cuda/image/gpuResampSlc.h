//-*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright: 2018

#ifndef ISCE_CUDA_GEOMETRY_GPURESAMPSLC_H
#define ISCE_CUDA_GEOMETRY_GPURESAMPSLC_H

#include <complex>
#include <thrust/complex.h>

// isce::core
#include "isce/core/Poly2d.h"
#include "isce/core/LUT1d.h"

// isce:io:
#include  "isce/io/Raster.h"

// isce::image
#include "isce/image/Tile.h"

// isce::cuda::core
#include "isce/cuda/core/gpuInterpolator.h"

using isce::cuda::core::gpuSinc2dInterpolator;

namespace isce {
    namespace cuda {
        namespace image {
            // C++ interface for running topo for a block of data on GPU
            // Tile transformation
            void gpuTransformTile(
               isce::image::Tile<std::complex<float>> & tile,
               isce::io::Raster & outputSlc,
               isce::image::Tile<float> & rgOffTile,
               isce::image::Tile<float> & azOffTile,
               const isce::core::Poly2d & rgCarrier,
               const isce::core::Poly2d & azCarrier,
               const isce::core::LUT1d<double> & dopplerLUT,
               isce::cuda::core::gpuSinc2dInterpolator<thrust::complex<float>> interp,
               int inWidth, int inLength, double startingRange, double rangePixelSpacing,
               double prf, double wavelength, double refStartingRange,
               double refRangePixelSpacing, double refWavelength,
               bool flatten, int chipSize
            );
        }
    }
}


#endif

// end of file
