//-*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright: 2018

#pragma once

#include "forward.h"

#include <complex>
#include <thrust/complex.h>

#include <isce3/core/forward.h>
#include <isce3/image/forward.h>
#include <isce3/io/forward.h>
#include <isce3/cuda/core/forward.h>

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
