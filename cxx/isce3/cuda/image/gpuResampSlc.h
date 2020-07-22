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

using isce3::cuda::core::gpuSinc2dInterpolator;

namespace isce3 {
    namespace cuda {
        namespace image {
            // C++ interface for running topo for a block of data on GPU
            // Tile transformation
            void gpuTransformTile(
               isce3::image::Tile<std::complex<float>> & tile,
               isce3::io::Raster & outputSlc,
               isce3::image::Tile<float> & rgOffTile,
               isce3::image::Tile<float> & azOffTile,
               const isce3::core::Poly2d & rgCarrier,
               const isce3::core::Poly2d & azCarrier,
               const isce3::core::LUT1d<double> & dopplerLUT,
               isce3::cuda::core::gpuSinc2dInterpolator<thrust::complex<float>> interp,
               int inWidth, int inLength, double startingRange, double rangePixelSpacing,
               double prf, double wavelength, double refStartingRange,
               double refRangePixelSpacing, double refWavelength,
               bool flatten, int chipSize
            );
        }
    }
}
