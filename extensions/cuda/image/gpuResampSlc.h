//-*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright: 2018

#ifndef ISCE_CUDA_GEOMETRY_GPURESAMPSLC_H
#define ISCE_CUDA_GEOMETRY_GPURESAMPSLC_H

#include <complex>

// isce::core
#include "isce/core/Poly2d.h"

// isce:io:
#include  "isce/io/Raster.h"

// isce::image
#include "isce/image/Tile.h"

// isce::product
#include "isce/product/Product.h"

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
                                const isce::core::Poly2d & doppler,
                                int inLength, bool flatten);
        }
    }
}


#endif

// end of file
