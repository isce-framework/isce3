//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright: 2017-2018

#ifndef ISCE_CUDA_GEOMETRY_GPUGEO2RDR_H
#define ISCE_CUDA_GEOMETRY_GPUGEO2RDR_H

// isce::core
#include "isce/core/Ellipsoid.h"
#include "isce/core/Orbit.h"
#include "isce/core/LUT1d.h"

namespace isce {
    namespace cuda {
        namespace geometry {
            // C++ interface for running geo2rdr for a block of data on GPU
            void runGPUGeo2rdr(const isce::core::Ellipsoid & ellipsoid,
                               const isce::core::Orbit & orbit,
                               const isce::core::LUT1d<double> & doppler,
                               const std::valarray<double> & x,
                               const std::valarray<double> & y,
                               const std::valarray<double> & hgt,
                               std::valarray<float> & azoff,
                               std::valarray<float> & rgoff,
                               int topoEPSG, size_t lineStart, size_t blockWidth,
                               double t0, double r0, size_t numberAzimuthLooks,
                               size_t numberRangeLooks, size_t length, size_t width,
                               double prf, double rangePixelSpacing, double wavelength,
                               double threshold, double numiter, unsigned int & totalconv);
        }
    }
}

#endif

// end of file
