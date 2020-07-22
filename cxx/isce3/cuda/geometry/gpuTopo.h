//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright: 2017-2018

#pragma once

#include <isce3/core/forward.h>
#include <isce3/geometry/forward.h>

namespace isce3 {
    namespace cuda {
        namespace geometry {
            // C++ interface for running topo for a block of data on GPU
            void runGPUTopo(const isce3::core::Ellipsoid & ellipsoid,
                            const isce3::core::Orbit & orbit,
                            const isce3::core::LUT1d<double> & doppler,
                            isce3::geometry::DEMInterpolator & demInterp,
                            isce3::geometry::TopoLayers & layers,
                            size_t lineStart,
                            isce3::core::LookSide lookSide,
                            int epsgOut,
                            double startAzUTCTime,
                            double wavelength,
                            double prf,
                            double startingRange,
                            double rangePixelSpacing,
                            double threshold, int numiter, int extraiter,
                            unsigned int & totalconv);
        }
    }
}
