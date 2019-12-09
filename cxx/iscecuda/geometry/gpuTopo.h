//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright: 2017-2018

#pragma once

#include <isce/core/forward.h>
#include <isce/geometry/forward.h>

namespace isce {
    namespace cuda {
        namespace geometry {
            // C++ interface for running topo for a block of data on GPU
            void runGPUTopo(const isce::core::Ellipsoid & ellipsoid,
                            const isce::core::Orbit & orbit,
                            const isce::core::LUT1d<double> & doppler,
                            isce::geometry::DEMInterpolator & demInterp,
                            isce::geometry::TopoLayers & layers,
                            size_t lineStart,
                            int lookSide,
                            int epsgOut,
                            size_t numberAzimuthLooks,
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
