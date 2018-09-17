//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright: 2017-2018

#ifndef ISCE_CUDA_GEOMETRY_GPUTOPO_H
#define ISCE_CUDA_GEOMETRY_GPUTOPO_H

// isce::core
#include "isce/core/Ellipsoid.h"
#include "isce/core/Orbit.h"
#include "isce/core/Poly2d.h"
// isce::product
#include  "isce/product/ImageMode.h"
// isce::geometry
#include "isce/geometry/DEMInterpolator.h"
#include "isce/geometry/TopoLayers.h"

namespace isce {
    namespace cuda {
        namespace geometry {
            // C++ interface for running topo for a block of data on GPU
            void runGPUTopo(const isce::core::Ellipsoid & ellipsoid,
                            const isce::core::Orbit & orbit,
                            const isce::core::Poly2d & doppler,
                            const isce::product::ImageMode & mode,
                            isce::geometry::DEMInterpolator & demInterp,
                            isce::geometry::TopoLayers & layers,
                            size_t lineStart, int lookSide, int epsgOut,
                            double threshold, int numiter, int extraiter);
        }
    }
}

#endif

// end of file
