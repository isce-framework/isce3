//-*- coding: utf-8 -*_
//
// Author: Bryan V. Riel
// Copyright: 2017-2018

#ifndef ISCE_CUDA_GEOMETRY_TOPO_H
#define ISCE_CUDA_GEOMETRY_TOPO_H

// isce::geometry
#include "isce/geometry/Topo.h"

// Declaration
namespace isce {
    namespace cuda {
        namespace geometry {
            class Topo;
        }
    }
}

// CUDA Topo class definition
class isce::cuda::geometry::Topo : public isce::geometry::Topo {

    public:
        // Constructor from Product
        inline Topo(isce::product::Product & product) :
            isce::geometry::Topo(product) {}

        // Constructor from isce::core objects
        inline Topo(const isce::core::Ellipsoid & ellps,
                    const isce::core::Orbit & orbit,
                    const isce::core::Poly2d & doppler,
                    const isce::core::Metadata & meta) :
            isce::geometry::Topo(ellps, orbit, doppler, meta) {}

};

#endif

// end of file
