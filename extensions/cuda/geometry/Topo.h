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

        // Run topo - main entrypoint; internal creation of topo rasters
        void topo(isce::io::Raster &, const std::string);

        // Run topo with externally created topo rasters
        void topo(isce::io::Raster & demRaster, isce::io::Raster & xRaster,
                  isce::io::Raster & yRaster, isce::io::Raster & heightRaster,
                  isce::io::Raster & incRaster, isce::io::Raster & hdgRaster,
                  isce::io::Raster & localIncRaster, isce::io::Raster & localPsiRaster,
                  isce::io::Raster & simRaster);

    private:
        // Default number of lines per block
        size_t _linesPerBlock = 1000;

        // Compute number of lines per block dynamically from GPU memmory
        void computeLinesPerBlock(isce::io::Raster &);
};

#endif

// end of file
