//-*- coding: utf-8 -*_
//
// Author: Bryan V. Riel
// Copyright: 2017-2018

#ifndef ISCE_CUDA_GEOMETRY_GEO2RDR_H
#define ISCE_CUDA_GEOMETRY_GEO2RDR_H

// isce::geometry
#include "isce/geometry/Geo2rdr.h"

// Declaration
namespace isce {
    namespace cuda {
        namespace geometry {
            class Geo2rdr;
        }
    }
}

// CUDA Topo class definition
class isce::cuda::geometry::Geo2rdr : public isce::geometry::Geo2rdr {

    public:
        // Constructor from Product
        inline Geo2rdr(isce::product::Product & product) :
            isce::geometry::Geo2rdr(product) {}

        // Constructor from isce::core objects
        inline Geo2rdr(const isce::core::Ellipsoid & ellps,
                       const isce::core::Orbit & orbit,
                       const isce::core::Poly2d & doppler,
                       const isce::core::Metadata & meta) :
            isce::geometry::Geo2rdr(ellps, orbit, doppler, meta) {}

        /** Run geo2rdr with offsets and externally created offset rasters */
        void geo2rdr(isce::io::Raster & topoRaster,
                     isce::io::Raster & rgoffRaster,
                     isce::io::Raster & azoffRaster,
                     double azshift=0.0, double rgshift=0.0);

        /** Run geo2rdr with constant offsets and internally created offset rasters */
        void geo2rdr(isce::io::Raster & topoRaster,
                     const std::string & outdir,
                     double azshift=0.0, double rgshift=0.0);

    private:
        // Default number of lines per block
        size_t _linesPerBlock = 1000;

    private:
        // Print extents and image info
        void _printExtents(pyre::journal::info_t &,
                           double, double, double,
                           double, double, double,
                           size_t, size_t);

        // Check we can interpolate orbit to middle of DEM
        void _checkOrbitInterpolation(double);

        // Compute number of lines per block dynamically from GPU memmory
        void computeLinesPerBlock();
};

#endif

// end of file
