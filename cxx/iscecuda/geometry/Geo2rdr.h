//-*- coding: utf-8 -*_
//
// Author: Bryan V. Riel
// Copyright: 2017-2018

#pragma once

#include "forward.h"

#include <isce/core/forward.h>
#include <isce/geometry/Geo2rdr.h>

// CUDA Topo class definition
/** Transformer from map coordinates to radar geometry coordinates using GPU.
 *
 * See <a href="overview_geometry.html#inversegeom">geometry overview</a> for description of the algorithm. */
class isce::cuda::geometry::Geo2rdr : public isce::geometry::Geo2rdr {

    public:
        /** Constructor from Product */
        inline Geo2rdr(const isce::product::Product & product,
                       char frequency = 'A',
                       bool nativeDoppler = false) :
            isce::geometry::Geo2rdr(product, frequency, nativeDoppler) {}

        /** Constructor from isce::core objects */
        inline Geo2rdr(const isce::core::Ellipsoid & ellps,
                       const isce::core::Orbit & orbit,
                       const isce::core::LUT2d<double> & doppler,
                       const isce::core::Metadata & meta) :
            isce::geometry::Geo2rdr(ellps, orbit, doppler, meta) {}

        inline Geo2rdr(const isce::product::RadarGridParameters & radarGrid,
                const isce::core::Orbit & orbit,
                const isce::core::Ellipsoid & ellipsoid,
                const isce::core::LUT2d<double> & doppler = {}) : 
            isce::geometry::Geo2rdr(radarGrid, orbit, ellipsoid, doppler) {}

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
        /** Print extents and image info */
        void _printExtents(pyre::journal::info_t &,
                           double, double, double,
                           double, double, double,
                           size_t, size_t);

        /** Check we can interpolate orbit to middle of DEM */
        void _checkOrbitInterpolation(double);

        /** Compute number of lines per block dynamically from GPU memmory */
        void computeLinesPerBlock();
};
