//-*- coding: utf-8 -*_
//
// Author: Bryan V. Riel
// Copyright: 2017-2018

#pragma once

#include "forward.h"

#include <isce3/geometry/forward.h>
#include <isce3/geometry/Topo.h>

// CUDA Topo class definition
/** Transformer from radar geometry coordinates to map coordinates with DEM / reference altitude on GPU
 *
 * See <a href="overview_geometry.html#forwardgeom">geometry overview</a> for a description of the algorithm*/
class isce::cuda::geometry::Topo : public isce::geometry::Topo {

    public:
        /** Constructor from Product */
        inline Topo(const isce::product::Product & product,
                    char frequency = 'A',
                    bool nativeDoppler = false) :
            isce::geometry::Topo(product, frequency, nativeDoppler){}

        inline Topo(const isce::product::RadarGridParameters & radarGrid,
             const isce::core::Orbit & orbit,
             const isce::core::Ellipsoid & ellipsoid,
             const isce::core::LUT2d<double> & doppler = {}) :
            isce::geometry::Topo(radarGrid, orbit, ellipsoid, doppler) {}

        /** Constructor from isce::core objects */
        inline Topo(const isce::core::Ellipsoid & ellps,
                    const isce::core::Orbit & orbit,
                    const isce::core::LUT2d<double> & doppler,
                    const isce::core::Metadata & meta) :
            isce::geometry::Topo(ellps, orbit, doppler, meta) {}

        /** Run topo - main entrypoint; internal creation of topo rasters */
        void topo(isce::io::Raster &, const std::string &);

        /** Run topo with externally created topo rasters */
        void topo(isce::io::Raster & demRaster, isce::io::Raster & xRaster,
                  isce::io::Raster & yRaster, isce::io::Raster & heightRaster,
                  isce::io::Raster & incRaster, isce::io::Raster & hdgRaster,
                  isce::io::Raster & localIncRaster, isce::io::Raster & localPsiRaster,
                  isce::io::Raster & simRaster);

        /** Run topo with externally created topo rasters (plus mask raster) */
        void topo(isce::io::Raster & demRaster, isce::io::Raster & xRaster,
                  isce::io::Raster & yRaster, isce::io::Raster & heightRaster,
                  isce::io::Raster & incRaster, isce::io::Raster & hdgRaster,
                  isce::io::Raster & localIncRaster, isce::io::Raster & localPsiRaster,
                  isce::io::Raster & simRaster, isce::io::Raster & maskRaster);

        /** Run topo - main entrypoint; internal creation of topo rasters */
        void topo(isce::io::Raster &, isce::geometry::TopoLayers &);

    private:
        // Default number of lines per block
        size_t _linesPerBlock = 1000;

        // Compute number of lines per block dynamically from GPU memmory
        void computeLinesPerBlock(isce::io::Raster &,
                                  isce::geometry::TopoLayers &);

        // Generate layover/shadow masks using an orbit
        void _setLayoverShadowWithOrbit(const isce::core::Orbit & orbit,
                                        isce::geometry::TopoLayers & layers,
                                        isce::geometry::DEMInterpolator & demInterp,
                                        size_t lineStart);
};
