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
class isce3::cuda::geometry::Topo : public isce3::geometry::Topo {

    public:
        /** Constructor from RadarGridProduct */
        inline Topo(const isce3::product::RadarGridProduct & product,
                    char frequency = 'A',
                    bool nativeDoppler = false) :
            isce3::geometry::Topo(product, frequency, nativeDoppler){}

        inline Topo(const isce3::product::RadarGridParameters & radarGrid,
             const isce3::core::Orbit & orbit,
             const isce3::core::Ellipsoid & ellipsoid,
             const isce3::core::LUT2d<double> & doppler = {}) :
            isce3::geometry::Topo(radarGrid, orbit, ellipsoid, doppler) {}

        /** Constructor from isce3::core objects */
        inline Topo(const isce3::core::Ellipsoid & ellps,
                    const isce3::core::Orbit & orbit,
                    const isce3::core::LUT2d<double> & doppler,
                    const isce3::core::Metadata & meta) :
            isce3::geometry::Topo(ellps, orbit, doppler, meta) {}

        /** Run topo - main entrypoint; internal creation of topo rasters */
        void topo(isce3::io::Raster &, const std::string &);

        /** Run topo with externally created topo rasters (plus mask raster) */
        void topo(isce3::io::Raster & demRaster,
                  isce3::io::Raster * xRaster = nullptr,
                  isce3::io::Raster * yRaster = nullptr,
				  isce3::io::Raster * heightRaster = nullptr,
                  isce3::io::Raster * incRaster = nullptr,
				  isce3::io::Raster * hdgRaster = nullptr,
                  isce3::io::Raster * localIncRaster = nullptr,
				  isce3::io::Raster * localPsiRaster = nullptr,
                  isce3::io::Raster * simRaster = nullptr,
				  isce3::io::Raster * maskRaster = nullptr,
                  isce3::io::Raster * groundToSatEastRaster = nullptr,
                  isce3::io::Raster * groundToSatNorthRaster = nullptr);

        /** Run topo - main entrypoint; internal creation of topo rasters */
        void topo(isce3::io::Raster &, isce3::geometry::TopoLayers &);

    private:
        // Generate layover/shadow masks using an orbit
        void _setLayoverShadowWithOrbit(const isce3::core::Orbit & orbit,
                                        isce3::geometry::TopoLayers & layers,
                                        isce3::geometry::DEMInterpolator & demInterp,
                                        size_t lineStart,
                                        size_t block,
                                        size_t n_blocks);
};
