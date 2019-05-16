#pragma once

#include "isce/product/Product.h"
#include "isce/io/Raster.h"

namespace isce {
    namespace geometry {
        void facetRTC(isce::product::Product& product,
                      isce::io::Raster& dem,
                      isce::io::Raster& out_raster,
                      char frequency = 'A');

        void facetRTC(isce::product::RadarGridParameters radarGrid,
                              isce::core::Orbit orbit,
                              isce::core::LUT2d<double> dop,
                              isce::io::Raster& dem,
                              isce::io::Raster& out_raster,
                              int lookSide);

    }
}
