#pragma once

#include "forward.h"
#include <isce/core/forward.h>
#include <isce/io/forward.h>
#include <isce/product/forward.h>

namespace isce { namespace geometry {
    void facetRTC(isce::product::Product& product,
                  isce::io::Raster& dem,
                  isce::io::Raster& out_raster,
                  char frequency = 'A');

    void facetRTC(const isce::product::RadarGridParameters& radarGrid,
                  const isce::core::Orbit& orbit,
                  const isce::core::LUT2d<double>& dop,
                  isce::io::Raster& dem,
                  isce::io::Raster& out_raster,
                  const int lookSide);
}}
