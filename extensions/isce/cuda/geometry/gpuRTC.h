#pragma once

#include <isce/io/forward.h>
#include <isce/product/forward.h>

namespace isce { namespace cuda { namespace geometry {
    void facetRTC(isce::product::Product& product,
                  isce::io::Raster& dem,
                  isce::io::Raster& out_raster,
                  char frequency = 'A');
}}}
