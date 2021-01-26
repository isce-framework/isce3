#pragma once

#include <isce3/io/forward.h>
#include <isce3/product/forward.h>

namespace isce3 { namespace cuda { namespace geometry {
void computeRtc(isce3::product::Product& product, isce3::io::Raster& dem,
                isce3::io::Raster& out_raster, char frequency = 'A');
}}}
