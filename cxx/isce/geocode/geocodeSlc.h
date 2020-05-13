#pragma once

// isce::core
#include <isce/core/Orbit.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Constants.h>
#include <isce/core/Projections.h>
#include <isce/geometry/DEMInterpolator.h>
#include <isce/geometry/geometry.h>
// isce::io
#include <isce/io/Raster.h>

// isce::product
#include <isce/product/Product.h>
#include <isce/product/RadarGridParameters.h>
#include <isce/product/GeoGridParameters.h>

// isce::geometry
#include <isce/geometry/geometry.h>

#include <isce/geocode/baseband.h>
#include <isce/geocode/geo2rdr.h>
#include <isce/geocode/interpolate.h>
#include <isce/geocode/loadDem.h>

#define modulo_f(a,b) fmod(fmod(a,b)+(b),(b))
const int SINC_ONE = isce::core::SINC_ONE;
const int SINC_HALF = isce::core::SINC_HALF;

namespace isce { namespace geocode {

    void geocodeSlc(isce::io::Raster & outputRaster,
            isce::io::Raster & inputRaster,
        isce::io::Raster & demRaster,
        const isce::product::RadarGridParameters & radarGrid,
        const isce::product::GeoGridParameters & geoGrid,
        const isce::core::Orbit& orbit,
        const isce::core::LUT2d<double>& nativeDoppler,
        const isce::core::LUT2d<double>& imageGridDoppler,
        const isce::core::Ellipsoid & ellipsoid,
        const double & thresholdGeo2rdr,
        const int & numiterGeo2rdr,
        const size_t & linesPerBlock,
        const double & demBlockMargin,
        const int sincLength,
        const bool flatten = true);

}
}

