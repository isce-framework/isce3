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

const int SINC_ONE = isce::core::SINC_ONE;
const int SINC_HALF = isce::core::SINC_HALF;

namespace isce { namespace geocode {

    /**
    * Geocode SLC 
    * \param[out] outputRaster  output raster for the geocoded SLC
    * \param[in]  inputRaster   input raster of the SLC to be geocoded
    * \param[in]  demRaster     raster object of the DEM
    * \param[in]  radarGrid     radar grid parameters
    * \param[in]  geoGrid       geo grid parameters
    * \param[in]  orbit             orbit 
    * \param[in]  nativeDoppler     2D LUT Doppler of the SLC image (used for interpolations and resampling)
    * \param[in]  imageGridDoppler  2D LUT Doppler of the image grid used for geometrical computations (is zero for zero Doppler SLCs)
    * \param[in]  ellipsoid         ellipsoid object
    * \param[in]  thresholdGeo2rdr  threshold for geo2rdr computations
    * \param[in]  numiterGeo2rdr    maximum number of iterations for Geo2rdr convergence
    * \param[in]  linesPerBlock     number lines in each block
    * \param[in]  demBlockMargin    margin of the DEM
    * \param[in]  sincLength        length of the sinc interpolator used in geocoding
    * \param[in]  flatten           flag which determines if the geocoded SLC needs to be flattened
    */
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
        const int sincLength = isce::core::SINC_ONE,
        const bool flatten = true);

}
}

