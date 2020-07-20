#pragma once
#include <cstddef>
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>
#include <isce3/product/forward.h>

namespace isce { namespace geocode {

/**
 * Geocode SLC
 * \param[out] outputRaster  output raster for the geocoded SLC
 * \param[in]  inputRaster   input raster of the SLC in radar coordinates
 * \param[in]  demRaster     raster of the DEM
 * \param[in]  radarGrid     radar grid parameters
 * \param[in]  geoGrid       geo grid parameters
 * \param[in]  orbit             orbit
 * \param[in]  nativeDoppler     2D LUT Doppler of the SLC image 
 * \param[in]  imageGridDoppler  2D LUT Doppler of the image grid 
 * \param[in]  ellipsoid         ellipsoid object 
 * \param[in]  thresholdGeo2rdr  threshold for geo2rdr computations 
 * \param[in]  numiterGeo2rdr    maximum number of iterations for Geo2rdr convergence
 * \param[in]  linesPerBlock     number of lines in each block
 * \param[in]  demBlockMargin    margin of a DEM block in degrees
 * \param[in]  flatten           flag to flatten the geocoded SLC
 */
void geocodeSlc(isce::io::Raster& outputRaster, isce::io::Raster& inputRaster,
                isce::io::Raster& demRaster,
                const isce::product::RadarGridParameters& radarGrid,
                const isce::product::GeoGridParameters& geoGrid,
                const isce::core::Orbit& orbit,
                const isce::core::LUT2d<double>& nativeDoppler,
                const isce::core::LUT2d<double>& imageGridDoppler,
                const isce::core::Ellipsoid& ellipsoid,
                const double& thresholdGeo2rdr, const int& numiterGeo2rdr,
                const size_t& linesPerBlock, const double& demBlockMargin,
                const bool flatten = true);

}} // namespace isce::geocode
