#pragma once
#include <cstddef>
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>
#include <isce3/product/forward.h>

namespace isce3 { namespace geocode {

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
void geocodeSlc(isce3::io::Raster& outputRaster, isce3::io::Raster& inputRaster,
                isce3::io::Raster& demRaster,
                const isce3::product::RadarGridParameters& radarGrid,
                const isce3::product::GeoGridParameters& geoGrid,
                const isce3::core::Orbit& orbit,
                const isce3::core::LUT2d<double>& nativeDoppler,
                const isce3::core::LUT2d<double>& imageGridDoppler,
                const isce3::core::Ellipsoid& ellipsoid,
                const double& thresholdGeo2rdr, const int& numiterGeo2rdr,
                const size_t& linesPerBlock, const double& demBlockMargin,
                const bool flatten = true);

}} // namespace isce3::geocode
