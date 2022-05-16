#pragma once
#include <cstddef>
#include <isce3/core/forward.h>
#include <isce3/core/Poly2d.h>
#include <isce3/io/forward.h>
#include <isce3/product/forward.h>

namespace isce3 { namespace geocode {

/**
 * Geocode SLC to a given geogrid
 *
 * \tparam[in]  AzRgFunc  2-D real-valued function of azimuth and range
 *
 * \param[out] outputRaster  output raster for the geocoded SLC
 * \param[in]  inputRaster   input raster of the SLC in radar coordinates
 * \param[in]  demRaster     raster of the DEM
 * \param[in]  radarGrid     radar grid parameters
 * \param[in]  geoGrid       geo grid parameters
 * \param[in]  orbit            orbit
 * \param[in]  nativeDoppler    2D LUT Doppler of the SLC image
 * \param[in]  imageGridDoppler 2D LUT Doppler of the image grid
 * \param[in]  ellipsoid        ellipsoid object
 * \param[in]  thresholdGeo2rdr threshold for geo2rdr computations
 * \param[in]  numiterGeo2rdr   maximum number of iterations for Geo2rdr convergence
 * \param[in]  linesPerBlock    number of lines in each block
 * \param[in]  flatten          flag to flatten the geocoded SLC
 * \param[in]  azCarrier        azimuth carrier phase of the SLC data, in radians, as a function of azimuth and range
 * \param[in]  rgCarrier        range carrier phase of the SLC data, in radians, as a function of azimuth and range
 * \param[in]  invalidValue     invalid pixel fill value
 */
template<typename AzRgFunc = isce3::core::Poly2d>
void geocodeSlc(isce3::io::Raster& outputRaster, isce3::io::Raster& inputRaster,
                isce3::io::Raster& demRaster,
                const isce3::product::RadarGridParameters& radarGrid,
                const isce3::product::GeoGridParameters& geoGrid,
                const isce3::core::Orbit& orbit,
                const isce3::core::LUT2d<double>& nativeDoppler,
                const isce3::core::LUT2d<double>& imageGridDoppler,
                const isce3::core::Ellipsoid& ellipsoid,
                const double& thresholdGeo2rdr, const int& numiterGeo2rdr,
                const size_t& linesPerBlock,
                const bool flatten = true,
                const AzRgFunc& azCarrier = AzRgFunc(),
                const AzRgFunc& rgCarrier = AzRgFunc(),
                const std::complex<float> invalidValue =
                    std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                        std::numeric_limits<float>::quiet_NaN()));

/**
 * Geocode SLC to a slice of a given geogrid
 *
 * \tparam[in]  AzRgFunc  2-D real-valued function of azimuth and range
 *
 * \param[out] outputRaster     output raster for the geocoded SLC
 * \param[in]  inputRaster      input raster of the SLC in radar coordinates
 * \param[in]  demRaster        raster of the DEM
 * \param[in]  radarGrid        full sized radar grid parameters
 * \param[in]  slicedRadarGrid  sliced radar grid parameters
 * \param[in]  geoGrid          geo grid parameters
 * \param[in]  orbit            orbit
 * \param[in]  nativeDoppler    2D LUT Doppler of the SLC image
 * \param[in]  imageGridDoppler 2D LUT Doppler of the image grid
 * \param[in]  ellipsoid        ellipsoid object
 * \param[in]  thresholdGeo2rdr threshold for geo2rdr computations
 * \param[in]  numiterGeo2rdr   maximum number of iterations for Geo2rdr convergence
 * \param[in]  linesPerBlock    number of lines in each block
 * \param[in]  flatten          flag to flatten the geocoded SLC
 * \param[in]  azCarrier        azimuth carrier phase of the SLC data, in radians, as a function of azimuth and range
 * \param[in]  rgCarrier        range carrier phase of the SLC data, in radians, as a function of azimuth and range
 * \param[in]  invalidValue     invalid pixel fill value
 */
template<typename AzRgFunc = isce3::core::Poly2d>
void geocodeSlc(isce3::io::Raster& outputRaster, isce3::io::Raster& inputRaster,
                isce3::io::Raster& demRaster,
                const isce3::product::RadarGridParameters& radarGrid,
                const isce3::product::RadarGridParameters& slicedRadarGrid,
                const isce3::product::GeoGridParameters& geoGrid,
                const isce3::core::Orbit& orbit,
                const isce3::core::LUT2d<double>& nativeDoppler,
                const isce3::core::LUT2d<double>& imageGridDoppler,
                const isce3::core::Ellipsoid& ellipsoid,
                const double& thresholdGeo2rdr, const int& numiterGeo2rdr,
                const size_t& linesPerBlock,
                const bool flatten = true,
                const AzRgFunc& azCarrier = AzRgFunc(),
                const AzRgFunc& rgCarrier = AzRgFunc(),
                const std::complex<float> invalidValue =
                    std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                        std::numeric_limits<float>::quiet_NaN()));

}} // namespace isce3::geocode
