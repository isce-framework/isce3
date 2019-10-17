#pragma once

#include "forward.h"
#include <isce/core/forward.h>
#include <isce/io/forward.h>
#include <isce/product/forward.h>
#include <isce/core/Constants.h>

namespace isce { namespace geometry {
/** @param[in] product input DEM (raster or DEMInterpolator)
  * @param[in] dem input DEM raster
  * @param[out] output raster
  * @param[in] frequency Product frequency
  * @param[in] inputRadiometry Input radiometry. Options: 0: sigma-naught ellipsoid (default);
  *                                                       1: beta/beta-naught.
  * @param[in] outputMode Output mode. Options: 0: gamma-naught area (default); 
  *                                             1: gamma-naught RTC ratio;
  *                                             2: sigma-naught area (under test);
  *                                             3: sigma-naught RTC ratio (under test).
  * */
void facetRTC(isce::product::Product &product,
              isce::io::Raster &dem,
              isce::io::Raster &out_raster,
              char frequency = 'A',
              isce::core::rtcInputRadiometry inputRadiometry = isce::core::rtcInputRadiometry::SIGMA_NAUGHT,
              isce::core::rtcOutputMode outputMode = isce::core::rtcOutputMode::GAMMA_NAUGHT_AREA);

void facetRTC(const isce::product::RadarGridParameters &radarGrid,
              const isce::core::Orbit &orbit,
              const isce::core::LUT2d<double> &dop,
              isce::io::Raster &dem,
              isce::io::Raster &out_raster,
              isce::core::rtcInputRadiometry inputRadiometry = isce::core::rtcInputRadiometry::SIGMA_NAUGHT,
              isce::core::rtcOutputMode outputMode = isce::core::rtcOutputMode::GAMMA_NAUGHT_AREA);

void facetRTC(const isce::product::RadarGridParameters &radarGrid,
              const isce::core::Orbit &orbit,
              const isce::core::LUT2d<double> &dop,
              isce::geometry::DEMInterpolator &dem_interp,
              isce::io::Raster &out_raster,
              isce::core::rtcInputRadiometry inputRadiometry = isce::core::rtcInputRadiometry::SIGMA_NAUGHT,
              isce::core::rtcOutputMode outputMode = isce::core::rtcOutputMode::GAMMA_NAUGHT_AREA);
}}