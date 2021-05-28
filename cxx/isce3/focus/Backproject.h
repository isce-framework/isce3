#pragma once

#include <isce3/container/forward.h>
#include <isce3/core/forward.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>
#include <isce3/geometry/forward.h>

#include <complex>

#include "DryTroposphereModel.h"

namespace isce3 {
namespace focus {

/**
 * Focus in azimuth via time-domain backprojection
 *
 * \param[out] out             Output focused signal data
 * \param[in]  out_geometry    Target output grid, orbit, & doppler to focus to
 * \param[in]  in              Input range-compressed signal data
 * \param[in]  in_geometry     Input data grid, orbit, & doppler
 * \param[in]  dem             DEM
 * \param[in]  fc              Center frequency (Hz)
 * \param[in]  ds              Desired azimuth resolution (m)
 * \param[in]  kernel          1-D interpolation kernel
 * \param[in]  dry_tropo_model Dry troposphere path delay model
 * \param[in]  r2g_params      rdr2geo configuration parameters
 * \param[in]  g2r_params      geo2rdr configuration parameters
 */
void backproject(std::complex<float>* out,
                 const isce3::container::RadarGeometry& out_geometry,
                 const std::complex<float>* in,
                 const isce3::container::RadarGeometry& in_geometry,
                 const isce3::geometry::DEMInterpolator& dem,
                 double fc,
                 double ds,
                 const isce3::core::Kernel<float>& kernel,
                 DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
                 const isce3::geometry::detail::Rdr2GeoParams& r2g_params = {},
                 const isce3::geometry::detail::Geo2RdrParams& g2r_params = {});

} // namespace focus
} // namespace isce3
