#pragma once

#include <isce3/container/forward.h>
#include <isce3/core/forward.h>
#include <isce3/geometry/forward.h>

#include <complex>

#include "DryTroposphereModel.h"

namespace isce {
namespace focus {

struct Rdr2GeoParams {
    double threshold = 1e-8;
    int maxiter = 25;
    int extraiter = 15;
};

struct Geo2RdrParams {
    double threshold = 1e-8;
    int maxiter = 50;
    double delta_range = 10.;
};

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
                 const isce::container::RadarGeometry& out_geometry,
                 const std::complex<float>* in,
                 const isce::container::RadarGeometry& in_geometry,
                 const isce::geometry::DEMInterpolator& dem,
                 double fc,
                 double ds,
                 const isce::core::Kernel<float>& kernel,
                 DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
                 const Rdr2GeoParams& r2g_params = {},
                 const Geo2RdrParams& g2r_params = {});

} // namespace focus
} // namespace isce
