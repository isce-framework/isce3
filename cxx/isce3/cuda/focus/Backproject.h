#pragma once

#include <isce3/container/forward.h>
#include <isce3/cuda/container/forward.h>
#include <isce3/cuda/geometry/forward.h>
#include <isce3/geometry/forward.h>

#include <complex>

#include <isce3/core/Kernels.h>
#include <isce3/focus/Backproject.h>
#include <isce3/focus/DryTroposphereModel.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

namespace isce3 { namespace cuda { namespace focus {

using isce3::focus::DryTroposphereModel;
using isce3::geometry::detail::Geo2RdrParams;
using isce3::geometry::detail::Rdr2GeoParams;

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
 * \param[in]  kernel          1D interpolation kernel
 * \param[in]  dry_tropo_model Dry troposphere path delay model
 * \param[in]  rdr2geo_params  rdr2geo configuration parameters
 * \param[in]  geo2rdr_params  geo2rdr configuration parameters
 * \param[in]  batch           Number of range-compressed data lines per batch
 */
// XXX must pass dem by non-const reference
// XXX const gpuDEMInterpolator cannot be copied due to implementation details
template<class Kernel>
void backproject(std::complex<float>* out,
                 const isce3::cuda::container::RadarGeometry& out_geometry,
                 const std::complex<float>* in,
                 const isce3::cuda::container::RadarGeometry& in_geometry,
                 isce3::cuda::geometry::gpuDEMInterpolator& dem, double fc,
                 double ds, const Kernel& kernel,
                 DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
                 const Rdr2GeoParams& rdr2geo_params = {},
                 const Geo2RdrParams& geo2rdr_params = {}, int batch = 1024);

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
 * \param[in]  kernel          1D interpolation kernel
 * \param[in]  dry_tropo_model Dry troposphere path delay model
 * \param[in]  rdr2geo_params  rdr2geo configuration parameters
 * \param[in]  geo2rdr_params  geo2rdr configuration parameters
 * \param[in]  batch           Number of range-compressed data lines per batch
 */
void backproject(std::complex<float>* out,
                 const isce3::container::RadarGeometry& out_geometry,
                 const std::complex<float>* in,
                 const isce3::container::RadarGeometry& in_geometry,
                 const isce3::geometry::DEMInterpolator& dem, double fc,
                 double ds, const isce3::core::Kernel<float>& kernel,
                 DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
                 const Rdr2GeoParams& rdr2geo_params = {},
                 const Geo2RdrParams& geo2rdr_params = {}, int batch = 1024);

}}} // namespace isce3::cuda::focus
