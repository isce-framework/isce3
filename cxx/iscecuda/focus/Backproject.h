#pragma once

#include <isce/container/forward.h>
#include <isce/cuda/container/forward.h>
#include <isce/cuda/geometry/forward.h>
#include <isce/geometry/forward.h>

#include <complex>

#include <isce/core/Kernels.h>
#include <isce/focus/Backproject.h>
#include <isce/focus/DryTroposphereModel.h>

namespace isce { namespace cuda { namespace focus {

using isce::focus::DryTroposphereModel;
using isce::focus::Geo2RdrParams;
using isce::focus::Rdr2GeoParams;

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
                 const isce::cuda::container::RadarGeometry& out_geometry,
                 const std::complex<float>* in,
                 const isce::cuda::container::RadarGeometry& in_geometry,
                 isce::cuda::geometry::gpuDEMInterpolator& dem, double fc,
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
                 const isce::container::RadarGeometry& out_geometry,
                 const std::complex<float>* in,
                 const isce::container::RadarGeometry& in_geometry,
                 const isce::geometry::DEMInterpolator& dem, double fc,
                 double ds, const isce::core::Kernel<float>& kernel,
                 DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
                 const Rdr2GeoParams& rdr2geo_params = {},
                 const Geo2RdrParams& geo2rdr_params = {}, int batch = 1024);

}}} // namespace isce::cuda::focus
