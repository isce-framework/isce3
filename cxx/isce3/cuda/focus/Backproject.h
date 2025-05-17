#pragma once

#include <isce3/container/forward.h>
#include <isce3/cuda/container/forward.h>
#include <isce3/cuda/geometry/forward.h>
#include <isce3/geometry/forward.h>

#include <complex>

#include <isce3/core/Kernels.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/focus/Backproject.h>
#include <isce3/focus/DryTroposphereModel.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

namespace isce3 { namespace cuda { namespace focus {

using isce3::focus::DryTroposphereModel;
using isce3::geometry::detail::Geo2RdrBracketParams;
using isce3::geometry::detail::Rdr2GeoBracketParams;

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
 * \param[out] height          Height of each pixel in meters above ellipsoid
 *
 * \returns Non-zero error code if geometry fails to converge for any pixel,
 *          and the values for these pixels are set to NaN.
 */
// XXX must pass dem by non-const reference
// XXX const gpuDEMInterpolator cannot be copied due to implementation details
template<class Kernel>
isce3::error::ErrorCode
backproject(std::complex<float>* out,
            const isce3::cuda::container::RadarGeometry& out_geometry,
            const std::complex<float>* in,
            const isce3::cuda::container::RadarGeometry& in_geometry,
            isce3::cuda::geometry::gpuDEMInterpolator& dem, double fc,
            double ds, const Kernel& kernel,
            DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
            const Rdr2GeoBracketParams& rdr2geo_params = {},
            const Geo2RdrBracketParams& geo2rdr_params = {},
            int batch = 1024, float* height = nullptr);

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
 * \param[out] height          Height of each pixel in meters above ellipsoid
 *
 * \returns Non-zero error code if geometry fails to converge for any pixel,
 *          and the values for these pixels are set to NaN.
 */
isce3::error::ErrorCode
backproject(std::complex<float>* out,
            const isce3::container::RadarGeometry& out_geometry,
            const std::complex<float>* in,
            const isce3::container::RadarGeometry& in_geometry,
            const isce3::geometry::DEMInterpolator& dem, double fc,
            double ds, const isce3::core::Kernel<float>& kernel,
            DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
            const Rdr2GeoBracketParams& rdr2geo_params = {},
            const Geo2RdrBracketParams& geo2rdr_params = {},
            int batch = 1024, float* height = nullptr);


std::tuple<
        isce3::error::ErrorCode,
        std::unique_ptr<std::complex<float>[]>, // image
        std::unique_ptr<float[]>> // height
backprojectToPolarGrid(
        const std::complex<float>* in,
        const isce3::core::Linspace<double>& in_slant_range,
        const std::vector<isce3::core::Vec3>& pos,
        const std::vector<isce3::core::Vec3>& vel,
        const isce3::focus::PolarGrid& out_grid,
        const isce3::geometry::DEMInterpolator& dem,
        double fc,
        const isce3::core::Kernel<float>& kernel,
        DryTroposphereModel dry_tropo_model,
        const Rdr2GeoBracketParams& r2g_params = {});

isce3::error::ErrorCode
projectPolarToGeo(
        std::complex<float>* geo_image,
        const isce3::core::Vec3* geo_points,
        const size_t n,
        const isce3::focus::PolarGrid& grid,
        const std::complex<float>* polar_image,
        const double wavelength,
        const isce3::signal::NFFT2dParams& params);

template <class SequenceType>
isce3::error::ErrorCode
accumulatePolarImagesToRadarGrid(std::complex<float>* out,
        const isce3::container::RadarGeometry& out_geometry,
        const isce3::core::Orbit& in_orbit,
        const isce3::core::LUT2d<double>& in_doppler,
        const std::vector<isce3::focus::PolarGrid>& grids,
        const SequenceType& image_interpolators,
        const isce3::geometry::DEMInterpolator& dem, double fc, double ds,
        const Rdr2GeoBracketParams& r2g_params,
        const Geo2RdrBracketParams& g2r_params,
        float* height);

}}} // namespace isce3::cuda::focus

#define ISCE_CUDA_FOCUS_BACKPROJECT_ICC
#include "Backproject.icc"
#undef ISCE_CUDA_FOCUS_BACKPROJECT_ICC