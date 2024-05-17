#pragma once

#include <isce3/container/forward.h>
#include <isce3/core/forward.h>
#include <isce3/geometry/forward.h>

#include <complex>
#include <memory>

#include <isce3/error/ErrorCode.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

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
        const isce3::geometry::DEMInterpolator& dem, double fc, double ds,
        const isce3::core::Kernel<float>& kernel,
        DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params = {},
        const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params = {},
        float* height = nullptr);


struct PolarGrid {
    double aztime_start, aztime_end;
    isce3::core::Vec3 origin, axis;
    isce3::core::Linspace<double> range;
    isce3::core::Linspace<double> sin_squint;

    PolarGrid() = delete;

    auto width() const { return range.size(); }
    auto length() const { return sin_squint.size(); }
};

std::tuple<isce3::error::ErrorCode,
        PolarGrid,
        std::unique_ptr<std::complex<float>[]>,
        std::unique_ptr<float[]>>
backprojectFirstStage(const std::complex<float>* in,
        const isce3::container::RadarGeometry& in_geometry,
        const std::vector<double>& in_azimuth_time, double range_bandwidth,
        const isce3::geometry::DEMInterpolator& dem, double fc, double ds,
        const isce3::core::Kernel<float>& kernel,
        DryTroposphereModel dry_tropo_model,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params = {},
        double oversample_range = 1.2, double oversample_azimuth = 1.2);

isce3::error::ErrorCode
backprojectFinalStage(std::complex<float>* out,
        const isce3::container::RadarGeometry& out_geometry,
        const isce3::core::Orbit& in_orbit,
        const isce3::core::LUT2d<double>& in_doppler,
        const std::vector<PolarGrid>& grids,
        const std::vector<const std::complex<float>*>& images,
        const isce3::geometry::DEMInterpolator& dem, double fc, double ds,
        const isce3::core::Kernel<float>& kernel_rg,
        const isce3::core::Kernel<float>& kernel_az,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
        const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
        float* height);

} // namespace focus
} // namespace isce3
