#pragma once

#include <isce3/container/forward.h>
#include <isce3/core/forward.h>
#include <isce3/geometry/forward.h>
#include <isce3/product/forward.h>

#include <complex>
#include <memory>
#include <Eigen/Dense>

#include <isce3/error/ErrorCode.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>
#include <isce3/signal/NFFT2d.h>

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
    isce3::core::LookSide look_side;

    PolarGrid() = delete;

    CUDA_HOSTDEV auto width() const { return range.size(); }
    CUDA_HOSTDEV auto length() const { return sin_squint.size(); }
};

std::tuple<isce3::error::ErrorCode,
        std::unique_ptr<std::complex<float>[]>,
        std::unique_ptr<float[]>>
backprojectToPolarGrid(const std::complex<float>* in,
        const isce3::core::Linspace<double>& in_slant_range,
        const std::vector<isce3::core::Vec3>& pos,
        const std::vector<isce3::core::Vec3>& vel,
        const PolarGrid& out_grid,
        const isce3::geometry::DEMInterpolator& dem,
        double fc,
        const isce3::core::Kernel<float>& kernel,
        DryTroposphereModel dry_tropo_model,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params = {});

isce3::error::ErrorCode
backprojectFinalStage(std::complex<float>* out,
        const isce3::container::RadarGeometry& out_geometry,
        const isce3::core::Orbit& in_orbit,
        const isce3::core::LUT2d<double>& in_doppler,
        const std::vector<PolarGrid>& grids,
        const std::vector<isce3::signal::NFFT2d<float>>& image_interpolators,
        const isce3::geometry::DEMInterpolator& dem, double fc, double ds,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
        const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
        float* height);


// WIP stuff to do one polar image at a time.

/**
 * @brief Get the time constant associated with polar angle spacing
 *
 * @param fc        Radar center frequency, Hz
 * @param vs        Satellite velocity (along azimuth axis), m/s
 * @param bandwidth Radar bandwidth, Hz (defaults to zero, e.g., narrow band)
 * @param c         Speed of light, m/s (defaults to vacuum sol)
 * @return          Time constant $T_q$, s
 *
 * This time constant is used to determine the sampling requirement for the
 * sine of the squint angle (dimensionless Doppler)
 *      $$ q = \frac{\vec{v}}{v} \cdot \hat{l} $$
 * where $\vec{v}$ is the velocity and $\hat{l}$ is the line-of-sight direction.
 * Specifically, the Nyquist criterion is
 *      $$ \Delta q \leq \frac{T_q}{T_{sa}} $$
 * where $T_{sa}$ is the time duration of the synthetic aperture.
 *
 * Helps implement equation (11) in @cite yegulalp2013
 */
double
getPolarAngleTimeConstant(const double fc, const double vs,
    const double bandwidth = 0.0, const double c = isce3::core::speed_of_light);

// set up polar grid for a group of pulses
std::tuple<PolarGrid, std::vector<isce3::core::Vec3>, std::vector<isce3::core::Vec3>>
setupPolarGridForPulses(
        const isce3::container::RadarGeometry& in_geometry,
        const Eigen::Ref<const Eigen::VectorXd>& azimuth_time,
        double range_bandwidth,
        double azimuth_resolution,
        double oversample_range = 1.2, double oversample_azimuth = 1.2,
        int num_doppler_eval = 2);

/**
 * @brief Create polar grid capable of sampling data from all input grids.
 *
 * @param grids         List of subaperture grids.
 * @param dem           Digital elevation model reporting height (m) above the
 *                      ellispoid associated with its CRS.
 * @param r2g_params    Root finding parameters for radar2geo
 * @param dq_min        Minimum allowed dimensionless Doppler spacing.
 *                      Necessary for stripmap processing large subapertures.
 * @param tq            Time constant for dimensionless Doppler spacing.
 *                      If not provided it will be inferred from input grids.
 */
PolarGrid
mergePolarGrids(const std::vector<PolarGrid>& grids,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params = {},
    const std::optional<double>& dq_min = {},
    const std::optional<double>& tq = {});

void mergePolarImages(
    const std::vector<PolarGrid>& grids,
    const std::vector<isce3::signal::NFFT2d<float>>& image_interpolators,
    const PolarGrid& output_grid,
    Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> output_image,
    const double fc,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params = {},
    int az_block_size = 1024);

// interpolate polar grid to given set of XYZ positions
isce3::error::ErrorCode
projectPolarToGeo(
        std::complex<float>* geo_image,  // accumulates, so init to zero!
        const isce3::core::Vec3* geo_points,
        const size_t n,
        const PolarGrid& grid,
        const isce3::signal::NFFT2d<float>& nfft,
        const double kw);

// figure out bounds of polar grid in stripmap radar coordinates
std::tuple<double, double, double, double, isce3::error::ErrorCode>
findPolarGridBoundingBoxInRadarCoord(
    const PolarGrid& polar_grid,
    const isce3::core::Orbit& orbit,
    const isce3::core::LUT2d<double>& doppler,
    const double wavelength,
    const isce3::core::LookSide lookside,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
    const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
    const int nextra = 0);

// figure out subset of stripmap radar grid that is covered by a polar grid
std::tuple<isce3::product::RadarGridParameters, isce3::error::ErrorCode>
findPolarGridBoundingBoxInRadarGrid(
    const PolarGrid& polar_grid,
    const isce3::container::RadarGeometry& radar_geom,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params,
    const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params,
    const int nextra = 0);

std::tuple<std::vector<isce3::core::Vec3>, isce3::error::ErrorCode>
computeRadarGridGeoPoints(
    const isce3::container::RadarGeometry& geom,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params);

isce3::error::ErrorCode
computeRadarGridGeoPoints(
    isce3::core::Vec3* points,
    const isce3::container::RadarGeometry& geom,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params);

} // namespace focus
} // namespace isce3
