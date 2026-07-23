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


/** Structure describing the coordinate system of a polar image. */
struct PolarGrid {
    /** Start azimuth time of the synthetic aperture (s) */
    double aztime_start;

    /** End azimuth time of the synthetic aperture (s); one PRI past the last pulse */
    double aztime_end;

    /** ECEF origin of the polar coordinate system (m).
     *
     * This is the reference point from which polar coordinates are defined,
     * typically computed as the mean platform position over the aperture.
     */
    isce3::core::Vec3 origin;

    /** Unit vector along the azimuth (along-track) axis of the polar grid.
     *
     * This is the normalized mean platform velocity, used to define the
     * azimuth direction of the polar coordinate system.
     */
    isce3::core::Vec3 axis;

    /** Slant range grid (m) */
    isce3::core::Linspace<double> range;

    /** Sine of the squint angle (dimensionless Doppler) grid */
    isce3::core::Linspace<double> sin_squint;

    /** Side looking direction (left or right of flight track) */
    isce3::core::LookSide look_side;

    PolarGrid() = delete;

    CUDA_HOSTDEV auto width() const { return range.size(); }
    CUDA_HOSTDEV auto length() const { return sin_squint.size(); }

    PolarGrid offsetAndResize(int q_off, int r_off, int nq, int nr) const
    {
        using LS = isce3::core::Linspace<double>;
        const auto q = LS(sin_squint[q_off], sin_squint.spacing(), nq);
        const auto r = LS(range[r_off], range.spacing(), nr);
        return PolarGrid {
                aztime_start, aztime_end, origin, axis, r, q, look_side};
    }
};

/**
 * @brief Backproject range-compressed signal into a polar grid.
 *
 * Performs time-domain backprojection of range-compressed SAR signal data
 * onto a polar coordinate grid. For each output pixel in the polar grid,
 * the signal is resampled from the input data by accumulating contributions
 * from all pulses according to the instantaneous slant range, with phase
 * compensation for motion and the troposphere.
 *
 * @param[in]  in              Input range-compressed signal data
 * @param[in]  in_slant_range  Slant range grid of the input data (m)
 * @param[in]  pos             Platform position vectors at each pulse
 *                             (ECEF, m)
 * @param[in]  vel             Platform velocity vectors at each pulse
 *                             (ECEF, m/s)
 * @param[in]  out_grid        Target polar grid to backproject onto
 * @param[in]  dem             DEM
 * @param[in]  fc              Center frequency (Hz)
 * @param[in]  kernel          1-D interpolation kernel
 * @param[in]  dry_tropo_model Dry troposphere path delay model
 * @param[in]  r2g_params      rdr2geo configuration parameters
 *
 * @returns A tuple containing:
 *          - error code (non-zero if geometry fails to converge for
 *            any pixel, in which case values for those pixels are NaN)
 *          - focused signal data on the polar grid (size =
 *            out_grid.width() * out_grid.length())
 *          - per-pixel height above the ellipsoid (m)
 *
 * @see PolarGrid for a description of the polar grid coordinate system.
 */
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

/**
 * @brief Accumulate polar grid images onto an output radar geometry grid.
 *
 * Combines multiple subaperture polar grid images together onto a stripmap
 * radar geometry grid. For each pixel in the output grid, the target
 * position is computed via rdr2geo, the corresponding coherent processing
 * interval is determined via geo2rdr, and the polar image data is
 * accumulated using NFFT-based interpolation.
 *
 * The caller is responsible for allocating the output arrays to the
 * appropriate size (out_geometry.gridLength() * out_geometry.gridWidth()).
 *
 * @param[out] out              Accumulated focused signal data
 * @param[in]  out_geometry     Target output grid, orbit, and Doppler
 * @param[in]  in_orbit         Input data orbit
 * @param[in]  in_doppler       Input data Doppler centroid LUT
 * @param[in]  grids            List of subaperture polar grids
 * @param[in]  image_interpolators  NFFT interpolators for each polar grid
 * @param[in]  dem              Digital elevation model (DEM)
 * @param[in]  fc               Center frequency (Hz)
 * @param[in]  ds               Desired azimuth resolution (m)
 * @param[in]  dry_tropo_model  Dry troposphere path delay model
 * @param[in]  r2g_params       rdr2geo_bracket configuration parameters
 * @param[in]  g2r_params       geo2rdr_bracket configuration parameters
 * @param[out] height           Height of each pixel (m) above the
 *                              ellipsoid (optional, may be nullptr)
 *
 * @returns Non-zero error code if rdr2geo or geo2rdr fails to converge
 *          for any pixel, and the values for these pixels are set to NaN.
 */
isce3::error::ErrorCode
accumulatePolarImagesToRadarGrid(std::complex<float>* out,
        const isce3::container::RadarGeometry& out_geometry,
        const isce3::core::Orbit& in_orbit,
        const isce3::core::LUT2d<double>& in_doppler,
        const std::vector<PolarGrid>& grids,
        const std::vector<isce3::signal::NFFT2dResult<float>>& image_interpolators,
        const isce3::geometry::DEMInterpolator& dem, double fc, double ds,
        const DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
        const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params = {},
        const isce3::geometry::detail::Geo2RdrBracketParams& g2r_params = {},
        float* height = nullptr);



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

/**
 * @brief Set up a polar grid for a group of pulses.
 *
 * Constructs a PolarGrid that covers the synthetic aperture defined by
 * the given azimuth times, along with the platform position and velocity
 * vectors interpolated at each pulse. The grid spacing in the sine of
 * the squint angle and range is determined by the Doppler bandwidth,
 * range bandwidth, and desired azimuth resolution.
 *
 * @param[in]  in_geometry          Input data grid, orbit, & doppler
 * @param[in]  azimuth_time         Azimuth times at which to set up the grid (s)
 * @param[in]  range_bandwidth      Radar range bandwidth (Hz)
 * @param[in]  azimuth_resolution   Desired azimuth resolution (m)
 * @param[in]  oversample_range     Range oversampling factor
 *                                  (defaults to 1.2)
 * @param[in]  oversample_azimuth   Azimuth oversampling factor
 *                                  (defaults to 1.2)
 * @param[in]  num_doppler_eval     Number of range locations to evaluate
 *                                  Doppler centroid for bandwidth estimation
 *                                  (defaults to 2)
 * @param[in]  pri                  Pulse repetition interval (s); if not
 *                                  provided, it is inferred from azimuth_time
 *
 * @returns A tuple containing:
 *          - the constructed PolarGrid
 *          - platform position vectors at each pulse (ECEF, m)
 *          - platform velocity vectors at each pulse (ECEF, m/s)
 *
 * @see PolarGrid for a description of the polar grid coordinate system.
 */
std::tuple<PolarGrid, std::vector<isce3::core::Vec3>, std::vector<isce3::core::Vec3>>
setupPolarGridForPulses(
        const isce3::container::RadarGeometry& in_geometry,
        const Eigen::Ref<const Eigen::VectorXd>& azimuth_time,
        double range_bandwidth,
        double azimuth_resolution,
        double oversample_range = 1.2, double oversample_azimuth = 1.2,
        int num_doppler_eval = 2, std::optional<double> pri = std::nullopt);

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

/**
 * @brief Merge subaperture polar grid images into a single output grid.
 *
 * Combines multiple subaperture polar grid images onto a merged output
 * polar grid. For each pixel in the output grid, the 3D target position
 * is computed via polar2geo, and the input image data is accumulated
 * via NFFT-based interpolation. The output image is expected to be
 * zero-initialized by the caller.
 *
 * @param[in]  grids               List of subaperture input polar grids
 * @param[in]  image_interpolators NFFT interpolators for each input grid
 * @param[in]  output_grid         Merged output polar grid
 * @param[out] output_image        Accumulated output image (must be
 *                                 zero-initialized); dimensions must
 *                                 match output_grid
 * @param[in]  fc                  Center frequency (Hz)
 * @param[in]  dem                 Digital elevation model (DEM)
 * @param[in]  r2g_params          rdr2geo_bracket configuration parameters
 * @param[in]  az_block_size       Number of azimuth rows to process at
 *                                 a time (defaults to 1024)
 *
 * @throws isce3::except::LengthError if output image dimensions or
 *         grid/interpolator counts are inconsistent
 * @throws isce3::except::InvalidArgument if look directions are
 *         inconsistent or az_block_size is negative
 * @throws isce3::except::DomainError if polar2geo fails to converge
 * @throws isce3::except::RuntimeError if NFFT interpolation fails
 */
void mergePolarImages(
    const std::vector<PolarGrid>& grids,
    const std::vector<isce3::signal::NFFT2dResult<float>>& image_interpolators,
    const PolarGrid& output_grid,
    Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> output_image,
    const double fc,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params = {},
    int az_block_size = 1024);

/**
 * @brief Compute a mask of pixels overlapping a subaperture.
 *
 * For each pixel, determines whether the pixel's coherent processing
 * interval [pixel_start, pixel_end] overlaps with the subaperture
 * [subaperture_start, subaperture_end].
 *
 * @param[in]  subaperture_start Start time of the subaperture (s)
 * @param[in]  subaperture_end   End time of the subaperture (s)
 * @param[in]  n                 Number of pixels
 * @param[in]  pixel_start       Start time of each pixel's CPI (s)
 * @param[in]  pixel_end         End time of each pixel's CPI (s)
 * @param[out] mask              Output mask; true if the pixel
 *                               overlaps the subaperture
 */
void
makeSubApertureMask(
    const double subaperture_start, const double subaperture_end,
    const size_t n,
    const double* pixel_start,
    const double* pixel_end,
    bool* mask);

/**
 * @brief Interpolate a polar grid image to given XYZ positions.
 *
 * Accumulates (adds) contributions from a polar grid image into an
 * output complex signal array at specified 3D positions. For each
 * position, the target location in the polar grid is computed via
 * geo2polar, and the image is interpolated using NFFT. The phase is
 * compensated by the wavenumber-range product kw * range.
 *
 * The output image is accumulated, so it must be zero-initialized
 * by the caller before calling this function.
 *
 * @param[out] image     Output complex signal data (accumulates,
 *                       so caller must init to zero)
 * @param[in]  xyz       Target 3D positions (ECEF, m)
 * @param[in]  n         Number of target positions
 * @param[in]  grid      Polar grid containing the image data
 * @param[in]  nfft      NFFT interpolator for the polar grid
 * @param[in]  kw        Wavenumber (rad/m); equals 4*pi*fc/c
 * @param[in]  mask      Optional mask; pixels with false are skipped
 * @param[in]  dr_atm    Optional atmospheric path delay correction
 *                       (m), added to the range before interpolation
 *
 * @returns Error code indicating success or failure of the interpolation
 */
isce3::error::ErrorCode
accumulatePolarImageToGeoPoints(
        std::complex<float>* image,
        const isce3::core::Vec3* xyz,
        const size_t n,
        const PolarGrid& grid,
        const isce3::signal::NFFT2dResult<float>& nfft,
        const double kw,
        const std::optional<const bool*>& mask = std::nullopt,
        const std::optional<const double*>& dr_atm = std::nullopt);

/**
 * @brief Find the bounding box of a polar grid in radar coordinates.
 *
 * Converts the perimeter of the polar grid to stripmap radar coordinates
 * (azimuth time, slant range) and finds the minimum bounding box.
 * The perimeter is sampled with nextra+1 points along each edge
 * (4 edges), yielding 4*(nextra+1) total perimeter points.
 *
 * @param[in]  polar_grid   Input polar grid
 * @param[in]  orbit        Orbit used to convert XYZ to radar coords
 * @param[in]  doppler      Doppler centroid LUT
 * @param[in]  wavelength   Radar wavelength (m)
 * @param[in]  lookside     Look side (left or right)
 * @param[in]  dem          Digital elevation model (DEM)
 * @param[in]  r2g_params   rdr2geo_bracket configuration parameters
 * @param[in]  g2r_params   geo2rdr_bracket configuration parameters
 * @param[in]  nextra       Number of extra perimeter points per edge
 *                          (defaults to 0)
 *
 * @returns A tuple of:
 *          - tmin: minimum azimuth time (s)
 *          - tmax: maximum azimuth time (s)
 *          - rmin: minimum slant range (m)
 *          - rmax: maximum slant range (m)
 *          - error code (non-zero if any perimeter point fails to
 *            converge in polar2geo or geo2rdr)
 */
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

/**
 * @brief Find the subset of a radar grid covered by a polar grid.
 *
 * Computes the bounding box of the polar grid in stripmap radar
 * coordinates using findPolarGridBoundingBoxInRadarCoord, then
 * converts this bounding box to integer radar grid indices
 * (azimuth line, range sample). If the polar grid does not
 * overlap the radar grid at all, a zero-sized subset (0, 0, 0, 0)
 * is returned.
 *
 * @param[in]  polar_grid    Input polar grid
 * @param[in]  radar_geom    Target radar geometry grid
 * @param[in]  dem           Digital elevation model (DEM)
 * @param[in]  r2g_params    rdr2geo_bracket configuration parameters
 * @param[in]  g2r_params    geo2rdr_bracket configuration parameters
 * @param[in]  nextra        Number of extra perimeter points per
 *                           edge (defaults to 0)
 *
 * @returns A tuple of:
 *          - i0: starting azimuth line index
 *          - i1: ending azimuth line index (exclusive)
 *          - j0: starting range sample index
 *          - j1: ending range sample index (exclusive)
 *          - error code (from findPolarGridBoundingBoxInRadarCoord)
 */
std::tuple<int, int, int, int, isce3::error::ErrorCode>
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
