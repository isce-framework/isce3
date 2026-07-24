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


/**
 * @brief Backproject range-compressed signal into a polar grid (GPU).
 *
 * Performs time-domain backprojection of range-compressed SAR signal
 * data onto a polar coordinate grid. For each output pixel in the polar
 * grid, the signal is resampled from the input data by accumulating
 * contributions from all pulses according to the instantaneous slant
 * range, with phase compensation for motion and the troposphere.
 *
 * @param[in]  in              Input range-compressed signal data
 * @param[in]  in_slant_range  Slant range grid of the input data (m)
 * @param[in]  pos             Platform position vectors at each pulse
 *                             (ECEF, m)
 * @param[in]  vel             Platform velocity vectors at each pulse
 *                             (ECEF, m/s)
 * @param[in]  out_grid        Target polar grid to backproject onto
 * @param[in]  dem             Digital elevation model (DEM)
 * @param[in]  fc              Center frequency (Hz)
 * @param[in]  kernel          1-D interpolation kernel
 * @param[in]  dry_tropo_model Dry troposphere path delay model
 * @param[in]  r2g_params      rdr2geo_bracket configuration parameters
 *
 * @returns A tuple containing:
 *          - error code (non-zero if geometry fails to converge for
 *            any pixel, in which case values for those pixels are NaN)
 *          - focused signal data on the polar grid (size =
 *            out_grid.width() * out_grid.length())
 *          - per-pixel height above the ellipsoid (m)
 */
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

/**
 * @brief Project a polar grid image to geographic points (GPU).
 *
 * Interpolates a polar grid image to a set of 3D geographic (XYZ)
 * positions using NFFT-based interpolation.
 *
 * @param[out] geo_image     Output interpolated signal at geo points
 * @param[in]  geo_points    Target 3D positions (ECEF, m)
 * @param[in]  n             Number of target positions
 * @param[in]  grid          Polar grid containing the image data
 * @param[in]  polar_image   Input polar grid image data
 * @param[in]  wavelength    Radar wavelength (m)
 * @param[in]  params        NFFT interpolation parameters
 *
 * @returns Error code indicating success or failure
 */
isce3::error::ErrorCode
projectPolarToGeo(
        std::complex<float>* geo_image,
        const isce3::core::Vec3* geo_points,
        const size_t n,
        const isce3::focus::PolarGrid& grid,
        const std::complex<float>* polar_image,
        const double wavelength,
        const isce3::signal::NFFT2dParams& params);

/**
 * @brief Accumulate polar grid images onto an output radar grid (GPU).
 *
 * Combines multiple subaperture polar grid images back onto a
 * Cartesian radar geometry grid. For each pixel in the output grid,
 * the target position is computed via rdr2geo, the corresponding
 * coherent processing interval is determined via geo2rdr, and the
 * polar image data is accumulated using NFFT-based interpolation.
 *
 * The caller is responsible for allocating the output arrays to the
 * appropriate size (out_geometry.gridLength() *
 * out_geometry.gridWidth()).
 *
 * @param[out] out                  Accumulated focused signal data
 * @param[in]  out_geometry         Target output grid, orbit, and Doppler
 * @param[in]  in_orbit             Input data orbit
 * @param[in]  in_doppler           Input data Doppler centroid LUT
 * @param[in]  grids                List of subaperture polar grids
 * @param[in]  image_interpolators  NFFT interpolators for each grid
 * @param[in]  dem                  Digital elevation model (DEM)
 * @param[in]  fc                   Center frequency (Hz)
 * @param[in]  ds                   Desired azimuth resolution (m)
 * @param[in]  dry_tropo_model      Dry troposphere path delay model
 * @param[in]  r2g_params           rdr2geo_bracket configuration parameters
 * @param[in]  g2r_params           geo2rdr_bracket configuration parameters
 * @param[out] height               Height of each pixel (m) above the
 *                                  ellipsoid (optional, may be nullptr)
 *
 * @returns Non-zero error code if rdr2geo or geo2rdr fails to
 *          converge for any pixel, and the values for these
 *          pixels are set to NaN.
 */
template <class SequenceType>
isce3::error::ErrorCode
accumulatePolarImagesToRadarGrid(std::complex<float>* out,
        const isce3::container::RadarGeometry& out_geometry,
        const isce3::core::Orbit& in_orbit,
        const isce3::core::LUT2d<double>& in_doppler,
        const std::vector<isce3::focus::PolarGrid>& grids,
        const SequenceType& image_interpolators,
        const isce3::geometry::DEMInterpolator& dem, double fc, double ds,
        DryTroposphereModel dry_tropo_model = DryTroposphereModel::TSX,
        const Rdr2GeoBracketParams& r2g_params = {},
        const Geo2RdrBracketParams& g2r_params = {},
        float* height = nullptr);

/**
 * @brief Merge subaperture polar grid images into a single grid (GPU).
 *
 * Combines multiple subaperture polar grid images onto a merged
 * output polar grid. For each pixel in the output grid, the 3D
 * target position is computed via polar2geo, and the input image
 * data is accumulated via NFFT-based interpolation. The output
 * image is expected to be zero-initialized by the caller.
 *
 * @param[in]  grids                List of subaperture input polar grids
 * @param[in]  image_interpolators  NFFT interpolators for each input grid
 * @param[in]  output_grid          Merged output polar grid
 * @param[out] output_image         Accumulated output image (must be
 *                                  zero-initialized); dimensions must
 *                                  match output_grid
 * @param[in]  fc                   Center frequency (Hz)
 * @param[in]  dem                  Digital elevation model (DEM)
 * @param[in]  r2g_params           rdr2geo_bracket configuration parameters
 * @param[in]  az_block_size        Number of azimuth rows to process
 *                                  at a time (defaults to 1024)
 */
template <class SequenceType>
void mergePolarImages(
    const std::vector<isce3::focus::PolarGrid>& grids,
    const SequenceType& image_interpolators,
    const isce3::focus::PolarGrid& output_grid,
    Eigen::Ref<isce3::core::EArray2D<std::complex<float>>> output_image,
    const double fc,
    const isce3::geometry::DEMInterpolator& dem,
    const isce3::geometry::detail::Rdr2GeoBracketParams& r2g_params = {},
    int az_block_size = 1024);

}}} // namespace isce3::cuda::focus

#define ISCE_CUDA_FOCUS_BACKPROJECT_ICC
#include "Backproject.icc"
#undef ISCE_CUDA_FOCUS_BACKPROJECT_ICC