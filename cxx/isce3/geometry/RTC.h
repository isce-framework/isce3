#pragma once

#include "forward.h"
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>
#include <isce3/product/forward.h>

#include <limits>

#include <pyre/journal.h>

#include <isce3/core/Constants.h>
#include <isce3/core/blockProcessing.h>
#include <isce3/error/ErrorCode.h>

namespace isce3 { namespace geometry {

/**
 * Enumeration type to indicate input terrain radiometry (for RTC)
 */
enum rtcInputTerrainRadiometry {
    BETA_NAUGHT = 0,
    SIGMA_NAUGHT_ELLIPSOID = 1,
};

/**
 * Enumeration type to indicate output terrain radiometry (for RTC)
 */
enum rtcOutputTerrainRadiometry {
    SIGMA_NAUGHT = 0,
    GAMMA_NAUGHT = 1,
};

/**Enumeration type to indicate RTC area mode (AREA or AREA_FACTOR) */
enum rtcAreaMode { AREA = 0, AREA_FACTOR = 1 };

/**Enumeration type to select RTC algorithm (RTC_BILINEAR_DISTRIBUTION or
 * RTC_AREA_PROJECTION) */
enum rtcAlgorithm { RTC_BILINEAR_DISTRIBUTION = 0, RTC_AREA_PROJECTION = 1 };

/**Enumeration type to indicate RTC area beta mode
 * (option only available for rtcAlgorithm.RTC_AREA_PROJECTION)
 */
enum rtcAreaBetaMode {
        AUTO = 0,  /**< auto mode. Default value is defined by the
                        RTC algorithm that is being executed, i.e.,
                        PIXEL_AREA for rtcAlgorithm::RTC_BILINEAR_DISTRIBUTION
                        and PROJECTION_ANGLE for
                        rtcAlgorithm::RTC_AREA_PROJECTION */
        PIXEL_AREA = 1,  /**< estimate the beta surface reference area `A_beta`
                              using the pixel area, which is the
                              product of the range spacing by the
                              azimuth spacing (computed using the ground velocity) */
        PROJECTION_ANGLE = 2 /**< estimate the beta surface reference area `A_beta`
                                  using the projection angle method:
                                  `A_beta = A_sigma * cos(projection_angle)` */
};

/** Apply radiometric terrain correction (RTC) over an input raster
 *
 * @param[in]  radarGrid           Radar Grid
 * @param[in]  orbit               Orbit
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  input_raster        Input raster
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  input_terrain_radiometry  Input terrain radiometry
 * @param[in]  output_terrain_radiometry Output terrain radiometry
 * @param[in]  exponent            Exponent to be applied to the input data. The
 * value 0 indicates that the the exponent is based on the data type of the
 *  input raster (1 for real and 2 for complex rasters).
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  rtc_algorithm       RTC algorithm (RTC_BILINEAR_DISTRIBUTION or
 * RTC_AREA_PROJECTION)
 * @param[in]  rtc_area_beta_mode RTC area beta mode (AUTO, PIXEL_AREA,
 * PROJECTION_ANGLE)
 * @param[in]  geogrid_upsampling  Geogrid upsampling
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area normalization
 * factor. Radar data with RTC area normalization factor below this limit will
 * be set to NaN.
 * @param[in]  abs_cal_factor      Absolute calibration factor.
 * @param[in]  clip_min            Clip minimum output values
 * @param[in]  clip_max            Clip maximum output values
 * @param[out] out_sigma           Output sigma surface area
 * (rtc_area_mode = AREA) or area factor (rtc_area_mode = AREA_FACTOR) raster
 * @param[in]  input_rtc           Raster containing pre-computed RTC area
 * factor
 * @param[out] output_rtc          Output RTC area normalization factor
 * @param[in]  rtc_memory_mode     Select memory mode
 * */
void applyRtc(const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        isce3::io::Raster& input_raster, isce3::io::Raster& dem_raster,
        isce3::io::Raster& output_raster,
        rtcInputTerrainRadiometry input_terrain_radiometry =
                rtcInputTerrainRadiometry::BETA_NAUGHT,
        rtcOutputTerrainRadiometry output_terrain_radiometry =
                rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
        int exponent = 0, rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
        rtcAreaBetaMode rtc_area_beta_mode = rtcAreaBetaMode::AUTO,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        double abs_cal_factor = 1,
        float clip_min = std::numeric_limits<float>::quiet_NaN(),
        float clip_max = std::numeric_limits<float>::quiet_NaN(),
        isce3::io::Raster* out_sigma = nullptr,
        isce3::io::Raster* input_rtc = nullptr,
        isce3::io::Raster* output_rtc = nullptr,
        isce3::core::MemoryModeBlocksY rtc_memory_mode = 
                isce3::core::MemoryModeBlocksY::AutoBlocksY);


/** Generate radiometric terrain correction (RTC) area or area normalization
 * factor
 *
 * @param[in]  radarGrid           Radar Grid
 * @param[in]  orbit               Orbit
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  input_terrain_radiometry  Input terrain radiometry
 * @param[in]  output_terrain_radiometry Output terrain radiometry
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  rtc_algorithm       RTC algorithm (RTC_BILINEAR_DISTRIBUTION or
 * RTC_AREA_PROJECTION)
 * @param[in]  rtc_area_beta_mode RTC area beta mode (AUTO, PIXEL_AREA,
 * PROJECTION_ANGLE)
 * @param[in]  geogrid_upsampling  Geogrid upsampling
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area normalization
 * factor. Radar data with RTC area normalization factor below this limit will
 * be set to NaN..
 * @param[out] out_sigma           Output sigma surface area
 * (rtc_area_mode = AREA) or area factor (rtc_area_mode = AREA_FACTOR) raster
 * @param[in]  rtc_memory_mode     Select memory mode
 * @param[in]  interp_method       Interpolation Method
 * @param[in]  threshold           Azimuth time threshold for convergence (s)
 * @param[in]  num_iter            Maximum number of Newton-Raphson iterations
 * @param[in]  delta_range         Step size used for computing derivative of
 * doppler
 * @param[in]  min_block_size       Minimum block size (per thread)
 * @param[in]  max_block_size       Maximum block size (per thread)
 * */
void computeRtc(const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        isce3::io::Raster& dem, isce3::io::Raster& output_raster,
        rtcInputTerrainRadiometry inputTerrainRadiometry =
                rtcInputTerrainRadiometry::BETA_NAUGHT,
        rtcOutputTerrainRadiometry outputTerrainRadiometry =
                rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
        rtcAreaBetaMode rtc_area_beta_mode = rtcAreaBetaMode::AUTO,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        isce3::io::Raster* out_sigma = nullptr,
        isce3::core::MemoryModeBlocksY rtc_memory_mode = 
                isce3::core::MemoryModeBlocksY::AutoBlocksY,
        isce3::core::dataInterpMethod interp_method =
                isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
        double threshold = 1e-8, int num_iter = 100, double delta_range = 1e-8,
        const long long min_block_size = isce3::core::DEFAULT_MIN_BLOCK_SIZE,
        const long long max_block_size = isce3::core::DEFAULT_MAX_BLOCK_SIZE);

/** Generate radiometric terrain correction (RTC) area or area normalization
 * factor
 *
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  radarGrid           Radar Grid
 * @param[in]  orbit               Orbit
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  y0                  Starting northing position
 * @param[in]  dy                  Northing step size
 * @param[in]  x0                  Starting easting position
 * @param[in]  dx                  Easting step size
 * @param[in]  geogrid_length      Geographic length (number of pixels) in
 * the Northing direction
 * @param[in]  geogrid_width       Geographic width (number of pixels) in
 * the Easting direction
 * @param[in]  epsg                Output geographic grid EPSG
 * @param[in]  input_terrain_radiometry  Input terrain radiometry
 * @param[in]  output_terrain_radiometry Output terrain radiometry
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  rtc_algorithm       RTC algorithm (RTC_BILINEAR_DISTRIBUTION or
 * RTC_AREA_PROJECTION)
 * @param[in]  rtc_area_beta_mode RTC area beta mode (AUTO, PIXEL_AREA,
 * PROJECTION_ANGLE)
 * @param[in]  geogrid_upsampling  Geogrid upsampling
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area normalization
 * factor. Radar data with RTC area normalization factor below this limit will
 * be set to NaN..
 * @param[out] out_geo_rdr    Raster to which the radar-grid positions
 * (range and azimuth) of the geogrid pixels vertices will be saved.
 * @param[out] out_geo_grid        Raster to which the radar-grid positions
 * (range and azimuth) of the geogrid pixels center will be saved.
 * @param[out] out_sigma           Output sigma surface area
 * (rtc_area_mode = AREA) or area factor (rtc_area_mode = AREA_FACTOR) raster
 * @param[in]  rtc_memory_mode     Select memory mode
 * @param[in]  interp_method       Interpolation Method
 * @param[in]  threshold           Azimuth time threshold for convergence (s)
 * @param[in]  num_iter            Maximum number of Newton-Raphson iterations
 * @param[in]  delta_range         Step size used for computing derivative of
 * doppler
 * @param[in]  min_block_size       Minimum block size (per thread)
 * @param[in]  max_block_size       Maximum block size (per thread)
 * */
void computeRtc(isce3::io::Raster& dem_raster, isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        const double y0, const double dy, const double x0, const double dx,
        const int geogrid_length, const int geogrid_width, const int epsg,
        rtcInputTerrainRadiometry inputTerrainRadiometry =
                rtcInputTerrainRadiometry::BETA_NAUGHT,
        rtcOutputTerrainRadiometry outputTerrainRadiometry =
                rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
        rtcAreaBetaMode rtc_area_beta_mode = rtcAreaBetaMode::AUTO,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        isce3::io::Raster* out_geo_rdr = nullptr,
        isce3::io::Raster* out_geo_grid = nullptr,
        isce3::io::Raster* out_sigma = nullptr,
        isce3::core::MemoryModeBlocksY rtc_memory_mode = 
                isce3::core::MemoryModeBlocksY::AutoBlocksY,
        isce3::core::dataInterpMethod interp_method =
                isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
        double threshold = 1e-8, int num_iter = 100, double delta_range = 1e-8,
        const long long min_block_size = isce3::core::DEFAULT_MIN_BLOCK_SIZE,
        const long long max_block_size = isce3::core::DEFAULT_MAX_BLOCK_SIZE);

/** Generate radiometric terrain correction (RTC) area or area normalization
 * factor using the Bilinear Distribution (D. Small) algorithm @cite small2011.
 *
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  radarGrid           Radar Grid
 * @param[in]  orbit               Orbit
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  geogrid             Geogrid parameters
 * @param[in]  epsg                Output geographic grid EPSG
 * @param[in]  input_terrain_radiometry  Input terrain radiometry
 * @param[in]  output_terrain_radiometry Output terrain radiometry
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  geogrid_upsampling  Geogrid upsampling
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area normalization
 * factor. Radar data with RTC area normalization factor below this limit will
 * be set to NaN.
 * @param[out] out_sigma           Output sigma surface area
 * (rtc_area_mode = AREA) or area factor (rtc_area_mode = AREA_FACTOR) raster
 * */
void computeRtcBilinearDistribution(isce3::io::Raster& dem_raster,
        isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        const isce3::product::GeoGridParameters& geogrid,
        rtcInputTerrainRadiometry input_terrain_radiometry =
                rtcInputTerrainRadiometry::BETA_NAUGHT,
        rtcOutputTerrainRadiometry output_terrain_radiometry =
                rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        isce3::io::Raster* out_sigma = nullptr);

/** Generate radiometric terrain correction (RTC) area or area normalization
 * factor using the area projection algorithms
 *
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  radarGrid           Radar Grid
 * @param[in]  orbit               Orbit
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  y0                  Starting easting position
 * @param[in]  dy
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  geogrid             Geogrid parameters
 * @param[in]  input_terrain_radiometry  Input terrain radiometry
 * @param[in]  output_terrain_radiometry Output terrain radiometry
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  rtc_area_beta_mode RTC area beta mode (AUTO, PIXEL_AREA,
 * PROJECTION_ANGLE)
 * @param[in]  geogrid_upsampling  Geogrid upsampling
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area normalization
 * factor. Radar data with RTC area normalization factor below this limit will
 * be set to NaN..
 * @param[out] out_geo_rdr       Raster to which the radar-grid positions
 * (range and azimuth) of the geogrid pixels vertices will be saved.
 * @param[out] out_geo_grid        Raster to which the radar-grid positions
 * (range and azimuth) of the geogrid pixels center will be saved.
 * @param[out] out_sigma           Output sigma surface area
 * (rtc_area_mode = AREA) or area factor (rtc_area_mode = AREA_FACTOR) raster
 * @param[in] rtc_memory_mode      Select memory mode
 * @param[in] interp_method        Interpolation Method
 * @param[in] threshold            Azimuth time threshold for convergence (s)
 * @param[in] num_iter             Maximum number of Newton-Raphson iterations
 * @param[in] delta_range          Step size used for computing derivative of
 * doppler
 * @param[in]  min_block_size       Minimum block size (per thread)
 * @param[in]  max_block_size       Maximum block size (per thread)
 * */
void computeRtcAreaProj(isce3::io::Raster& dem,
        isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        const isce3::product::GeoGridParameters& geogrid,
        rtcInputTerrainRadiometry input_terrain_radiometry =
                rtcInputTerrainRadiometry::BETA_NAUGHT,
        rtcOutputTerrainRadiometry output_terrain_radiometry =
                rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        rtcAreaBetaMode rtc_area_beta_mode = rtcAreaBetaMode::AUTO,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        isce3::io::Raster* out_geo_rdr = nullptr,
        isce3::io::Raster* out_geo_grid = nullptr,
        isce3::io::Raster* out_sigma = nullptr,
        isce3::core::MemoryModeBlocksY rtc_memory_mode = 
                isce3::core::MemoryModeBlocksY::AutoBlocksY,
        isce3::core::dataInterpMethod interp_method =
                isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
        double threshold = 1e-8, int num_iter = 100, double delta_range = 1e-8,
        const long long min_block_size = isce3::core::DEFAULT_MIN_BLOCK_SIZE,
        const long long max_block_size = isce3::core::DEFAULT_MAX_BLOCK_SIZE);

void areaProjIntegrateSegment(double y1, double y2, double x1, double x2,
        int length, int width, isce3::core::Matrix<double>& w_arr,
        double& w_total, int plane_orientation);

double
computeUpsamplingFactor(const DEMInterpolator& dem_interp,
                        const isce3::product::RadarGridParameters& radar_grid,
                        const isce3::core::Ellipsoid& ellps);

std::string get_input_terrain_radiometry_str(
        rtcInputTerrainRadiometry input_terrain_radiometry);
std::string get_output_terrain_radiometry_str(
        rtcOutputTerrainRadiometry output_terrain_radiometry);
std::string get_rtc_area_mode_str(rtcAreaMode rtc_area_mode);
std::string get_rtc_area_beta_mode_str(rtcAreaBetaMode rtc_area_beta_mode);
std::string get_rtc_algorithm_str(rtcAlgorithm rtc_algorithm);

void print_parameters(pyre::journal::info_t& channel,
        const isce3::product::RadarGridParameters& radar_grid,
        rtcInputTerrainRadiometry input_terrain_radiometry,
        rtcOutputTerrainRadiometry output_terrain_radiometry,
        rtcAreaMode rtc_area_mode,
        rtcAreaBetaMode rtc_area_beta_mode,
        double geogrid_upsampling,
        float rtc_min_value_db);
}} // namespace isce3::geometry
