#pragma once

#include "forward.h"

#include <limits>

#include <isce3/core/forward.h>
#include <isce3/io/forward.h>
#include <isce3/product/forward.h>

#include <isce3/core/Constants.h>
#include <pyre/journal.h>

namespace isce3 { namespace geometry {

/**
 * Enumeration type to indicate input terrain radiometry (for RTC)
 */
enum rtcInputRadiometry {
    BETA_NAUGHT = 0,
    SIGMA_NAUGHT_ELLIPSOID = 1,
};

/** Enumeration type to indicate memory management */
enum rtcMemoryMode {
    RTC_AUTO = 0,
    RTC_SINGLE_BLOCK = 1,
    RTC_BLOCKS_GEOGRID = 2,
};

/**Enumeration type to indicate RTC area mode (AREA or AREA_FACTOR) */
enum rtcAreaMode { AREA = 0, AREA_FACTOR = 1 };

/**Enumeration type to select RTC algorithm (RTC_DAVID_SMALL or
 * RTC_AREA_PROJECTION) */
enum rtcAlgorithm { RTC_DAVID_SMALL = 0, RTC_AREA_PROJECTION = 1 };

/** Apply radiometric terrain correction (RTC) over an input raster
 *
 * @param[in]  radarGrid           Radar Grid
 * @param[in]  orbit               Orbit
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  input_raster        Input raster
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  input_radiometry    Terrain radiometry of the input raster
 * @param[in]  exponent            Exponent to be applied to the input data. The
 * value 0 indicates that the the exponent is based on the data type of the
 *  input raster (1 for real and 2 for complex rasters).
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  rtc_algorithm       RTC algorithm (RTC_DAVID_SMALL or
 * RTC_AREA_PROJECTION)
 * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor. Radar
 * data with RTC area factor below this limit are ignored.
 * @param[in]  abs_cal_factor      Absolute calibration factor.
 * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
 * parameters determines the multilooking factor used to compute out_nlooks.
 * @param[out] out_nlooks          Raster to which the number of radar-grid
 * looks associated with the geogrid will be saved
 * @param[in]  input_rtc           Raster containing pre-computed RTC area
 * factor
 * @param[out] output_rtc          Output RTC area factor
 * @param[in]  rtc_memory_mode     Select memory mode
 * */
void applyRTC(
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        isce3::io::Raster& input_raster, isce3::io::Raster& dem_raster,
        isce3::io::Raster& output_raster,
        rtcInputRadiometry inputRadiometry = rtcInputRadiometry::BETA_NAUGHT,
        int exponent = 0, rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        double abs_cal_factor = 1, float radar_grid_nlooks = 1,
        isce3::io::Raster* out_nlooks = nullptr,
        isce3::io::Raster* input_rtc = nullptr,
        isce3::io::Raster* output_rtc = nullptr,
        rtcMemoryMode rtc_memory_mode = rtcMemoryMode::RTC_AUTO);

/** Generate radiometric terrain correction (RTC) area or area factor
 *
 * @param[in]  product             Product
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  frequency           Product frequency
 * @param[in]  native_doppler      Use native doppler
 * @param[in]  input_radiometry    Terrain radiometry of the input raster
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  rtc_algorithm       RTC algorithm (RTC_DAVID_SMALL or
 * RTC_AREA_PROJECTION)
 * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor. Radar
 * data with RTC area factor below this limit are ignored.
 * @param[in]  nlooks_az           Number of azimuth looks.
 * @param[in]  nlooks_rg           Number of range looks.
 * @param[out] out_nlooks          Raster to which the number of radar-grid
 * @param[in]  rtc_memory_mode     Select memory mode
 * looks associated with the geogrid will be saved
 * */
void facetRTC(
        isce3::product::Product& product, isce3::io::Raster& dem_raster,
        isce3::io::Raster& output_raster, char frequency = 'A',
        bool native_doppler = false,
        rtcInputRadiometry inputRadiometry = rtcInputRadiometry::BETA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        size_t nlooks_az = 1, size_t nlooks_rg = 1,
        isce3::io::Raster* out_nlooks = nullptr,
        rtcMemoryMode rtc_memory_mode = rtcMemoryMode::RTC_AUTO);

/** Generate radiometric terrain correction (RTC) area or area factor
 *
 * @param[in]  radarGrid           Radar Grid
 * @param[in]  orbit               Orbit
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  frequency           Product frequency
 * @param[in]  input_radiometry    Terrain radiometry of the input raster
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  rtc_algorithm       RTC algorithm (RTC_DAVID_SMALL or
 * RTC_AREA_PROJECTION)
 * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor. Radar
 * data with RTC area factor below this limit are ignored.
 * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
 * parameters determines the multilooking factor used to compute out_nlooks.
 * @param[out] out_nlooks          Raster to which the number of radar-grid
 * looks associated with the geogrid will be saved
 * @param[in] rtc_memory_mode     Select memory mode
 * @param[in] interp_method        Interpolation Method
 * @param[in] threshold            Distance threshold for convergence
 * @param[in] num_iter             Maximum number of Newton-Raphson iterations
 * @param[in] delta_range          Step size used for computing derivative of
 * doppler
 * */
void facetRTC(
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        isce3::io::Raster& dem, isce3::io::Raster& output_raster,
        rtcInputRadiometry inputRadiometry = rtcInputRadiometry::BETA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        float radar_grid_nlooks = 1,
        isce3::io::Raster* out_nlooks = nullptr,
        rtcMemoryMode rtc_memory_mode = rtcMemoryMode::RTC_AUTO,
        isce3::core::dataInterpMethod interp_method =
                isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
        double threshold = 1e-4, int num_iter = 100, double delta_range = 1e-4);

/** Generate radiometric terrain correction (RTC) area or area factor
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
 * @param[in]  input_radiometry    Terrain radiometry of the input raster
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  rtc_algorithm       RTC algorithm (RTC_DAVID_SMALL or
 * RTC_AREA_PROJECTION)
 * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor. Radar
 * data with RTC area factor below this limit are ignored.
 * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
 * parameters determines the multilooking factor used to compute out_nlooks.
 * @param[out] out_geo_vertices    Raster to which the radar-grid positions
 * (range and azimuth) of the geogrid pixels vertices will be saved.
 * @param[out] out_geo_grid        Raster to which the radar-grid positions
 * (range and azimuth) of the geogrid pixels center will be saved.
 * @param[out] out_nlooks          Raster to which the number of radar-grid
 * looks associated with the geogrid will be saved
 * @param[in] rtc_memory_mode      Select memory mode
 * @param[in] interp_method        Interpolation Method
 * @param[in] threshold            Distance threshold for convergence
 * @param[in] num_iter             Maximum number of Newton-Raphson iterations
 * @param[in] delta_range          Step size used for computing derivative of
 * doppler
 * */
void facetRTC(
        isce3::io::Raster& dem_raster, isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        const double y0, const double dy, const double x0, const double dx,
        const int geogrid_length, const int geogrid_width,
        const int epsg,
        rtcInputRadiometry inputRadiometry = rtcInputRadiometry::BETA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        float radar_grid_nlooks = 1,
        isce3::io::Raster* out_geo_vertices = nullptr,
        isce3::io::Raster* out_geo_grid = nullptr,
        isce3::io::Raster* out_nlooks = nullptr,
        rtcMemoryMode rtc_memory_mode = rtcMemoryMode::RTC_AUTO,
        isce3::core::dataInterpMethod interp_method =
                isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
        double threshold = 1e-4, int num_iter = 100, double delta_range = 1e-4);

/** Generate radiometric terrain correction (RTC) area or area factor using the
 * David Small algorithm
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
 * @param[in]  input_radiometry    Terrain radiometry of the input raster
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
 * */
void facetRTCDavidSmall(
        isce3::io::Raster& dem_raster, isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        const double y0, const double dy, const double x0, const double dx,
        const int geogrid_length, const int geogrid_width,
        const int epsg,
        rtcInputRadiometry inputRadiometry = rtcInputRadiometry::BETA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN());

/** Generate radiometric terrain correction (RTC) area or area factor using the
 * area projection algorithms
 *
 * @param[in]  dem_raster          Input DEM raster
 * @param[out] output_raster       Output raster
 * @param[in]  radarGrid           Radar Grid
 * @param[in]  orbit               Orbit
 * @param[in]  input_dop           Doppler LUT
 * @param[in]  y0                  Starting easting position
 * @param[in]  dy
 * @param[in]  y0                  Starting northing position
 * @param[in]  dy                  Northing step size
 * @param[in]  x0                  Starting easting position
 * @param[in]  dx                  Easting step size
 * @param[in]  geogrid_length      Geographic length (number of pixels) in
 * the Northing direction
 * @param[in]  geogrid_width       Geographic width (number of pixels) in
 * the Easting direction
 * @param[in]  epsg                Output geographic grid EPSG
 * @param[in]  input_radiometry    Terrain radiometry of the input raster
 * @param[in]  rtc_area_mode       RTC area mode (AREA or AREA_FACTOR)
 * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
 * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor. Radar
 * data with RTC area factor below this limit are ignored.
 * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
 * parameters determines the multilooking factor used to compute out_nlooks.
 * @param[out] out_geo_vertices       Raster to which the radar-grid positions
 * (range and azimuth) of the geogrid pixels vertices will be saved.
 * @param[out] out_geo_grid        Raster to which the radar-grid positions
 * (range and azimuth) of the geogrid pixels center will be saved.
 * @param[out] out_nlooks          Raster to which the number of radar-grid
 * looks associated with the geogrid will be saved
 * @param[in] rtc_memory_mode      Select memory mode
 * @param[in] interp_method        Interpolation Method
 * @param[in] threshold            Distance threshold for convergence
 * @param[in] num_iter             Maximum number of Newton-Raphson iterations
 * @param[in] delta_range          Step size used for computing derivative of
 * doppler
 * */
void facetRTCAreaProj(
        isce3::io::Raster& dem, isce3::io::Raster& output_raster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::LUT2d<double>& dop,
        const double y0, const double dy, const double x0, const double dx,
        const int geogrid_length, const int geogrid_width,
        const int epsg,
        rtcInputRadiometry inputRadiometry = rtcInputRadiometry::BETA_NAUGHT,
        rtcAreaMode rtc_area_mode = rtcAreaMode::AREA_FACTOR,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        float radar_grid_nlooks = 1,
        isce3::io::Raster* out_geo_vertices = nullptr,
        isce3::io::Raster* out_geo_grid = nullptr,
        isce3::io::Raster* out_nlooks = nullptr,
        rtcMemoryMode rtc_memory_mode = rtcMemoryMode::RTC_AUTO,
        isce3::core::dataInterpMethod interp_method =
                isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
        double threshold = 1e-4, int num_iter = 100, double delta_range = 1e-4);

void areaProjIntegrateSegment(double y1, double y2, double x1, double x2,
                              int length, int width,
                              isce3::core::Matrix<double>& w_arr,
                              double& w_total, int plane_orientation);

int areaProjGetNBlocks(int array_length,
                       pyre::journal::info_t* channel = nullptr,
                       int upsampling = 0,
                       int* block_length_with_upsampling = nullptr, 
                       int* block_length = nullptr,
                       int min_block_length = std::pow(2, 8),  // 256
                       int max_block_length = std::pow(2, 10)); // 1024

double
computeUpsamplingFactor(const DEMInterpolator& dem_interp,
                        const isce3::product::RadarGridParameters& radar_grid,
                        const isce3::core::Ellipsoid& ellps);

double computeFacet(isce3::core::Vec3 xyz_center, isce3::core::Vec3 xyz_left,
                    isce3::core::Vec3 xyz_right, isce3::core::Vec3 lookXYZ,
                    double p1, double& p3, double divisor,
                    bool clockwise_direction);

std::string get_input_radiometry_str(rtcInputRadiometry input_radiometry);
std::string get_rtc_area_mode_str(rtcAreaMode rtc_area_mode);
std::string get_rtc_algorithm_str(rtcAlgorithm rtc_algorithm);

void print_parameters(
        pyre::journal::info_t& channel,
        const isce3::product::RadarGridParameters& radar_grid, const double y0,
        const double dy, const double x0, const double dx,
        const int geogrid_length, const int geogrid_width,
        rtcInputRadiometry input_radiometry, rtcAreaMode rtc_area_mode,
        double geogrid_upsampling,
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN());
} // namespace geometry
} // namespace isce3
