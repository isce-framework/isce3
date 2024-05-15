#pragma once

#include <functional>
#include <optional>

// pyre
#include <pyre/journal.h>

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/product/SubSwaths.h>
#include <isce3/core/blockProcessing.h>

// isce3::io
#include <isce3/io/Raster.h>

// isce3::product
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/RadarGridParameters.h>

// isce3::geometry
#include <isce3/geometry/RTC.h>

namespace isce3 { namespace geocode {

/** Enumeration type to indicate the algorithm used for geocoding */
enum geocodeOutputMode {
    INTERP = 0,
    AREA_PROJECTION = 1,
};


template<class T>
class Geocode {
public:
    /** Geocode data from slant-range to map coordinates
     *
     * @param[in]  radar_grid          Radar grid
     * @param[in]  input_raster        Input raster
     * @param[out] output_raster       Output raster
     * @param[in]  dem_raster          Input DEM raster
     * @param[in]  output_mode         Geocode method
     * @param[in]  flag_az_baseband_doppler Shift SLC azimuth spectrum to
     * baseband (using Doppler centroid) before interpolation
     * @param[in]  flatten             Flatten the geocoded SLC
     * @param[in]  geogrid_upsampling  Geogrid upsampling
     * @param[in]  flag_upsample_radar_grid Double the radar grid sampling rate
     * @param[in]  flag_apply_rtc      Apply radiometric terrain correction
     * (RTC)
     * @param[in]  input_terrain_radiometry  Input terrain radiometry
     * @param[in]  output_terrain_radiometry Output terrain radiometry
     * @param[in]  exponent            Exponent to be applied to the input data.
     * The value 0 indicates that the the exponent is based on the data type of
     * the input raster (1 for real and 2 for complex rasters).
     * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor.
     * Radar data with RTC area factor below this limit will be set to NaN.
     * @param[in]  rtc_geogrid_upsampling  Geogrid upsampling to compute
     * the radiometric terrain correction RTC.
     * @param[in]  rtc_algorithm       RTC algorithm
     * @param[in]  rtc_area_beta_mode RTC area beta mode (AUTO, PIXEL_AREA,
     * PROJECTION_ANGLE)
     * @param[in]  abs_cal_factor      Absolute calibration factor.
     * @param[in]  clip_min            Clip (limit) minimum output values
     * @param[in]  clip_max            Clip (limit) maximum output values
     * @param[in]  min_nlooks          Minimum number of looks. Geogrid data
     * below this limit will be set to NaN
     * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
     * parameters determines the multilooking factor used to compute
     * out_geo_nlooks.
     * @param[out] out_off_diag_terms  Output raster containing the
     * off-diagonal terms of the covariance matrix.
     * @param[out] out_geo_rdr         Raster to which the radar-grid
     * positions (range and azimuth) of the geogrid pixels vertices will be
     * saved.
     * @param[out] out_geo_dem         Raster to which the interpolated DEM
     * will be saved.
     * @param[out] out_nlooks          Raster to which the number of radar-grid
     * looks associated with the geogrid will be saved.
     * @param[out] out_geo_rtc         Output RTC area factor (in
     * geo-coordinates).
     * @param[out] out_geo_rtc_gamma0_to_sigma0 Output RTC area factor 
     * gamma0 to sigma0 (in geo-coordinates).
     * @param[in]  phase_screen_raster Phase screen to be removed before
     * geocoding
     * \param[in]  az_time_correction     geo2rdr azimuth additive correction, in
     * seconds, as a function of azimuth and range
     * \param[in]  slant_range_correction  geo2rdr slant range additive correction,
     * in meters, as a function of azimuth and range
     * @param[in]  input_rtc           Input RTC area factor (in slant-range geometry).
     * @param[out] output_rtc          Output RTC area factor (in slant-range geometry).
     * @param[in]  input_layover_shadow_mask_raster Input layover/shadow mask raster
     * (in radar geometry). Samples identified as SHADOW or LAYOVER_AND_SHADOW are
     * considered invalid.
     * @param[in]  sub_swaths          Sub-swaths metadata
     * @param[in]  apply_valid_samples_sub_swath_masking Flag indicating whether the
     * valid-samples sub-swath masking should be applied during geocoding.
     * If not given, then sub-swath masking will be applied if the sub_swaths
     * parameter is provided.
     * @param[out] out_mask            Output valid-pixels
     * sub-swath mask raster
     * @param[in]  geocode_memory_mode Select memory mode
     * @param[in]  min_block_size      Minimum block size (per thread)
     * @param[in]  max_block_size      Maximum block size (per thread)
     * @param[in]  dem_interp_method   DEM interpolation method
     */
    void geocode(const isce3::product::RadarGridParameters& radar_grid,
            isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
            isce3::io::Raster& dem_raster,
            geocodeOutputMode output_mode = geocodeOutputMode::INTERP,
            bool flag_az_baseband_doppler = false, bool flatten = false,
            double geogrid_upsampling = 1,
            bool flag_upsample_radar_grid = false, bool flag_apply_rtc = false,
            isce3::geometry::rtcInputTerrainRadiometry
                    input_terrain_radiometry = isce3::geometry::
                            rtcInputTerrainRadiometry::BETA_NAUGHT,
            isce3::geometry::rtcOutputTerrainRadiometry
                    output_terrain_radiometry = isce3::geometry::
                            rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
            int exponent = 0,
            float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
            double rtc_geogrid_upsampling =
                    std::numeric_limits<double>::quiet_NaN(),
            isce3::geometry::rtcAlgorithm rtc_algorithm =
                    isce3::geometry::rtcAlgorithm::RTC_AREA_PROJECTION,
            isce3::geometry::rtcAreaBetaMode rtc_area_beta_mode =
                isce3::geometry::rtcAreaBetaMode::AUTO,
            double abs_cal_factor = 1,
            float clip_min = std::numeric_limits<float>::quiet_NaN(),
            float clip_max = std::numeric_limits<float>::quiet_NaN(),
            float min_nlooks = std::numeric_limits<float>::quiet_NaN(),
            float radar_grid_nlooks = 1,
            isce3::io::Raster* out_off_diag_terms = nullptr,
            isce3::io::Raster* out_geo_rdr = nullptr,
            isce3::io::Raster* out_geo_dem = nullptr,
            isce3::io::Raster* out_geo_nlooks = nullptr,
            isce3::io::Raster* out_geo_rtc = nullptr,
            isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0 = nullptr,
            isce3::io::Raster* phase_screen_raster = nullptr,
            const isce3::core::LUT2d<double>& az_time_correction = {},
            const isce3::core::LUT2d<double>& slant_range_correction = {},
            isce3::io::Raster* input_rtc = nullptr,
            isce3::io::Raster* output_rtc = nullptr,
            isce3::io::Raster* input_layover_shadow_mask_raster = nullptr,
            isce3::product::SubSwaths* sub_swaths = nullptr,
            std::optional<bool> apply_valid_samples_sub_swath_masking = std::nullopt,
            isce3::io::Raster* out_mask = nullptr,
            isce3::core::GeocodeMemoryMode geocode_memory_mode =
                    isce3::core::GeocodeMemoryMode::Auto,
            const long long min_block_size =
                    isce3::core::DEFAULT_MIN_BLOCK_SIZE,
            const long long max_block_size =
                    isce3::core::DEFAULT_MAX_BLOCK_SIZE,
            isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD);

    /** Geocode using the interpolation algorithm.
     *
     * @param[in]  radar_grid          Radar grid
     * @param[in]  input_raster        Input raster
     * @param[out] output_raster       Output raster
     * @param[in]  dem_raster          Input DEM raster
     * @param[in]  flag_az_baseband_doppler Shift SLC azimuth spectrum to
     * baseband (using Doppler centroid) before interpolation
     * @param[in]  input_terrain_radiometry  Input terrain radiometry
     * @param[in]  output_terrain_radiometry Output terrain radiometry
     * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor.
     * Radar data with RTC area factor below this limit will be set to NaN.
     * @param[in]  rtc_geogrid_upsampling  Geogrid upsampling to compute the
     * radiometric terrain correction RTC.
     * @param[in]  rtc_algorithm       RTC algorithm
     * @param[in]  rtc_area_beta_mode RTC area beta mode (AUTO, PIXEL_AREA,
     * PROJECTION_ANGLE)
     * @param[in]  abs_cal_factor      Absolute calibration factor.
     * @param[in]  clip_min            Clip (limit) minimum output values
     * @param[in]  clip_max            Clip (limit) maximum output values
     * @param[out] out_geo_rdr         Raster to which the radar-grid
     * positions (range and azimuth) of the geogrid pixels centers will be
     * saved.
     * @param[out] out_geo_dem         Raster to which the interpolated DEM
     * will be saved.
     * @param[out] out_geo_rtc         Output RTC area factor (in
     * geo-coordinates).
     * @param[out] out_geo_rtc_gamma0_to_sigma0 Output RTC area factor 
     * gamma0 to sigma0 (in geo-coordinates).
     * @param[in]  flatten             Flatten the geocoded SLC
     * @param[in]  phase_screen_raster Phase screen to be removed before
     * geocoding
     * \param[in]  az_time_correction     geo2rdr azimuth additive correction, in
     * seconds, as a function of azimuth and range
     * \param[in]  slant_range_correction  geo2rdr slant range additive correction,
     * in meters, as a function of azimuth and range
     * @param[in]  input_rtc           Input RTC area factor (in slant-range geometry).
     * @param[out] output_rtc          Output RTC area factor (in slant-range geometry).
     * @param[in]  input_layover_shadow_mask_raster Input layover/shadow mask raster
     * (in radar geometry). Samples identified as SHADOW or LAYOVER_AND_SHADOW are
     * considered invalid.
     * @param[in]  sub_swaths          Sub-swaths metadata
     * @param[out] out_mask            Output valid-pixels
     * sub-swath mask raster
     * @param[in]  geocode_memory_mode Select memory mode
     * @param[in]  min_block_size      Minimum block size (per thread)
     * @param[in]  max_block_size      Maximum block size (per thread)
     * @param[in]  dem_interp_method   DEM interpolation method
     */
    template<class T_out>
    void geocodeInterp(const isce3::product::RadarGridParameters& radar_grid,
            isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
            isce3::io::Raster& dem_raster, bool flag_apply_rtc = false,
            bool flag_az_baseband_doppler = false, bool flatten = false,
            isce3::geometry::rtcInputTerrainRadiometry
                    input_terrain_radiometry = isce3::geometry::
                            rtcInputTerrainRadiometry::BETA_NAUGHT,
            isce3::geometry::rtcOutputTerrainRadiometry
                    output_terrain_radiometry = isce3::geometry::
                            rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
            float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
            double rtc_geogrid_upsampling =
                    std::numeric_limits<double>::quiet_NaN(),
            isce3::geometry::rtcAlgorithm rtc_algorithm =
                    isce3::geometry::rtcAlgorithm::RTC_AREA_PROJECTION,
            isce3::geometry::rtcAreaBetaMode rtc_area_beta_mode =
                isce3::geometry::rtcAreaBetaMode::AUTO,
            double abs_cal_factor = 1,
            float clip_min = std::numeric_limits<float>::quiet_NaN(),
            float clip_max = std::numeric_limits<float>::quiet_NaN(),
            isce3::io::Raster* out_geo_rdr = nullptr,
            isce3::io::Raster* out_geo_dem = nullptr,
            isce3::io::Raster* out_geo_rtc = nullptr,
            isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0 = nullptr,
            isce3::io::Raster* phase_screen_raster = nullptr,
            const isce3::core::LUT2d<double>& az_time_correction = {},
            const isce3::core::LUT2d<double>& slant_range_correction = {},
            isce3::io::Raster* input_rtc = nullptr,
            isce3::io::Raster* output_rtc = nullptr,
            isce3::io::Raster* input_layover_shadow_mask_raster = nullptr,
            isce3::product::SubSwaths* sub_swaths = nullptr,
            isce3::io::Raster* out_mask = nullptr,
            isce3::core::GeocodeMemoryMode geocode_memory_mode =
                isce3::core::GeocodeMemoryMode::Auto,
            const long long min_block_size =
                    isce3::core::DEFAULT_MIN_BLOCK_SIZE,
            const long long max_block_size =
                    isce3::core::DEFAULT_MAX_BLOCK_SIZE,
            isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD);

    /** Geocode using the area projection algorithm (adaptive multilooking)
     *
     * @param[in]  radar_grid          Radar grid
     * @param[in]  input_raster        Input raster
     * @param[out] output_raster       Output raster
     * @param[in]  dem_raster          Input DEM raster
     * @param[in]  geogrid_upsampling  Geogrid upsampling
     * @param[in]  flag_upsample_radar_grid Double the radar grid sampling rate
     * @param[in]  flag_apply_rtc      Apply radiometric terrain correction
     * (RTC)
     * @param[in]  input_terrain_radiometry  Input terrain radiometry
     * @param[in]  output_terrain_radiometry Output terrain radiometry
     * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor.
     * Radar data with RTC area factor below this limit will be set to NaN.
     * @param[in]  rtc_geogrid_upsampling  Geogrid upsampling to compute the
     * radiometric terrain correction RTC.
     * @param[in]  rtc_algorithm       RTC algorithm
     * @param[in]  rtc_area_beta_mode RTC area beta mode (AUTO, PIXEL_AREA,
     * PROJECTION_ANGLE)
     * @param[in]  abs_cal_factor      Absolute calibration factor.
     * @param[in]  clip_min            Clip (limit) minimum output values
     * @param[in]  clip_max            Clip (limit) maximum output values
     * @param[in]  min_nlooks          Minimum number of looks. Geogrid data
     * below this limit will be set to NaN
     * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
     * parameters determines the multilooking factor used to compute out_nlooks.
     * @param[out] out_off_diag_terms  Output raster containing the
     * off-diagonal terms of the covariance matrix.
     * @param[out] out_geo_rdr         Raster to which the radar-grid
     * positions (range and azimuth) of the geogrid pixels vertices will be
     * saved.
     * @param[out] out_geo_dem         Raster to which the interpolated DEM
     * will be saved.
     * @param[out] out_geo_nlooks      Raster to which the number of radar-grid
     * looks associated with the geogrid will be saved.
     * @param[out] out_geo_rtc         Output RTC area factor (in
     * geo-coordinates).
     * @param[out] out_geo_rtc_gamma0_to_sigma0 Output RTC area factor 
     * gamma0 to sigma0 (in geo-coordinates).
     * \param[in]  az_time_correction     geo2rdr azimuth additive correction, in
     * seconds, as a function of azimuth and range
     * \param[in]  slant_range_correction  geo2rdr slant range additive correction,
     * in meters, as a function of azimuth and range
     * @param[in]  input_rtc              Input RTC area factor (in slant-range geometry).
     * @param[out] output_rtc             Output RTC area factor (in slant-range geometry).
     * @param[in]  input_layover_shadow_mask_raster Input layover/shadow mask raster
     * (in radar geometry). Samples identified as SHADOW or LAYOVER_AND_SHADOW are
     * considered invalid.
     * @param[in]  sub_swaths          Sub-swaths metadata
     * @param[in]  apply_valid_samples_sub_swath_masking Flag indicating whether the
     * valid-samples sub-swath masking should be applied during geocoding.
     * If not given, then sub-swath masking will be applied if the sub_swaths
     * parameter is provided.
     * @param[out] out_mask            Output valid-pixels
     * sub-swath mask raster
     * @param[in]  geocode_memory_mode Select memory mode
     * @param[in]  min_block_size      Minimum block size (per thread)
     * @param[in]  max_block_size      Maximum block size (per thread)
     * @param[in]  dem_interp_method   DEM interpolation method
     */
    template<class T_out>
    void geocodeAreaProj(
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::io::Raster& input_raster, isce3::io::Raster& output_raster,
            isce3::io::Raster& dem_raster,
            double geogrid_upsampling = 1,
            bool flag_upsample_radar_grid = false,
            bool flag_apply_rtc = false,
            isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry =
                    isce3::geometry::rtcInputTerrainRadiometry::BETA_NAUGHT,
            isce3::geometry::rtcOutputTerrainRadiometry
                    output_terrain_radiometry = isce3::geometry::
                            rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
            float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
            double rtc_geogrid_upsampling =
                    std::numeric_limits<double>::quiet_NaN(),
            isce3::geometry::rtcAlgorithm rtc_algorithm =
                    isce3::geometry::rtcAlgorithm::RTC_AREA_PROJECTION,
            isce3::geometry::rtcAreaBetaMode rtc_area_beta_mode =
                isce3::geometry::rtcAreaBetaMode::AUTO,
            double abs_cal_factor = 1,
            float clip_min = std::numeric_limits<float>::quiet_NaN(),
            float clip_max = std::numeric_limits<float>::quiet_NaN(),
            float min_nlooks = std::numeric_limits<float>::quiet_NaN(),
            float radar_grid_nlooks = 1,
            isce3::io::Raster* out_off_diag_terms = nullptr,
            isce3::io::Raster* out_geo_rdr = nullptr,
            isce3::io::Raster* out_geo_dem = nullptr,
            isce3::io::Raster* out_geo_nlooks = nullptr,
            isce3::io::Raster* out_geo_rtc = nullptr,
            isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0 = nullptr,
            const isce3::core::LUT2d<double>& az_time_correction = {},
            const isce3::core::LUT2d<double>& slant_range_correction = {},
            isce3::io::Raster* input_rtc = nullptr,
            isce3::io::Raster* output_rtc = nullptr,
            isce3::io::Raster* input_layover_shadow_mask_raster = nullptr,
            isce3::product::SubSwaths* sub_swaths = nullptr,
            std::optional<bool> apply_valid_samples_sub_swath_masking = std::nullopt,
            isce3::io::Raster* out_mask = nullptr,
            isce3::core::GeocodeMemoryMode geocode_memory_mode =
                    isce3::core::GeocodeMemoryMode::Auto,
            const long long min_block_size =
                    isce3::core::DEFAULT_MIN_BLOCK_SIZE,
            const long long max_block_size =
                    isce3::core::DEFAULT_MAX_BLOCK_SIZE,
            isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD);

    /** Set the output geogrid
     * @param[in]  geoGridStartY       Starting Lat/Northing position
     * @param[in]  geoGridSpacingY     Lat/Northing step size
     * @param[in]  geoGridStartX       Starting Lon/Easting position
     * @param[in]  geoGridSpacingX     Lon/Easting step size
     * @param[in]  geogrid_width       Geographic width (number of pixels) in
     * the Lon/Easting direction
     * @param[in]  geogrid_length      Geographic length (number of pixels) in
     * the Lat/Northing direction
     * @param[in]  epsgcode            Output geographic grid EPSG
     */
    void geoGrid(double geoGridStartX, double geoGridStartY,
                 double geoGridSpacingX, double geoGridSpacingY, int width,
                 int length, int epsgcode);

    /** Update the output geogrid with radar grid and DEM attributes
     * @param[in]  radar_grid          Radar grid
     * @param[in]  dem_raster          Input DEM raster
     */
    void updateGeoGrid(const isce3::product::RadarGridParameters& radar_grid,
                       isce3::io::Raster& dem_raster);

    // Get/set data interpolator
    isce3::core::dataInterpMethod dataInterpolator() const 
    { 
            return _data_interp_method; 
    }

    void dataInterpolator(isce3::core::dataInterpMethod method)
    {
        _data_interp_method = method;
    }

    void doppler(isce3::core::LUT2d<double> doppler) { _doppler = doppler; }

    void nativeDoppler(isce3::core::LUT2d<double> nativeDoppler)
    {
        _nativeDoppler = nativeDoppler;
    }

    void orbit(isce3::core::Orbit& orbit) { _orbit = orbit; }

    void ellipsoid(isce3::core::Ellipsoid& ellipsoid)
    {
        _ellipsoid = ellipsoid;
    }

    void thresholdGeo2rdr(double threshold) { _threshold = threshold; }

    void numiterGeo2rdr(int numiter) { _numiter = numiter; }

    void radarBlockMargin(int radarBlockMargin)
    {
        _radarBlockMargin = radarBlockMargin;
    }

    // start X position for the output geogrid
    double geoGridStartX() const { return _geoGridStartX; }

    // start Y position for the output geogrid
    double geoGridStartY() const { return _geoGridStartY; }

    // X spacing for the output geogrid
    double geoGridSpacingX() const { return _geoGridSpacingX; }

    // Y spacing for the output geogrid
    double geoGridSpacingY() const { return _geoGridSpacingY; }

    // number of pixels in east-west direction (X direction)
    int geoGridWidth() const { return _geoGridWidth; }

    // number of lines in north-south direction (Y direction)
    int geoGridLength() const { return _geoGridLength; }

private:
    /*
    Get radar grid boundaries (offsets and window size) based on
    the Geocode object geogrid attributes.
    */
    void _getRadarGridBoundaries(
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::io::Raster& input_raster, isce3::io::Raster& dem_raster,
            isce3::core::ProjectionBase* proj, double geogrid_upsampling,
            bool flag_upsample_radar_grid,
            isce3::core::dataInterpMethod dem_interp_method, int* offset_y,
            int* offset_x, int* grid_size_y, int* grid_size_x);

    /*
    Compute radar positions (az, rg, DEM vect.) for a geogrid vector
    (e.g. geogrid border) in X or Y direction (defined by flag_direction_line).
    If flag_compute_min_max is True, the function also return the min/max
    az. and rg. positions
    */
    void _getRadarPositionVect(double dem_y1, const int k_start,
            const int k_end, double geogrid_upsampling, double* a11,
            double* r11, double* y_min, double* x_min, double* y_max,
            double* x_max,
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::core::ProjectionBase* proj,
            isce3::geometry::DEMInterpolator& dem_interp_block,
            const std::function<Vec3(double, double,
                    const isce3::geometry::DEMInterpolator&,
                    isce3::core::ProjectionBase*)>& getDemCoords,
            bool flag_direction_line, bool flag_save_vectors,
            bool flag_compute_min_max,
            const isce3::core::LUT2d<double>& az_time_correction,
            const isce3::core::LUT2d<double>& slant_range_correction,
            std::vector<double>* a_last = nullptr,
            std::vector<double>* r_last = nullptr,
            std::vector<Vec3>* dem_last = nullptr);
    /*
    Check if a geogrid bounding box (y0, x0, yf, xf) fully
    covers the RSLC (represented by the radar_grid).
    */
    bool _checkLoadEntireRslcCorners(const double y0, const double x0,
            const double yf, const double xf,
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::core::ProjectionBase* proj,
            const std::function<Vec3(double, double,
                    const isce3::geometry::DEMInterpolator&,
                    isce3::core::ProjectionBase*)>& getDemCoords,
            isce3::geometry::DEMInterpolator& dem_interp, int margin_pixels);

    /*
    Get radar grid boundaries, i.e. min and max rg. and az. indexes, using
    the border of a geogrid bounding box.
    */
    void _getRadarPositionBorder(double geogrid_upsampling, const double dem_y1,
            const double dem_x1, const double dem_yf, const double dem_xf,
            double* a_min, double* r_min, double* a_max, double* r_max,
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::core::ProjectionBase* proj,
            const std::function<Vec3(double, double,
                    const isce3::geometry::DEMInterpolator&,
                    isce3::core::ProjectionBase*)>& getDemCoords,
            isce3::geometry::DEMInterpolator& dem_interp,
            const isce3::core::LUT2d<double>& az_time_correction = {},
            const isce3::core::LUT2d<double>& slant_range_correction = {});

    template<class T2, class T_out>
    void _runBlock(const isce3::product::RadarGridParameters& radar_grid,
            bool is_radar_grid_single_block,
            std::vector<std::unique_ptr<isce3::core::Matrix<T2>>>& rdrData,
            int block_size_y, int block_size_with_upsampling_y, int block_y,
            int block_size_x, int block_size_with_upsampling_x, int block_x,
            long long& numdone, const long long& progress_block,
            double geogrid_upsampling, int nbands, int nbands_off_diag_terms,
            isce3::core::dataInterpMethod dem_interp_method,
            isce3::io::Raster& dem_raster,
            isce3::io::Raster* out_off_diag_terms,
            isce3::io::Raster* out_geo_rdr, isce3::io::Raster* out_geo_dem,
            isce3::io::Raster* out_geo_nlooks, isce3::io::Raster* out_geo_rtc,
            isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0,
            isce3::core::ProjectionBase* proj, bool flag_apply_rtc,
            isce3::io::Raster* rtc_raster,
            isce3::io::Raster* rtc_gamma0_to_sigma0_raster,
            const isce3::core::LUT2d<double>& az_time_correction,
            const isce3::core::LUT2d<double>& slant_range_correction,
            isce3::io::Raster& input_raster,
            int raster_offset_y, int raster_offset_x,
            isce3::io::Raster& output_raster,
            isce3::core::Matrix<float>& rtc_area,
            isce3::core::Matrix<float>& rtc_area_sigma,
            float rtc_min_value, double abs_cal_factor,
            float clip_min, float clip_max,
            float min_nlooks, float radar_grid_nlooks,
            bool flag_upsample_radar_grid,
            isce3::io::Raster* input_layover_shadow_mask_raster,
            isce3::core::Matrix<uint8_t>& input_layover_shadow_mask,
            isce3::product::SubSwaths * sub_swaths,
            bool apply_valid_samples_sub_swath_masking,
            isce3::io::Raster* out_mask,
            isce3::core::GeocodeMemoryMode geocode_memory_mode,
            const long long min_block_size, const long long max_block_size,
            pyre::journal::info_t& info);

    std::string _get_nbytes_str(long nbytes);

    int _geo2rdr(const isce3::product::RadarGridParameters& radar_grid,
            double x, double y, double& azimuthTime, double& slantRange,
            isce3::geometry::DEMInterpolator& demInterp,
            isce3::core::ProjectionBase* proj, float& dem_value);

    /**
     * @param[in] rdrDataBlock a basebanded block of data in radar coordinate
     * @param[out] geoDataBlock a block of data in geo coordinates
     * @param[in] radarX the radar-coordinates x-index of the pixels in geo-grid
     * @param[in] radarY the radar-coordinates y-index of the pixels in geo-grid
     * @param[in] radarBlockWidth width of the data block in radar coordinates
     * @param[in] radarBlockLength length of the data block in radar coordinates
     * @param[in] azimuthFirstLine azimuth time of the first sample
     * @param[in] rangeFirstPixel  range of the first sample
     * @param[in] interp interpolator object
     * @param[in] radarGrid radar grid parameter
     * @param[in] nativeDoppler 2D LUT Doppler of the SLC image
     * @param[in] flatten flag to flatten the geocoded SLC
     * @param[in] phase_screen_raster Phase screen raster
     * @param[in] phase_screen_array  Phase screen array
     * @param[in] abs_cal_factor      Absolute calibration factor.
     * @param[in] clip_min            Clip (limit) minimum output values
     * @param[in] clip_max            Clip (limit) maximum output values
     * @param[in] flag_run_rtc        Flag to indicate if RTC is enabled
     * @param[in] rtc_area            RTC area normalization factor array
     * @param[in] rtc_area_sigma      RTC area normalization factor array
     * gamma0 to sigma0
     * @param[out] out_geo_rtc        Output RTC area factor raster (in
     * geo-coordinates)
     * @param[out] out_geo_rtc_array  Output RTC area factor array (in
     * geo-coordinates)
     * @param[out] out_geo_rtc_gamma0_to_sigma0  Output RTC area factor 
     * gamma0 to sigma0 raster (in geo-coordinates).
     * @param[out] out_geo_rtc_gamma0_to_sigma0_array Output RTC area factor 
     * gamma0 to sigma0 array (in geo-coordinates).
     * @param[in]  input_layover_shadow_mask_raster Input layover/shadow mask raster
     * (in radar geometry). Samples identified as SHADOW or LAYOVER_AND_SHADOW are
     * considered invalid.
     * @param[in]  input_layover_shadow_mask Input layover/shadow mask
     * array (in radar geometry). Samples identified as SHADOW or LAYOVER_AND_SHADOW are
     * considered invalid.
     * @param[in]  sub_swaths         Sub-swaths metadata
     * @param[out] out_mask           Output valid-pixels
     * sub-swath mask raster
     * @param[out] out_mask_array     Output valid-pixels
     * sub-swath mask array
     */
    template<class T_out>
    inline void _interpolate(const isce3::core::Matrix<T_out>& rdrDataBlock,
            isce3::core::Matrix<T_out>& geoDataBlock,
            const std::valarray<double>& radarX,
            const std::valarray<double>& radarY, const int radarBlockWidth,
            const int radarBlockLength, const int azimuthFirstLine,
            const int rangeFirstPixel,
            const isce3::core::Interpolator<T_out>* interp,
            const isce3::product::RadarGridParameters& radarGrid,
            const bool flag_az_baseband_doppler, const bool flatten,
            isce3::io::Raster* phase_screen_raster,
            isce3::core::Matrix<float>& phase_screen_array,
            double abs_cal_factor, float clip_min, float clip_max,
            bool flag_run_rtc, const isce3::core::Matrix<float>& rtc_area,
            const isce3::core::Matrix<float>& rtc_area_sigma,
            isce3::io::Raster* out_geo_rtc,
            isce3::core::Matrix<float>& out_geo_rtc_array,
            isce3::io::Raster* out_geo_rtc_gamma0_to_sigma0,
            isce3::core::Matrix<float>& out_geo_rtc_gamma0_to_sigma0_array,
            isce3::io::Raster* input_layover_shadow_mask_raster,
            isce3::core::Matrix<uint8_t>& input_layover_shadow_mask,
            isce3::product::SubSwaths * sub_swaths,
            isce3::io::Raster* out_mask,
            isce3::core::Matrix<uint8_t>& out_mask_array);

    /**
     * param[in,out] data a matrix of data that needs to be base-banded in
     * azimuth param[in] starting_range starting range of the data block
     * param[in] sensing_start starting azimuth time of the data block
     * param[in] range_pixel_spacing spacing of the slant range
     * param[in] prf pulse repetition frequency
     * param[in] doppler_lut 2D LUT of the image Doppler
     */
    template<class T2>
    inline void _baseband(isce3::core::Matrix<T2>& data,
            const double starting_range, const double sensing_start,
            const double range_pixel_spacing, const double prf,
            const isce3::core::LUT2d<double>& doppler_lut);

    /**
     * param[in,out] data a matrix of data that needs to be base-banded in
     * azimuth param[in] starting_range starting range of the data block
     * param[in] sensing_start starting azimuth time of the data block
     * param[in] range_pixel_spacing spacing of the slant range
     * param[in] prf pulse repetition frequency
     * param[in] doppler_lut 2D LUT of the image Doppler
     */
    template<class T2>
    inline void _baseband(isce3::core::Matrix<std::complex<T2>>& data,
            const double starting_range, const double sensing_start,
            const double range_pixel_spacing, const double prf,
            const isce3::core::LUT2d<double>& doppler_lut);

    void _print_parameters(pyre::journal::info_t& channel, 
                           isce3::core::GeocodeMemoryMode& geocode_memory_mode,
                           const long long min_block_size,
                           const long long max_block_size);

    // isce3::core objects
    isce3::core::Orbit _orbit;
    isce3::core::Ellipsoid _ellipsoid;

    // Optimization options

    double _threshold = 1e-8;
    int _numiter = 100;

    // radar grids parameters
    isce3::core::LUT2d<double> _doppler;

    // native Doppler
    isce3::core::LUT2d<double> _nativeDoppler;

    // start X position for the output geogrid
    double _geoGridStartX = std::numeric_limits<double>::quiet_NaN();

    // start Y position for the output geogrid
    double _geoGridStartY = std::numeric_limits<double>::quiet_NaN();

    // X spacing for the output geogrid
    double _geoGridSpacingX = std::numeric_limits<double>::quiet_NaN();

    // Y spacing for the output geogrid
    double _geoGridSpacingY = std::numeric_limits<double>::quiet_NaN();

    // number of pixels in east-west direction (X direction)
    int _geoGridWidth = -32768;

    // number of lines in north-south direction (Y direction)
    int _geoGridLength = -32768;

    // epsg code for the output geogrid
    int _epsgOut = 0;

    // margin around the computed bounding box for radar dara (integer number of
    // lines/pixels)
    int _radarBlockMargin;

    // interpolator
    isce3::core::dataInterpMethod _data_interp_method =
            isce3::core::dataInterpMethod::BIQUINTIC_METHOD;
};

}} // namespace isce3::geocode


// Get inline implementations for Geocode
#define ISCE_GEOMETRY_GEOCODE_ICC
#include "GeocodeCov.icc"
#undef ISCE_GEOMETRY_GEOCODE_ICC
