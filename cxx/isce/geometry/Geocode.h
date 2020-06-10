//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Gustavo H. X. Shiroma
// Copyright 2019-

#pragma once

// pyre
#include <pyre/journal.h>

// isce::core
#include <isce/core/Orbit.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Constants.h>

// isce::io
#include <isce/io/Raster.h>

// isce::product
#include <isce/product/Product.h>
#include <isce/product/RadarGridParameters.h>

// isce::geometry
#include <isce/geometry/RTC.h>

#include "geometry.h"

namespace isce {
namespace geometry {

/** Enumeration type to indicate the algorithm used for geocoding */
enum geocodeOutputMode {
    INTERP = 0,
    AREA_PROJECTION = 1,
    AREA_PROJECTION_GAMMA_NAUGHT = 2
};

/** Enumeration type to indicate memory management */
enum geocodeMemoryMode {
    AUTO = 0,
    SINGLE_BLOCK = 1,
    BLOCKS_GEOGRID = 2,
    BLOCKS_GEOGRID_AND_RADARGRID = 3
};

template<class T> class Geocode {
public:

    /** Geocode data from slant-range to map coordinates
     *
     * @param[in]  radar_grid          Radar grid
     * @param[in]  input_raster        Input raster
     * @param[out] output_raster       Output raster
     * @param[in]  dem_raster          Input DEM raster
     * @param[in]  output_mode         Geocode method
     * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
     * @param[in]  input_radiometry    Terrain radiometry of the input raster
     * @param[in]  exponent            Exponent to be applied to the input data.
     * The value 0 indicates that the the exponent is based on the data type of
     * the input raster (1 for real and 2 for complex rasters).
     * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor.
     * Radar data with RTC area factor below this limit are ignored.
     * @param[in]  rtc_geogrid_upsampling  Geogrid upsampling (in each direction)
     * used to compute the radiometric terrain correction RTC.
     * @param[in]  rtc_algorithm       RTC algorithm (RTC_DAVID_SMALL or
     * RTC_AREA_PROJECTION)
     * @param[in]  abs_cal_factor      Absolute calibration factor.
     * @param[in]  clip_min            Clip (limit) minimum output values
     * @param[in]  clip_max            Clip (limit) maximum output values
     * @param[in]  min_nlooks          Minimum number of looks. Geogrid data
     * below this limit will be set to NaN 
     * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
     * parameters determines the multilooking factor used to compute
     * out_geo_nlooks.
     * @param[out] out_geo_vertices      Raster to which the radar-grid
     * positions (range and azimuth) of the geogrid pixels vertices will be
     * saved.
     * @param[out] out_dem_vertices     Raster to which the interpolated DEM
     * will be saved.
     * @param[out] out_nlooks          Raster to which the number of radar-grid
     * looks associated with the geogrid will be saved.
     * @param[out] out_geo_rtc         Output RTC area factor (in
     * geo-coordinates).
     * @param[in]  in_rtc               Input RTC area factor (in slant-range).
     * @param[out] out_rtc              Output RTC area factor (in slant-range).
     * @param[in]  geocode_memory_mode  Select memory mode 
     * @param[in]  interp_method        Data interpolation method
     */
    void
    geocode(const isce::product::RadarGridParameters& radar_grid,
            isce::io::Raster& input_raster, isce::io::Raster& output_raster,
            isce::io::Raster& dem_raster,
            isce::geometry::geocodeOutputMode output_mode =
                    isce::geometry::geocodeOutputMode::INTERP,
            double geogrid_upsampling = 1,
            isce::geometry::rtcInputRadiometry input_radiometry =
                    isce::geometry::rtcInputRadiometry::BETA_NAUGHT,
            int exponent = 0,
            float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
            double rtc_geogrid_upsampling = 
                std::numeric_limits<double>::quiet_NaN(),
            rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
            double abs_cal_factor = 1,
            float clip_min = std::numeric_limits<float>::quiet_NaN(),
            float clip_max = std::numeric_limits<float>::quiet_NaN(),
            float min_nlooks = std::numeric_limits<float>::quiet_NaN(),
            float radar_grid_nlooks = 1,
            isce::io::Raster* out_geo_vertices = nullptr,
            isce::io::Raster* out_dem_vertices = nullptr,
            isce::io::Raster* out_geo_nlooks = nullptr,
            isce::io::Raster* out_geo_rtc = nullptr,
            isce::io::Raster* input_rtc = nullptr,
            isce::io::Raster* output_rtc = nullptr,
            geocodeMemoryMode geocode_memory_mode = geocodeMemoryMode::AUTO,
            isce::core::dataInterpMethod interp_method =
                    isce::core::dataInterpMethod::BIQUINTIC_METHOD);

    /** Geocode using the interpolation algorithm.
     *
     * @param[in]  radar_grid          Radar grid
     * @param[in]  input_raster        Input raster
     * @param[out] output_raster       Output raster
     * @param[in]  dem_raster          Input DEM raster
     */
    template<class T_out>
    void geocodeInterp(const isce::product::RadarGridParameters& radar_grid,
                       isce::io::Raster& input_raster,
                       isce::io::Raster& output_raster,
                       isce::io::Raster& demRaster);

    /** Geocode using the area projection algorithm (adaptive multilooking)
     *
     * @param[in]  radar_grid          Radar grid
     * @param[in]  input_raster        Input raster
     * @param[out] output_raster       Output raster
     * @param[in]  dem_raster          Input DEM raster
     * @param[in]  output_mode         Output mode
     * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
     * @param[in]  input_radiometry    Terrain radiometry of the input raster
     * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor.
     * Radar data with RTC area factor below this limit are ignored.
     * @param[in]  rtc_geogrid_upsampling  Geogrid upsampling (in each direction)
     * used to compute the radiometric terrain correction RTC.
     * @param[in]  rtc_algorithm       RTC algorithm (RTC_DAVID_SMALL or
     * RTC_AREA_PROJECTION)
     * @param[in]  abs_cal_factor      Absolute calibration factor.
     * @param[in]  clip_min            Clip (limit) minimum output values
     * @param[in]  clip_max            Clip (limit) maximum output values
     * @param[in]  min_nlooks          Minimum number of looks. Geogrid data
     * below this limit will be set to NaN 
     * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
     * parameters determines the multilooking factor used to compute out_nlooks.
     * @param[out] out_geo_vertices       Raster to which the radar-grid
     * positions (range and azimuth) of the geogrid pixels vertices will be
     * saved.
     * @param[out] out_dem_vertices     Raster to which the interpolated DEM
     * will be saved.
     * @param[out] out_geo_nlooks      Raster to which the number of radar-grid
     * looks associated with the geogrid will be saved.
     * @param[out] out_geo_rtc         Output RTC area factor (in
     * geo-coordinates).
     * @param[in]  in_rtc              Input RTC area factor (in slant-range).
     * @param[out] out_rtc             Output RTC area factor (in slant-range).
     * @param[in]  interp_method       Data interpolation method
     * @param[in]  geocode_memory_mode Select memory mode
     */
    template<class T_out>
    void geocodeAreaProj(
            const isce::product::RadarGridParameters& radar_grid,
            isce::io::Raster& input_raster, isce::io::Raster& output_raster,
            isce::io::Raster& dem_raster,
            isce::geometry::geocodeOutputMode output_mode =
                    isce::geometry::geocodeOutputMode::AREA_PROJECTION,
            double geogrid_upsampling = 1,
            isce::geometry::rtcInputRadiometry input_radiometry =
                    isce::geometry::rtcInputRadiometry::BETA_NAUGHT,
            float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
            double rtc_geogrid_upsampling = 
                std::numeric_limits<double>::quiet_NaN(),
            rtcAlgorithm rtc_algorithm = rtcAlgorithm::RTC_AREA_PROJECTION,
            double abs_cal_factor = 1, 
            float clip_min = std::numeric_limits<float>::quiet_NaN(),
            float clip_max = std::numeric_limits<float>::quiet_NaN(),
            float min_nlooks = std::numeric_limits<float>::quiet_NaN(),
            float radar_grid_nlooks = 1,
            isce::io::Raster* out_geo_vertices = nullptr,
            isce::io::Raster* out_dem_vertices = nullptr,
            isce::io::Raster* out_geo_nlooks = nullptr,
            isce::io::Raster* out_geo_rtc = nullptr,
            isce::io::Raster* input_rtc = nullptr,
            isce::io::Raster* output_rtc = nullptr,
            geocodeMemoryMode geocode_memory_mode = geocodeMemoryMode::AUTO,
            isce::core::dataInterpMethod interp_method =
                    isce::core::dataInterpMethod::BIQUINTIC_METHOD);

    /** Set the output geogrid
     * @param[in]  geoGridStartY       Starting lat/northing position
     * @param[in]  geoGridSpacingY     Lat/northing step size
     * @param[in]  geoGridStartX       Starting lon/easting position
     * @param[in]  geoGridSpacingX     Lon/easting step size
     * @param[in]  geogrid_width       Geographic width (number of pixels) in
     * the lon/easting direction
     * @param[in]  geogrid_length      Geographic length (number of pixels) in
     * the lat/northing direction
     * @param[in]  epsgcode            Output geographic grid EPSG
     */
    void geoGrid(double geoGridStartX, double geoGridStartY,
                 double geoGridSpacingX, double geoGridSpacingY, int width,
                 int length, int epsgcode);

    /** Update the output geogrid with radar grid and DEM attributes
     * @param[in]  radar_grid          Radar grid
     * @param[in]  dem_raster          Input DEM raster
     */
    void updateGeoGrid(const isce::product::RadarGridParameters& radar_grid,
                       isce::io::Raster& dem_raster);

    // Set interpolator
    void interpolator(isce::core::dataInterpMethod method) {
        _interp_method = method;
    }

    void doppler(isce::core::LUT2d<double> doppler) { _doppler = doppler; }

    void orbit(isce::core::Orbit& orbit) { _orbit = orbit; }

    void ellipsoid(isce::core::Ellipsoid& ellipsoid) { _ellipsoid = ellipsoid; }

    void thresholdGeo2rdr(double threshold) { _threshold = threshold; }

    void numiterGeo2rdr(int numiter) { _numiter = numiter; }

    void linesPerBlock(size_t linesPerBlock) { _linesPerBlock = linesPerBlock; }

    void demBlockMargin(double demBlockMargin) {
        _demBlockMargin = demBlockMargin;
    }

    void radarBlockMargin(int radarBlockMargin) {
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
    void _GetRadarPositionVect(
            double dem_y1, const int k_start, const int k_end,
            double geogrid_upsampling, double& a11, double& r11, double& a_min,
            double& r_min, double& a_max, double& r_max,
            std::vector<double>& a_last, std::vector<double>& r_last,
            std::vector<Vec3>& dem_last,
            const isce::product::RadarGridParameters& radar_grid,
            isce::core::ProjectionBase* proj, DEMInterpolator& dem_interp_block,
            bool flag_direction_line);

    template<class T_out>
    void
    _RunBlock(const isce::product::RadarGridParameters& radar_grid,
              bool is_radar_grid_single_block,
              std::vector<std::unique_ptr<isce::core::Matrix<T_out>>>& rdrData,
              const int jmax, int block_size, int block_size_with_upsampling,
              int block, int& numdone, int progress_block,
              double geogrid_upsampling, int nbands,
              isce::core::dataInterpMethod interp_method,
              isce::io::Raster& dem_raster, isce::io::Raster* out_geo_vertices,
              isce::io::Raster* out_dem_vertices,
              isce::io::Raster* out_geo_nlooks, isce::io::Raster* out_geo_rtc,
              const double start, const double pixazm, const double dr,
              double r0, int xbound, int ybound,
              isce::core::ProjectionBase* proj,
              isce::core::Matrix<float>& rtc_area,
              isce::io::Raster& input_raster, isce::io::Raster& output_raster,
              isce::geometry::geocodeOutputMode output_mode,
              float rtc_min_value, double abs_cal_factor, float clip_min,
              float clip_max, float min_nlooks, float radar_grid_nlooks,
              pyre::journal::info_t& info);

    void _loadDEM(isce::io::Raster& demRaster, DEMInterpolator& demInterp,
                  isce::core::ProjectionBase* _proj, int lineStart,
                  int blockLength, int blockWidth, double demMargin);

    std::string _get_nbytes_str(long nbytes);

    void _geo2rdr(const isce::product::RadarGridParameters& radar_grid,
                  double x, double y, double& azimuthTime, double& slantRange,
                  DEMInterpolator& demInterp, isce::core::ProjectionBase* proj);

    template<class T_out>
    void
    _interpolate(isce::core::Matrix<T_out>& rdrDataBlock,
                 isce::core::Matrix<T_out>& geoDataBlock,
                 std::valarray<double>& radarX, std::valarray<double>& radarY,
                 int rdrBlockWidth, int rdrBlockLength, int azimuthFirstLine,
                 int rangeFirstPixel, isce::core::Interpolator<T_out>* interp);

    // isce::core objects
    isce::core::Orbit _orbit;
    isce::core::Ellipsoid _ellipsoid;

    // Optimization options
    double _threshold;
    int _numiter;
    size_t _linesPerBlock = 1000;

    // radar grids parameters
    isce::core::LUT2d<double> _doppler;

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

    // margin around a computed bounding box for DEM (in degrees)
    double _demBlockMargin;

    // margin around the computed bounding box for radar dara (integer number of
    // lines/pixels)
    int _radarBlockMargin;

    // interpolator
    isce::core::dataInterpMethod _interp_method =
            isce::core::dataInterpMethod::BIQUINTIC_METHOD;
};

std::vector<float> getGeoAreaElementMean(
        const std::vector<double>& x_vect, const std::vector<double>& y_vect,
        const isce::product::RadarGridParameters& radar_grid,
        const isce::core::Orbit& orbit,
        const isce::core::LUT2d<double>& input_dop,
        isce::io::Raster& input_raster, isce::io::Raster& dem_raster,
        isce::geometry::rtcInputRadiometry input_radiometry =
                isce::geometry::rtcInputRadiometry::BETA_NAUGHT,
        int exponent = 0,
        isce::geometry::geocodeOutputMode output_mode =
                isce::geometry::geocodeOutputMode::AREA_PROJECTION,
        double geogrid_upsampling = std::numeric_limits<double>::quiet_NaN(),
        float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
        double abs_cal_factor = 1, float radar_grid_nlooks = 1,
        float* out_nlooks = nullptr,
        isce::core::dataInterpMethod interp_method =
                isce::core::dataInterpMethod::BIQUINTIC_METHOD,
        double threshold = 1e-4, int num_iter = 100, double delta_range = 1e-4);

template<typename T>
std::vector<float> _getGeoAreaElementMean(
        const std::vector<double>& r_vect, const std::vector<double>& a_vect,
        int x_min, int y_min, isce::core::Matrix<float>& rtc_area,
        const isce::product::RadarGridParameters& radar_grid,
        isce::io::Raster& input_raster,
        isce::geometry::geocodeOutputMode output_mode =
                isce::geometry::geocodeOutputMode::AREA_PROJECTION,
        float rtc_min_value = 0, float* out_nlooks = nullptr,
        double abs_cal_factor = 1, float radar_grid_nlooks = 1);

} // namespace geometry
} // namespace isce

// Get inline implementations for Geocode
#define ISCE_GEOMETRY_GEOCODE_ICC
#include "Geocode.icc"
#undef ISCE_GEOMETRY_GEOCODE_ICC
