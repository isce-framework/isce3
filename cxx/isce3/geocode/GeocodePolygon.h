#pragma once

// pyre
#include <pyre/journal.h>

// isce3::core
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>

// isce3::io
#include <isce3/io/Raster.h>

// isce3::product
#include <isce3/product/RadarGridParameters.h>

// isce3::geometry
#include <isce3/geometry/RTC.h>


namespace isce3 { namespace geocode {

template<class T>
class GeocodePolygon {
public:

    /** Calculate the mean value of radar-grid samples using a polygon defined
     * over geographical coordinates.
     *
     * @param[in]  x_vect              Polygon vertices Lon/Easting positions
     * @param[in]  y_vect              Polygon vertices Lon/Easting positions
     * @param[in]  radar_grid          Radar grid
     * @param[in]  orbit               Orbit
     * @param[in]  input_dop           Doppler LUT associated with the radar grid
     * @param[in]  dem_raster          Input DEM raster
     * @param[in]  threshold           Azimuth time threshold for convergence (s)
     * @param[in]  num_iter            Maximum number of Newton-Raphson iterations
     * @param[in]  delta_range         Step size used for computing Doppler derivative
     */
    GeocodePolygon(const std::vector<double>& x_vect,
                   const std::vector<double>& y_vect,
                   const isce3::product::RadarGridParameters& radar_grid,
                   const isce3::core::Orbit& orbit,
                   const isce3::core::Ellipsoid& ellipsoid,
                   const isce3::core::LUT2d<double>& input_dop,
                   isce3::io::Raster& dem_raster, double threshold = 1e-8,
                   int num_iter = 100, double delta_range = 1e-8);

    /** Calculate the mean value of radar-grid samples using a polygon defined
     * over geographical coordinates.
     *
     * @param[in]  radar_grid          Radar grid
     * @param[in]  input_dop           Doppler LUT associated with the radar grid
     * @param[in]  input_raster        Input raster
     * @param[out] output_raster       Output raster
     * @param[in]  dem_raster          Input DEM raster
     * @param[in]  flag_apply_rtc      Apply radiometric terrain correction (RTC)
     * @param[in]  input_terrain_radiometry    Terrain radiometry of the input raster
     * @param[in]  output_terrain_radiometry Output terrain radiometr
     * @param[in]  exponent            Exponent to be applied to the input data.
     * The value 0 indicates that the the exponent is based on the data type of
     * the input raster (1 for real and 2 for complex rasters).
     * @param[in]  geogrid_upsampling  Geogrid upsampling (in each direction)
     * @param[in]  rtc_min_value_db    Minimum value for the RTC area factor.
     * Radar data with RTC area factor below this limit are ignored.
     * @param[in]  abs_cal_factor      Absolute calibration factor.
     * @param[in]  radar_grid_nlooks   Radar grid number of looks. This
     * parameters determines the multilooking factor used to compute out_nlooks.
     * @param[out] output_off_diag_terms Output raster containing the 
     * off-diagonal terms of the covariance matrix.
     * @param[out] output_radargrid_data Radar-grid data multiplied by the
     * weights that was used to compute the polygon average backscatter
     * @param[out] output_rtc          Output RTC area factor (in slant-range).
     * @param[out] output_weights      Polygon weights (level of intersection
     * between the polygon with the radar grid).
     * @param[in]  interp_method       Data interpolation method
     */
    void getPolygonMean(
            const isce3::product::RadarGridParameters& radar_grid,
            const isce3::core::LUT2d<double>& input_dop,
            isce3::io::Raster& input_raster,
            isce3::io::Raster& output_raster,
            isce3::io::Raster& dem_raster,
            bool flag_apply_rtc = false,
            isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry =
                    isce3::geometry::rtcInputTerrainRadiometry::BETA_NAUGHT,
            isce3::geometry::rtcOutputTerrainRadiometry
                    output_terrain_radiometry = isce3::geometry::
                            rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
            int exponent = 0,
            double geogrid_upsampling =
                    std::numeric_limits<double>::quiet_NaN(),
            float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN(),
            double abs_cal_factor = 1, float radar_grid_nlooks = 1,
            isce3::io::Raster* output_off_diag_terms = nullptr,
            isce3::io::Raster* output_radargrid_data = nullptr,
            isce3::io::Raster* output_rtc = nullptr,
            isce3::io::Raster* output_weights = nullptr,
            isce3::core::dataInterpMethod interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD);

    // Radar grid X offset
    int xoff() const { return _xoff; }

    // Radar grid Y offset
    int yoff() const { return _yoff; }

    // Cropped radar grid X size
    int xsize() const { return _xsize; }

    // Cropped radar grid Y size
    int ysize() const { return _ysize; }

    // Output number of looks
    float out_nlooks() const { return _out_nlooks; }

private:

    std::vector<double> _az_time_vect, _slant_range_vect;
    int _xoff;
    int _yoff;
    int _xsize;
    int _ysize;

    float _out_nlooks;

    double _threshold;
    int _num_iter;
    double _delta_range;
        
    isce3::core::Orbit _orbit;

    template<class T_out>
    void _getPolygonMean(
            isce3::core::Matrix<float>& rtc_area,
            const isce3::product::RadarGridParameters& radar_grid,
            isce3::io::Raster& input_raster,
            isce3::io::Raster& output_raster,
            bool flag_apply_rtc = false,
            float rtc_min_value = 0, 
            double abs_cal_factor = 1, float radar_grid_nlooks = 1,
            isce3::io::Raster* output_off_diag_terms = nullptr,
            isce3::io::Raster* output_radargrid_data = nullptr,
            isce3::io::Raster* output_weights = nullptr);

    void _ValidatePolygon(
            const isce3::product::RadarGridParameters& radar_grid);

};

}} // namespace isce3::geocode
