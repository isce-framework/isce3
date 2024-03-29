#include "RTC.h"

#include <isce3/io/Raster.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/product/RadarGridParameters.h>

#include <limits>

namespace py = pybind11;

using isce3::geometry::rtcAlgorithm;
using isce3::geometry::rtcInputTerrainRadiometry;
using isce3::geometry::rtcOutputTerrainRadiometry;
using isce3::geometry::rtcAreaMode;
using isce3::geometry::rtcAreaBetaMode;

void addbinding(py::enum_<rtcInputTerrainRadiometry>& pyInputTerrainRadiometry)
{
    pyInputTerrainRadiometry
            .value("BETA_NAUGHT", rtcInputTerrainRadiometry::BETA_NAUGHT)
            .value("SIGMA_NAUGHT_ELLIPSOID",
                    rtcInputTerrainRadiometry::SIGMA_NAUGHT_ELLIPSOID);
}

void addbinding(
        py::enum_<rtcOutputTerrainRadiometry>& pyOutputTerrainRadiometry)
{
    pyOutputTerrainRadiometry
            .value("SIGMA_NAUGHT", rtcOutputTerrainRadiometry::SIGMA_NAUGHT)
            .value("GAMMA_NAUGHT", rtcOutputTerrainRadiometry::GAMMA_NAUGHT);
}

void addbinding(py::enum_<rtcAlgorithm> & pyAlgorithm)
{
    pyAlgorithm
            .value("RTC_BILINEAR_DISTRIBUTION",
                    rtcAlgorithm::RTC_BILINEAR_DISTRIBUTION)
            .value("RTC_AREA_PROJECTION", rtcAlgorithm::RTC_AREA_PROJECTION);
}

void addbinding(py::enum_<rtcAreaMode> & pyAreaMode)
{
    pyAreaMode
        .value("AREA", rtcAreaMode::AREA)
        .value("AREA_FACTOR", rtcAreaMode::AREA_FACTOR);
}

void addbinding(py::enum_<rtcAreaBetaMode> & pyAreaBetaMode)
{
    pyAreaBetaMode
        .value("AUTO", rtcAreaBetaMode::AUTO,
                R"(auto mode. Default value is defined by the
                    RTC algorithm that is being executed, i.e.,
                    PIXEL_AREA for rtcAlgorithm::RTC_BILINEAR_DISTRIBUTION
                    and PROJECTION_ANGLE for
                    rtcAlgorithm::RTC_AREA_PROJECTION.)")
        .value("PIXEL_AREA", rtcAreaBetaMode::PIXEL_AREA,
                R"(estimate the beta surface reference area `A_beta`
                   using the pixel area, which is the
                   product of the range spacing by the
                   azimuth spacing (computed using the ground velocity).)")
        .value("PROJECTION_ANGLE", rtcAreaBetaMode::PROJECTION_ANGLE,
               R"(estimate the beta surface reference area `A_beta`
                  using the projection angle method:
                  `A_beta = A_sigma * cos(projection_angle).)");
}


void addbinding_apply_rtc(pybind11::module& m)
{
    m.def("apply_rtc", &isce3::geometry::applyRtc, py::arg("radar_grid"),
            py::arg("orbit"), py::arg("input_dop"), py::arg("input_raster"),
            py::arg("dem_raster"), py::arg("output_raster"),
            py::arg("input_terrain_radiometry") =
                    rtcInputTerrainRadiometry::BETA_NAUGHT,
            py::arg("output_terrain_radiometry") =
                    rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
            py::arg("exponent") = 0,
            py::arg("rtc_area_mode") =
                    rtcAreaMode::AREA_FACTOR,
            py::arg("rtc_algorithm") = rtcAlgorithm::RTC_AREA_PROJECTION,
            py::arg("rtc_area_beta_mode") =
                    rtcAreaBetaMode::AUTO,
            py::arg("geogrid_upsampling") =
                    std::numeric_limits<double>::quiet_NaN(),
            py::arg("rtc_min_value_db") =
                    std::numeric_limits<float>::quiet_NaN(),
            py::arg("abs_cal_factor") = 1,
            py::arg("clip_min") = std::numeric_limits<float>::quiet_NaN(),
            py::arg("clip_max") = std::numeric_limits<float>::quiet_NaN(),
            py::arg("out_sigma") = nullptr,
            py::arg("input_rtc") = nullptr, py::arg("output_rtc") = nullptr,
            py::arg("rtc_memory_mode") =
                    isce3::core::MemoryModeBlocksY::AutoBlocksY,
            R"(This function computes and applies the radiometric terrain correction (RTC) to a multi-band
              raster.

              Parameters
              ---------
              radar_grid : isce3.product.RadarGridParameters
                  Radar Grid
              orbit : isce3.core.Orbit
                  Orbit
              input_dop : isce3.core.LUT2d
                  Doppler LUT
              input_raster : isce3.io.Raster
                  Input raster
              dem_raster : isce3.io.Raster
                  Input DEM raster
              output_raster : isce3.io.Raster
                  Output raster (output)
              input_terrain_radiometry : isce3.geometry.RtcInputTerrainRadiometry, optional
                  Input terrain radiometry
              output_terrain_radiometry : isce3.geometry.RtcOutputTerrainRadiometry, optional
                  Output terrain radiometry
              exponent : int, optional
                  Exponent to be applied to the input data. The
                  value 0 indicates that the the exponent is based on the
                  data type of the input raster (1 for real and 2 for complex
                  rasters).
              rtc_area_mode : isce3.geometry.RtcAreaMode, optional
                  RTC area mode
              rtc_algorithm : isce3.geometry.RtcAlgorithm, optional
                  RTC algorithm
              rtc_area_beta_mode : isce3.geometry.RtcAreaBetaMode, optional
                  RTC area beta mode
              geogrid_upsampling : double, optional
                  Geogrid upsampling (in each direction)
              rtc_min_value_db : float, optional
                  Minimum value for the RTC area factor. Radar data with RTC
                  area factor below this limit are ignored.
              abs_cal_factor : double, optional
                  Absolute calibration factor.
              clip_min : float, optional
                  Clip minimum output values
              clip_max : float, optional
                  Clip maximum output values
              out_sigma : isce3.io.Raster, optional
                  Output sigma surface area (rtc_area_mode = AREA) or area
                  factor (rtc_area_mode = AREA_FACTOR) raster
              input_rtc : isce3.io.Raster, optional
                  Raster containing pre-computed RTC area factor
              output_rtc : isce3.io.Raster, optional
                  Output RTC area factor (output)
              rtc_memory_mode : isce3.core.MemoryModeBlocksY, optional
                  Select memory mode
              )");
}

void addbinding_compute_rtc(pybind11::module& m)
{
    const isce3::geometry::detail::Geo2RdrParams defaults;
    m.def("compute_rtc",
            py::overload_cast<const isce3::product::RadarGridParameters&,
                    const isce3::core::Orbit&,
                    const isce3::core::LUT2d<double>&, isce3::io::Raster&,
                    isce3::io::Raster&, rtcInputTerrainRadiometry,
                    rtcOutputTerrainRadiometry, rtcAreaMode,
                    rtcAlgorithm, rtcAreaBetaMode,
                    double, float, isce3::io::Raster*,
                    isce3::core::MemoryModeBlocksY,
                    isce3::core::dataInterpMethod, double, int, double,
                    const long long, const long long>(
                    &isce3::geometry::computeRtc),
            py::arg("radar_grid"), py::arg("orbit"), py::arg("input_dop"),
            py::arg("dem"), py::arg("output_raster"),
            py::arg("input_terrain_radiometry") =
                    rtcInputTerrainRadiometry::BETA_NAUGHT,
            py::arg("output_terrain_radiometry") =
                    rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
            py::arg("rtc_area_mode") =
                    rtcAreaMode::AREA_FACTOR,
            py::arg("rtc_algorithm") = rtcAlgorithm::RTC_AREA_PROJECTION,
            py::arg("rtc_area_beta_mode") =
                    rtcAreaBetaMode::AUTO,
            py::arg("geogrid_upsampling") =
                    std::numeric_limits<double>::quiet_NaN(),
            py::arg("rtc_min_value_db") =
                    std::numeric_limits<float>::quiet_NaN(),
            py::arg("out_sigma") = nullptr,
            py::arg("rtc_memory_mode") =
                    isce3::core::MemoryModeBlocksY::AutoBlocksY,
            py::arg("interp_method") =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
            py::arg("threshold") = defaults.threshold,
            py::arg("num_iter") = defaults.maxiter,
            py::arg("delta_range") = defaults.delta_range,
            py::arg("min_block_size") =
                    isce3::core::DEFAULT_MIN_BLOCK_SIZE,
            py::arg("max_block_size") =
                    isce3::core::DEFAULT_MAX_BLOCK_SIZE,
            R"(This function computes and applies the radiometric terrain correction
             (RTC) to a multi-band raster.

             Parameters
             ----------
             radar_grid : isce3.product.RadarGridParameters
                 Radar Grid
             orbit : isce3.core.Orbit
                 Orbit
             input_dop : isce3.core.LUT2d
                 Doppler LUT
             dem_raster : isce3.io.Raster
                 Input DEM raster
             output_raster : isce3.io.Raster
                 Output raster (output)
             input_terrain_radiometry : isce3.geometry.RtcInputTerrainRadiometry, optional
                 Terrain radiometry of the input raster
             output_terrain_radiometry : isce3.geometry.RtcOutputTerrainRadiometry, optional
                 Terrain radiometry of the input raster
             rtc_area_mode : isce3.geometry.RtcAreaMode, optional
                 RTC area mode
             rtc_algorithm : isce3.geometry.RtcAlgorithm, optional
                 RTC algorithm
             rtc_factor_area_mode : isce3.geometry.RtcAreaBetaMode, optional
                 RTC area beta mode
             geogrid_upsampling : double, optional
                 Geogrid upsampling
             rtc_min_value_db : float, optional
                 Minimum value for the RTC area factor. Radar
              data with RTC area factor below this limit are ignored.
             out_sigma : isce3.io.Raster, optional
                 Output sigma surface area (rtc_area_mode = AREA) or area
                 factor (rtc_area_mode = AREA_FACTOR) raster
             rtc_memory_mode : isce3.core.MemoryModeBlocksY, optional
                 Select memory mode
             interp_method : isce3.core.DataInterpMethod, optional
                 Interpolation Method
             threshold : double, optional
                 Azimuth time threshold for convergence (s)
             num_iter : int, optional
                 Maximum number of Newton-Raphson iterations
             delta_range : double, optional
                Step size used for computing Doppler derivative
             min_block_size : long long, optional
                Minimum block size
             max_block_size : long long, optional
                Maximum block size
             )");
}

void addbinding_compute_rtc_bbox(pybind11::module& m)
{
    const isce3::geometry::detail::Geo2RdrParams defaults;
    m.def("compute_rtc_bbox",
            py::overload_cast<isce3::io::Raster&, isce3::io::Raster&,
                    const isce3::product::RadarGridParameters&,
                    const isce3::core::Orbit&,
                    const isce3::core::LUT2d<double>&, const double,
                    const double, const double, const double, const int,
                    const int, const int, rtcInputTerrainRadiometry,
                    rtcOutputTerrainRadiometry, rtcAreaMode,
                    rtcAlgorithm, rtcAreaBetaMode,
                    double, float, isce3::io::Raster*,
                    isce3::io::Raster*, isce3::io::Raster*,
                    isce3::core::MemoryModeBlocksY,
                    isce3::core::dataInterpMethod, double, int, double,
                    const long long, const long long>(
                    &isce3::geometry::computeRtc),
            py::arg("dem_raster"), py::arg("output_raster"),
            py::arg("radar_grid"), py::arg("orbit"), py::arg("input_dop"),
            py::arg("y0"), py::arg("dy"), py::arg("x0"), py::arg("dx"),
            py::arg("geogrid_length"), py::arg("geogrid_width"),
            py::arg("epsg"),
            py::arg("input_terrain_radiometry") =
                    rtcInputTerrainRadiometry::BETA_NAUGHT,
            py::arg("output_terrain_radiometry") =
                    rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
            py::arg("rtc_area_mode") =
                    rtcAreaMode::AREA_FACTOR,
            py::arg("rtc_algorithm") = rtcAlgorithm::RTC_AREA_PROJECTION,
            py::arg("rtc_area_beta_mode") =
                    rtcAreaBetaMode::AUTO,
            py::arg("geogrid_upsampling") =
                    std::numeric_limits<double>::quiet_NaN(),
            py::arg("rtc_min_value_db") =
                    std::numeric_limits<float>::quiet_NaN(),
            py::arg("out_geo_rdr") = nullptr,
            py::arg("out_geo_grid") = nullptr, py::arg("out_sigma") = nullptr,
            py::arg("rtc_memory_mode") =
                    isce3::core::MemoryModeBlocksY::AutoBlocksY,
            py::arg("interp_method") =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
            py::arg("threshold") = defaults.threshold,
            py::arg("num_iter") = defaults.maxiter,
            py::arg("delta_range") = defaults.delta_range,
            py::arg("min_block_size") =
                    isce3::core::DEFAULT_MIN_BLOCK_SIZE,
            py::arg("max_block_size") =
                    isce3::core::DEFAULT_MAX_BLOCK_SIZE,
            R"(This function computes and applies the radiometric terrain correction
             (RTC) to a multi-band raster using a predefined geogrid.

              Parameters
             ----------
             dem_raster : isce3.io.Raster
                 Input DEM raster
             output_raster : isce3.io.Raster
                 Output raster (output)
             radar_grid : isce3.product.RadarGridParameters
                 Radar Grid
             orbit : isce3.core.Orbit
                 Orbit
             input_dop : isce3.core.LUT2d
                 Doppler LUT
             y0 : double
                 Starting northing position
             dy : double
                 Northing step size
             x0 : double
                 Starting easting position
             dx : double
                 Easting step size
             geogrid_length : int
                 Geographic length (number of pixels) in the northing direction
             geogrid_width : int
                 Geographic width (number of pixels) in the easting direction
             epsg : int
                 Output geographic grid EPSG
             input_terrain_radiometry : isce3.geometry.RtcInputTerrainRadiometry, optional
                 Terrain radiometry of the input raster
             output_terrain_radiometry : isce3.geometry.RtcOutputTerrainRadiometry, optional
                 Terrain radiometry of the input raster
             rtc_area_mode : isce3.geometry.RtcAreaMode, optional
                 RTC area mode
             rtc_algorithm : isce3.geometry.RtcAlgorithm, optional
                 RTC algorithm
             rtc_area_beta_mode : isce3.geometry.RtcAreaBetaMode, optional
                 RTC area beta mode
             geogrid_upsampling : double, optional
                 Geogrid upsampling (in each direction)
             rtc_min_value_db : float, optional
                 Minimum value for the RTC area factor. Radar
              data with RTC area factor below this limit are ignored.
             out_geo_rdr : isce3.io.Raster, optional
                 Raster to which the radar-grid positions
                 (range and azimuth) of the geogrid pixels vertices will be saved (output).
             out_geo_grid : isce3.io.Raster, optional
                 Raster to which the radar-grid positions
                 (range and azimuth) of the geogrid pixels center will be saved (output).
             out_sigma : isce3.io.Raster, optional
                 Output sigma surface area (rtc_area_mode = AREA) or area
                 factor (rtc_area_mode = AREA_FACTOR) raster
             rtc_memory_mode : isce3.core.MemoryModeBlocksY, optional
                 Select memory mode
             interp_method : isce3.core.DataInterpMethod, optional
                 Interpolation Method
             threshold : double, optional
                 Azimuth time threshold for convergence (s)
             num_iter : int, optional
                 Maximum number of Newton-Raphson iterations
             delta_range : double, optional
                 Step size used for computing Doppler derivative
             min_block_size : long long, optional
                Minimum block size
             max_block_size : long long, optional
                Maximum block size
             )");
}
