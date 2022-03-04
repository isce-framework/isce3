#include "GeocodePolygon.h"

#include <isce3/geometry/RTC.h>
#include <isce3/io/Raster.h>

#include <pybind11/stl.h>

namespace py = pybind11;

using isce3::geocode::GeocodePolygon;
using isce3::geometry::rtcInputTerrainRadiometry;
using isce3::geometry::rtcOutputTerrainRadiometry;
using isce3::geometry::rtcInputTerrainRadiometry;

template<typename T>
void addbinding(py::class_<GeocodePolygon<T>> &pyGeocodePolygon)
{
    pyGeocodePolygon
        .def(py::init<
                   const std::vector<double> &,
                   const std::vector<double> &,
                   const isce3::product::RadarGridParameters &,
                   const isce3::core::Orbit &,
                   const isce3::core::Ellipsoid &,
                   const isce3::core::LUT2d<double> &,
                   isce3::io::Raster &, 
                   double,
                   int, 
                   double>(),
            py::arg("x_vect"),
            py::arg("y_vect"),
            py::arg("radar_grid"),
            py::arg("orbit"),
            py::arg("ellipsoid"),
            py::arg("input_dop"),
            py::arg("dem_raster"), 
            py::arg("threshold") = 1e-8,
            py::arg("num_iter") = 100, 
            py::arg("delta_range") = 1e-8,
            R"(
    Calculate the mean value of radar-grid samples using a polygon defined
    over geographical coordinates.
    Arguments:
        x_vect              Polygon vertices Lon/Easting positions
        y_vect              Polygon vertices Lon/Easting positions
        radar_grid          Radar grid
        orbit               Orbit
        input_dop           Doppler LUT associated with the radar grid
        dem_raster          Input DEM raster
        threshold           Azimuth time threshold for convergence (s)
        num_iter            Maximum number of Newton-Raphson iterations
        delta_range         Step size used for computing Doppler derivative
        )")

        .def_property_readonly("xoff", &GeocodePolygon<T>::xoff)
        .def_property_readonly("yoff", &GeocodePolygon<T>::yoff)
        .def_property_readonly("xsize", &GeocodePolygon<T>::xsize)
        .def_property_readonly("ysize", &GeocodePolygon<T>::ysize)
        .def_property_readonly("out_nlooks", &GeocodePolygon<T>::out_nlooks)
        .def("get_polygon_mean", &GeocodePolygon<T>::getPolygonMean,
            py::arg("radar_grid"),
            py::arg("input_dop"),
            py::arg("input_raster"),
            py::arg("output_raster"),
            py::arg("dem_raster"),
            py::arg("flag_apply_rtc") = false, 
            py::arg("input_terrain_radiometry") = rtcInputTerrainRadiometry::BETA_NAUGHT,
            py::arg("output_terrain_radiometry") =
                            rtcOutputTerrainRadiometry::GAMMA_NAUGHT,
            py::arg("exponent") = 0,
            py::arg("geogrid_upsampling") = 1,
            py::arg("rtc_min_value_db") = std::numeric_limits<float>::quiet_NaN(),
            py::arg("abs_cal_factor") = 1,
            py::arg("radargrid_nlooks") = 1,
            py::arg("output_off_diag_terms") = nullptr,
            py::arg("output_radargrid_data") = nullptr,
            py::arg("output_rtc") = nullptr,
            py::arg("output_weights") = nullptr,
            py::arg("interp_mode") = isce3::core::BIQUINTIC_METHOD,
            R"(
    Calculate the mean value of radar-grid samples using a polygon defined
    over geographical coordinates.
    Arguments:
        radar_grid          Radar grid
        input_dop           Doppler LUT associated with the radar grid
        input_raster        Input raster
        output_raster       Output raster (output)
        dem_raster          Input DEM raster
        flag_apply_rtc      Apply radiometric terrain correction (RTC)
        input_terrain_radiometry   Input terrain radiometry
        output_terrain_radiometry  Output terrain radiometry
        exponent            Exponent to be applied to the input data.
    The value 0 indicates that the the exponent is based on the data type of
    the input raster (1 for real and 2 for complex rasters).
        output_mode         Output mode
        geogrid_upsampling  Geogrid upsampling (in each direction)
        rtc_min_value_db    Minimum value for the RTC area factor.
    Radar data with RTC area factor below this limit are ignored.
        abs_cal_factor      Absolute calibration factor.
        radar_grid_nlooks   Radar grid number of looks. This
    parameters determines the multilooking factor used to compute out_nlooks.
        output_off_diag_terms Output raster containing the 
    off-diagonal terms of the covariance matrix (output)
        output_radargrid_data Radar-grid data multiplied by the
    weights that was used to compute the polygon average backscatter
    (output)
        output_rtc          Output RTC area factor (in slant-range)
    (output).
        output_weights      Polygon weights (level of intersection
    between the polygon with the radar grid) (output).
        interp_method       Data interpolation method
     )");
}

template void addbinding(py::class_<GeocodePolygon<float>> &);
template void addbinding(py::class_<GeocodePolygon<double>> &);
template void addbinding(py::class_<GeocodePolygon<std::complex<float>>> &);
template void addbinding(py::class_<GeocodePolygon<std::complex<double>>> &);

