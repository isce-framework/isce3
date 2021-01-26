#include "GeocodeCov.h"

#include <limits>

#include <isce3/core/Constants.h>
#include <isce3/geometry/RTC.h>
#include <isce3/io/Raster.h>

namespace py = pybind11;

using isce3::core::parseDataInterpMethod;
using isce3::geocode::Geocode;
using isce3::geocode::geocodeMemoryMode;
using isce3::geocode::geocodeOutputMode;
using isce3::geometry::rtcAlgorithm;
using isce3::geometry::rtcInputTerrainRadiometry;
using isce3::io::Raster;
using isce3::product::RadarGridParameters;

template<typename T>
void addbinding(py::class_<Geocode<T>>& pyGeocode)
{
    pyGeocode.def(py::init<>())
            .def_property("orbit", nullptr, &Geocode<T>::orbit)
            .def_property("doppler", nullptr, &Geocode<T>::doppler)
            .def_property("ellipsoid", nullptr, &Geocode<T>::ellipsoid)
            .def_property("threshold_geo2rdr", nullptr,
                          &Geocode<T>::thresholdGeo2rdr)
            .def_property("numiter_geo2rdr", nullptr,
                          &Geocode<T>::numiterGeo2rdr)
            .def_property("lines_per_block", nullptr,
                          &Geocode<T>::linesPerBlock)
            .def_property("dem_block_margin", nullptr,
                          &Geocode<T>::demBlockMargin)
            .def_property("radar_block_margin", nullptr,
                          &Geocode<T>::radarBlockMargin)
            .def_property("interpolator", nullptr,
                          [](Geocode<T>& self, std::string& method) {
                              // get interp method
                              auto interp_method =
                                      parseDataInterpMethod(method);

                              // set interp method
                              self.interpolator(interp_method);
                          })
            .def_property_readonly("geogrid_start_x",
                                   &Geocode<T>::geoGridStartX)
            .def_property_readonly("geogrid_start_y",
                                   &Geocode<T>::geoGridStartY)
            .def_property_readonly("geogrid_spacing_x",
                                   &Geocode<T>::geoGridSpacingX)
            .def_property_readonly("geogrid_spacing_y",
                                   &Geocode<T>::geoGridSpacingY)
            .def_property_readonly("geogrid_width", &Geocode<T>::geoGridWidth)
            .def_property_readonly("geogrid_length", &Geocode<T>::geoGridLength)
            .def("update_geogrid", &Geocode<T>::updateGeoGrid,
                 py::arg("radar_grid"), py::arg("dem_raster"))
            .def("geogrid", &Geocode<T>::geoGrid, py::arg("x_start"),
                 py::arg("y_start"), py::arg("x_spacing"), py::arg("y_spacing"),
                 py::arg("width"), py::arg("length"), py::arg("epsg"))
            .def("geocode", &Geocode<T>::geocode,
                    py::arg("radar_grid"), py::arg("input_raster"),
                    py::arg("output_raster"), py::arg("dem_raster"),
                    py::arg("output_mode") =
                            geocodeOutputMode::AREA_PROJECTION_WITH_RTC,
                    py::arg("geogrid_upsampling") = 1,
                    py::arg("flag_upsample_radar_grid") = false,
                    py::arg("input_terrain_radiometry") =
                            rtcInputTerrainRadiometry::BETA_NAUGHT,
                    py::arg("exponent") = 0,
                    py::arg("rtc_min_value_db") =
                            std::numeric_limits<float>::quiet_NaN(),
                    py::arg("rtc_upsampling") =
                            std::numeric_limits<double>::quiet_NaN(),
                    py::arg("rtc_algorithm") =
                            rtcAlgorithm::RTC_AREA_PROJECTION,
                    py::arg("abs_cal_factor") = 1,
                    py::arg("clip_min") =
                            std::numeric_limits<float>::quiet_NaN(),
                    py::arg("clip_max") =
                            std::numeric_limits<float>::quiet_NaN(),
                    py::arg("min_nlooks") =
                            std::numeric_limits<float>::quiet_NaN(),
                    py::arg("radargrid_nlooks") = 1,
                    py::arg("out_off_diag_terms") = nullptr,
                    py::arg("out_geo_rdr") = nullptr,
                    py::arg("out_geo_dem") = nullptr,
                    py::arg("out_geo_nlooks") = nullptr,
                    py::arg("out_geo_rtc") = nullptr,
                    py::arg("input_rtc") = nullptr,
                    py::arg("output_rtc") = nullptr,
                    py::arg("memory_mode") = geocodeMemoryMode::AUTO,
                    py::arg("min_block_size") =
                            isce3::geometry::AP_DEFAULT_MIN_BLOCK_SIZE,
                    py::arg("max_block_size") =
                            isce3::geometry::AP_DEFAULT_MAX_BLOCK_SIZE,
                    py::arg("interp_mode") = isce3::core::BIQUINTIC_METHOD);
}

void addbinding(pybind11::enum_<geocodeOutputMode>& pyGeocodeMode)
{
    pyGeocodeMode.value("INTERP", geocodeOutputMode::INTERP)
            .value("AREA_PROJECTION", geocodeOutputMode::AREA_PROJECTION)
            .value("AREA_PROJECTION_WITH_RTC",
                   geocodeOutputMode::AREA_PROJECTION_WITH_RTC);
};

void addbinding(pybind11::enum_<geocodeMemoryMode>& pyGeocodeMemoryMode)
{
    pyGeocodeMemoryMode.value("AUTO", geocodeMemoryMode::AUTO)
            .value("SINGLE_BLOCK", geocodeMemoryMode::SINGLE_BLOCK)
            .value("BLOCKS_GEOGRID", geocodeMemoryMode::BLOCKS_GEOGRID)
            .value("BLOCKS_GEOGRID_AND_RADARGRID",
                   geocodeMemoryMode::BLOCKS_GEOGRID_AND_RADARGRID);
};

template void addbinding(py::class_<Geocode<float>>&);
template void addbinding(py::class_<Geocode<double>>&);
template void addbinding(py::class_<Geocode<std::complex<float>>>&);
template void addbinding(py::class_<Geocode<std::complex<double>>>&);

// end of file
