#include "Geocode.h"

#include <isce3/core/Constants.h>
#include <isce3/geometry/RTC.h>
#include <isce3/io/Raster.h>

#include <limits>

namespace py = pybind11;

using isce3::geometry::Geocode;
using isce3::geometry::geocodeMemoryMode;
using isce3::geometry::geocodeOutputMode;
using isce3::geometry::rtcInputRadiometry;
using isce3::geometry::rtcAlgorithm;
using isce3::core::parseDataInterpMethod;
using isce3::product::RadarGridParameters;
using isce3::io::Raster;

template<typename T>
void addbinding(py::class_<Geocode<T>> &pyGeocode)
{
    pyGeocode
        .def(py::init<>())
        .def_property("orbit", nullptr, &Geocode<T>::orbit)
        .def_property("doppler", nullptr, &Geocode<T>::doppler)
        .def_property("ellipsoid", nullptr, &Geocode<T>::ellipsoid)
        .def_property("threshold_geo2rdr", nullptr, &Geocode<T>::thresholdGeo2rdr)
        .def_property("num_iter_geo2rdr", nullptr, &Geocode<T>::numiterGeo2rdr)
        .def_property("lines_per_block", nullptr, &Geocode<T>::linesPerBlock)
        .def_property("dem_block_margin", nullptr, &Geocode<T>::demBlockMargin)
        .def_property("radar_block_margin", nullptr, &Geocode<T>::radarBlockMargin)
        .def_property("interpolator",
                nullptr,
                [](Geocode<T> & self, std::string & method)
                {
                    // get interp method
                    auto interp_method = parseDataInterpMethod(method);

                    // set interp method
                    self.interpolator(interp_method);
                })
        .def_property_readonly("geogrid_start_x", &Geocode<T>::geoGridStartX)
        .def_property_readonly("geogrid_start_y", &Geocode<T>::geoGridStartY)
        .def_property_readonly("geogrid_spacing_x", &Geocode<T>::geoGridSpacingX)
        .def_property_readonly("geogrid_spacing_y", &Geocode<T>::geoGridSpacingY)
        .def_property_readonly("geogrid_width", &Geocode<T>::geoGridWidth)
        .def_property_readonly("geogrid_length", &Geocode<T>::geoGridLength)
        .def("update_geogrid", &Geocode<T>::updateGeoGrid,
            py::arg("radar_grid"),
            py::arg("dem_raster"))
        .def("geogrid", &Geocode<T>::geoGrid,
            py::arg("x_start"),
            py::arg("y_start"),
            py::arg("x_spacing"),
            py::arg("y_spacing"),
            py::arg("width"),
            py::arg("length"),
            py::arg("epsg"))
        .def("geocode", [](Geocode<T> & self, RadarGridParameters & radar_grid,
                    Raster & input_raster, Raster & output_raster,
                    Raster & dem_raster, geocodeOutputMode output_mode,
                    double geogrid_upsampling,
                    rtcInputRadiometry input_radiometry,
                    int exponent,
                    float rtc_min_val_db,
                    double rtc_upsampling,
                    rtcAlgorithm rtc_algorithm,
                    double abs_cal_factor,
                    float clip_min, float clip_max,
                    float min_nlooks, float radargrid_nlooks,
                    Raster * out_geo_vertices,
                    Raster * out_dem_vertices,
                    Raster * out_geo_nlooks,
                    Raster * out_geo_rtc,
                    Raster * input_rtc,
                    Raster * output_rtc,
                    geocodeMemoryMode mem_mode,
                    isce3::core::dataInterpMethod interp_mode
                    )
            {
                self.geocode(radar_grid,
                        input_raster,
                        output_raster,
                        dem_raster,
                        output_mode,
                        geogrid_upsampling,
                        input_radiometry,
                        exponent,
                        rtc_min_val_db,
                        rtc_upsampling,
                        rtc_algorithm,
                        abs_cal_factor,
                        clip_min, clip_max,
                        min_nlooks, radargrid_nlooks,
                        out_geo_vertices, out_dem_vertices,
                        out_geo_nlooks, out_geo_rtc,
                        input_rtc, output_rtc,
                        mem_mode, interp_mode
                        );
            },
            py::arg("radar_grid"),
            py::arg("input_raster"),
            py::arg("output_raster"),
            py::arg("dem_raster"),
            py::arg("output_mode") = geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT,
            py::arg("geogrid_upsampling") = 1,
            py::arg("input_radiometry") = rtcInputRadiometry::BETA_NAUGHT,
            py::arg("exponent") = 0,
            py::arg("rtc_min_val_db") = std::numeric_limits<float>::quiet_NaN(),
            py::arg("rtc_upsampling") = std::numeric_limits<double>::quiet_NaN(),
            py::arg("rtc_algorithm") = rtcAlgorithm::RTC_AREA_PROJECTION,
            py::arg("abs_cal_factor") = 1,
            py::arg("clip_min") = std::numeric_limits<float>::quiet_NaN(),
            py::arg("clip_max") = std::numeric_limits<float>::quiet_NaN(),
            py::arg("min_nlooks") = std::numeric_limits<float>::quiet_NaN(),
            py::arg("radargrid_nlooks") = 1,
            py::arg("out_geo_vertices") = nullptr,
            py::arg("out_dem_vertices") = nullptr,
            py::arg("out_geo_nlooks") = nullptr,
            py::arg("out_geo_rtc") = nullptr,
            py::arg("input_rtc") = nullptr,
            py::arg("output_rtc") = nullptr,
            py::arg("mem_mode") = geocodeMemoryMode::AUTO,
            py::arg("interp_mode") = isce3::core::BIQUINTIC_METHOD)
    ;
}

void addbinding(pybind11::enum_<geocodeOutputMode> & pyGeocodeMode)
{
    pyGeocodeMode
        .value("INTERP", geocodeOutputMode::INTERP)
        .value("AREA_PROJECTION", geocodeOutputMode::AREA_PROJECTION)
        .value("AREA_PROJECTION_GAMMA_NAUGHT", geocodeOutputMode::AREA_PROJECTION_GAMMA_NAUGHT)
        ;
};

void addbinding(pybind11::enum_<geocodeMemoryMode> & pyGeocodeMemoryMode)
{
    pyGeocodeMemoryMode
        .value("AUTO", geocodeMemoryMode::AUTO)
        .value("SINGLE_BLOCK", geocodeMemoryMode::SINGLE_BLOCK)
        .value("BLOCKS_GEOGRID", geocodeMemoryMode::BLOCKS_GEOGRID)
        .value("BLOCKS_GEOGRID_AND_RADARGRID", geocodeMemoryMode::BLOCKS_GEOGRID_AND_RADARGRID)
        ;
};

template void addbinding(py::class_<Geocode<float>> &);
template void addbinding(py::class_<Geocode<double>> &);
template void addbinding(py::class_<Geocode<std::complex<float>>> &);
template void addbinding(py::class_<Geocode<std::complex<double>>> &);

// end of file
