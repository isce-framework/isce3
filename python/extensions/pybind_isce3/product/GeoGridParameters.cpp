#include "GeoGridParameters.h"

#include <isce3/core/Constants.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridParameters.h>

using isce3::product::GeoGridParameters;

namespace py = pybind11;

void addbinding(py::class_<GeoGridParameters> & pyGeoGridParams)
{
    pyGeoGridParams
            .def(py::init<double, double, double, double, int, int, int>(),
                 py::arg("start_x") = 0.0, py::arg("start_y") = 0.0,
                 py::arg("spacing_x") = 0.0, py::arg("spacing_y") = 0.0,
                 py::arg("width") = 0, py::arg("length") = 0,
                 py::arg("epsg") = 4326)
            .def("print", &GeoGridParameters::print)
            .def("__str__",
                 [](GeoGridParameters self) {
                     return isce3::product::to_string(self);
                 })
            .def_property(
                    "start_x",
                    py::overload_cast<>(&GeoGridParameters::startX, py::const_),
                    py::overload_cast<double>(&GeoGridParameters::startX))
            .def_property(
                    "start_y",
                    py::overload_cast<>(&GeoGridParameters::startY, py::const_),
                    py::overload_cast<double>(&GeoGridParameters::startY))
            .def_property_readonly("end_x", &GeoGridParameters::endX)
            .def_property_readonly("end_y", &GeoGridParameters::endY)
            .def_property(
                    "spacing_x",
                    py::overload_cast<>(&GeoGridParameters::spacingX,
                                        py::const_),
                    py::overload_cast<double>(&GeoGridParameters::spacingX))
            .def_property(
                    "spacing_y",
                    py::overload_cast<>(&GeoGridParameters::spacingY,
                                        py::const_),
                    py::overload_cast<double>(&GeoGridParameters::spacingY))
            .def_property(
                    "width",
                    py::overload_cast<>(&GeoGridParameters::width, py::const_),
                    py::overload_cast<int>(&GeoGridParameters::width))
            .def_property(
                    "length",
                    py::overload_cast<>(&GeoGridParameters::length, py::const_),
                    py::overload_cast<int>(&GeoGridParameters::length))
            .def_property(
                    "epsg",
                    py::overload_cast<>(&GeoGridParameters::epsg, py::const_),
                    py::overload_cast<int>(&GeoGridParameters::epsg));
}

void addbinding_bbox_to_geogrid(py::module & m)
{
    m.def("bbox_to_geogrid_scaled",
            &isce3::product::bbox2GeoGridScaled,
            py::arg("radar_grid"),
            py::arg("orbit"),
            py::arg("doppler"),
            py::arg("dem_raster"),
            py::arg("spacing_scale") = 1.0,
            py::arg("min_height") = isce3::core::GLOBAL_MIN_HEIGHT,
            py::arg("max_height") = isce3::core::GLOBAL_MAX_HEIGHT,
            py::arg("margin") = 0.0,
            py::arg("pts_per_edge") = 11,
            py::arg("threshold") = isce3::geometry::detail::DEFAULT_TOL_HEIGHT,
            py::arg("height_threshold") = 100, R"(
    Create a GeoGridParameters object by using spacing and ESPG from a DEM, and
    by estimating a bounding box with a radar grid. Spacing adjustable via scalar.

    Arguments:
        radar_grid          Input RadarGridParameters
        orbit               Input orbit
        doppler             Input doppler
        dem_raster          DEM from which EPSG and spacing is extracted
        spacing_scale       Scalar increase or decrease geogrid spacing
        min_height          Height lower bound
        max_height          Height upper bound
        margin              Amount to pad estimated bounding box. In decimal degrees.
        point_per_edge      Number of points to use on each side of radar grid.
        threshold           Height threshold (m) for rdr2geo convergence.
        height_threshold    Height threshold for convergence.
            )")
    .def("bbox_to_geogrid",
            &isce3::product::bbox2GeoGrid,
            py::arg("radar_grid"),
            py::arg("orbit"),
            py::arg("doppler"),
            py::arg("spacing_x"),
            py::arg("spacing_y"),
            py::arg("epsg"),
            py::arg("min_height") = isce3::core::GLOBAL_MIN_HEIGHT,
            py::arg("max_height") = isce3::core::GLOBAL_MAX_HEIGHT,
            py::arg("margin") = 0.0,
            py::arg("pts_per_edge") = 11,
            py::arg("threshold") = isce3::geometry::detail::DEFAULT_TOL_HEIGHT,
            py::arg("height_threshold") = 100, R"(
    Create a GeoGridParameters object by using spacing and ESPG from a DEM, and
    by estimating a bounding box with a radar grid. Spacing adjustable via scalar.

    Arguments:
        radar_grid          Input RadarGridParameters
        orbit               Input orbit
        doppler             Input doppler
        spacing_x           Geogrid spacing in X axis
        spacing_y           Geogrid spacing in Y axis
        epsg                EPSG code
        min_height          Height lower bound
        max_height          Height upper bound
        margin              Amount to pad estimated bounding box. In decimal degrees.
        point_per_edge      Number of points to use on each side of radar grid.
        threshold           Height threshold (m) for rdr2geo convergence.
        height_threshold    Height threshold for convergence.
        )");
}
