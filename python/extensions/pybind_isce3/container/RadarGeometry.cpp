#include "RadarGeometry.h"

namespace py = pybind11;

using isce3::container::RadarGeometry;

void addbinding(py::class_<RadarGeometry>& pyRadarGeometry)
{
    pyRadarGeometry
        // constructor(s)
        .def(py::init<const isce3::product::RadarGridParameters&,
                      const isce3::core::Orbit&,
                      const isce3::core::LUT2d<double>&>(),
                py::arg("radar_grid"),
                py::arg("orbit"),
                py::arg("doppler"))

        // member access
        .def_property_readonly("radar_grid", &RadarGeometry::radarGrid)
        .def_property_readonly("orbit", &RadarGeometry::orbit)
        .def_property_readonly("doppler", &RadarGeometry::doppler)
        .def_property_readonly("reference_epoch", &RadarGeometry::referenceEpoch)
        .def_property_readonly("grid_length", &RadarGeometry::gridLength)
        .def_property_readonly("grid_width", &RadarGeometry::gridWidth)
        .def_property_readonly("sensing_time", &RadarGeometry::sensingTime)
        .def_property_readonly("slant_range", &RadarGeometry::slantRange)
        .def_property_readonly("look_side", &RadarGeometry::lookSide)
        ;
}
