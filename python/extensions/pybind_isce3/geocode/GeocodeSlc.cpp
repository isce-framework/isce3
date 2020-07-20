#include "GeocodeSlc.h"

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridParameters.h>
#include <isce3/product/GeoGridParameters.h>

#include <isce3/geocode/geocodeSlc.h>

namespace py = pybind11;

void addbinding_geocodeslc(py::module & m)
{
    m.def("geocode_slc", &isce3::geocode::geocodeSlc,
        py::arg("output_raster"),
        py::arg("input_raster"),
        py::arg("dem_raster"),
        py::arg("radargrid"),
        py::arg("geogrid"),
        py::arg("orbit"),
        py::arg("native_doppler"),
        py::arg("image_grid_doppler"),
        py::arg("ellipsoid"),
        py::arg("threshold_geo2rdr") = 1.0e-9,
        py::arg("numiter_geo2rdr") = 25,
        py::arg("lines_per_block") = 1000,
        py::arg("dem_block_margin") = 0.1,
        py::arg("flatten") = true);
}
