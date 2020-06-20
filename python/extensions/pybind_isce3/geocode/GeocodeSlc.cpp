#include "GeocodeSlc.h"

#include <isce/core/Ellipsoid.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Orbit.h>
#include <isce/io/Raster.h>
#include <isce/product/RadarGridParameters.h>
#include <isce/product/GeoGridParameters.h>

#include <isce/geocode/geocodeSlc.h>

namespace py = pybind11;

void addbinding_geocodeslc(py::module & m)
{
    m.def("geocode_slc", &isce::geocode::geocodeSlc,
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
