#pragma once

#include <pybind11/pybind11.h>

#include <isce/io/gdal/Raster.h>

namespace py = pybind11;

namespace isce { namespace extension { namespace io { namespace gdal {

void addbinding(py::class_<isce::io::gdal::Raster> &);

py::buffer_info toBuffer(isce::io::gdal::Raster &);

isce::io::gdal::Raster toRaster(py::buffer);

}}}}
