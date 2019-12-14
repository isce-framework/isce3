#pragma once

#include <pybind11/pybind11.h>

#include <isce/io/gdal/Dataset.h>

namespace py = pybind11;

namespace isce { namespace extension { namespace io { namespace gdal {

void addbinding(py::class_<isce::io::gdal::Dataset> &);

}}}}
