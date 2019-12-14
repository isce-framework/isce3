#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace isce { namespace extension { namespace io { namespace gdal {

void addsubmodule(py::module &);

}}}}
