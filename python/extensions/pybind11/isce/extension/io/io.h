#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace isce { namespace extension { namespace io {

void addsubmodule(py::module &);

}}}
