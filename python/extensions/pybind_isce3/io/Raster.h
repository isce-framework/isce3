#pragma once

#include <pybind11/pybind11.h>

#include <isce3/io/Raster.h>

namespace py = pybind11;

void addbinding(py::class_<isce3::io::Raster> &);
