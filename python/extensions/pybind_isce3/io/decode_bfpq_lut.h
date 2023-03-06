#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void addbinding_decode_bfpq_lut(py::module &m);
