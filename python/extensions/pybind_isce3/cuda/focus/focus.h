#pragma once

#include <pybind11/pybind11.h>

// XXX addsubmodule_focus would violate ODR
void addsubmodule_cuda_focus(pybind11::module&);
