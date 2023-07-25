#pragma once
#include <pybind11/pybind11.h>

template<typename EigenInputType>
void addbinding_multilook(pybind11::module& m);
