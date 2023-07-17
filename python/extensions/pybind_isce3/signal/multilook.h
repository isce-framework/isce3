#pragma once
#include <pybind11/pybind11.h>

template<typename EigenInputType, typename EigenWeightType>
void addbinding_multilook(pybind11::module& m);
