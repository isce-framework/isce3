// cost functions for edge method used in EL pointing
#pragma once

#include <pybind11/pybind11.h>
#include <isce3/antenna/EdgeMethodCostFunc.h>

void addbinding_edge_method_cost_func(pybind11::module& m);
