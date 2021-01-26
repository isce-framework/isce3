#pragma once

#include <pybind11/pybind11.h>

#include <isce3/signal/flatten.h>

void addbinding_flatten(pybind11::module& m);
