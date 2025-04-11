#pragma once

#include <pybind11/pybind11.h>

void addbinding_block_processing(pybind11::module&);

void addbinding_get_block_processing_parameters(pybind11::module& m);
void addbinding_get_block_processing_parameters_y(pybind11::module& m);
