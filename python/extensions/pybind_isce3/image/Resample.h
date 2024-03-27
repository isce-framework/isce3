#pragma once
#include <pybind11/pybind11.h>

void addbindings_resamp(pybind11::module&);

template<typename T>
void addbindings_modulate(pybind11::module&);
