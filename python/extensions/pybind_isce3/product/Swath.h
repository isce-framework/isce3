#pragma once

#include <isce/product/Swath.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce::product::Swath> &);
