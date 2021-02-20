#pragma once

#include <pybind11/pybind11.h>

#include <isce3/antenna/SphGridType.h>

void addbinding(pybind11::enum_<isce3::antenna::SphGridType>&);
