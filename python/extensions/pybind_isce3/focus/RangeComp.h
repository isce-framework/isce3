#pragma once

#include <isce3/focus/RangeComp.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::enum_<isce3::focus::RangeComp::Mode>&);
void addbinding(pybind11::class_<isce3::focus::RangeComp>&);
