#pragma once

#include <pybind11/pybind11.h>

#include <isce3/antenna/Frame.h>

void addbinding(pybind11::class_<isce3::antenna::Frame>&);
