#pragma once

#include <pybind11/pybind11.h>

#include <isce3/antenna/ElNullRangeEst.h>

// struct
void addbinding(pybind11::class_<isce3::antenna::NullProduct>&);
void addbinding(pybind11::class_<isce3::antenna::NullConvergenceFlags>&);
void addbinding(pybind11::class_<isce3::antenna::NullPowPatterns>&);
// class
void addbinding(pybind11::class_<isce3::antenna::ElNullRangeEst>&);
