#pragma once

#include <isce3/core/Orbit.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::enum_<isce3::core::OrbitInterpMethod> & pyOrbitInterpMethod);
void addbinding(pybind11::class_<isce3::core::Orbit> &);
