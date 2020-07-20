#pragma once

#include <isce3/focus/DryTroposphereModel.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::enum_<isce::focus::DryTroposphereModel>&);
