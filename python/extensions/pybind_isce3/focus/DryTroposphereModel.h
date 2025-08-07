#pragma once

#include <isce3/focus/DryTroposphereModel.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::enum_<isce3::focus::DryTroposphereModel>&);
void addbinding_tsx_delay(pybind11::module& m);
