#pragma once

#include <isce3/unwrap/phass/Phass.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::unwrap::phass::Phass> &);
