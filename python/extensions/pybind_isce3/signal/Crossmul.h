#pragma once

#include <isce3/signal/Crossmul.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::signal::Crossmul> &);
