#pragma once

#include <isce3/signal/Covariance.h>
#include <pybind11/pybind11.h>

template<typename T>
void addbinding(pybind11::class_<isce::signal::Covariance<T>> &);
