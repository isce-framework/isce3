#pragma once

#include <isce3/math/Stats.h>
#include <pybind11/pybind11.h>

template<typename T>
void addbinding_stats(pybind11::module& m, const char* name);

template<typename T>
void addbinding_stats_real_imag(pybind11::module& m);

template<typename T>
void addbinding(pybind11::class_<isce3::math::Stats<T>>&);

template<typename T>
void addbinding(pybind11::class_<isce3::math::StatsRealImag<T>>&);
