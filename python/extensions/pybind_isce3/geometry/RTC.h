#pragma once

#include <isce3/geometry/RTC.h>
#include <pybind11/pybind11.h>

using isce3::geometry::rtcInputRadiometry;
using isce3::geometry::rtcAlgorithm;

void addbinding(pybind11::enum_<rtcInputRadiometry> & pyInputRadiometry);
void addbinding(pybind11::enum_<rtcAlgorithm> & pyRtcAlgorithm);
