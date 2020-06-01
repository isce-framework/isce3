#pragma once

#include <isce/geometry/RTC.h>
#include <pybind11/pybind11.h>

using isce::geometry::rtcInputRadiometry;
using isce::geometry::rtcAlgorithm;

void addbinding(pybind11::enum_<rtcInputRadiometry> & pyInputRadiometry);
void addbinding(pybind11::enum_<rtcAlgorithm> & pyRtcAlgorithm);
