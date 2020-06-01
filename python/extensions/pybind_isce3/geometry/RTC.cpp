#include "RTC.h"

namespace py = pybind11;

using isce::geometry::rtcInputRadiometry;
using isce::geometry::rtcAlgorithm;

void addbinding(py::enum_<rtcInputRadiometry> & pyInputRadiometry)
{
    pyInputRadiometry
        .value("BETA_NAUGHT", rtcInputRadiometry::BETA_NAUGHT)
        .value("SIGMA_NAUGHT_ELLIPSOID", rtcInputRadiometry::SIGMA_NAUGHT_ELLIPSOID)
        ;
}

void addbinding(py::enum_<rtcAlgorithm> & pyAlgorithm)
{
    pyAlgorithm
        .value("RTC_DAVID_SMALL", rtcAlgorithm::RTC_DAVID_SMALL)
        .value("RTC_AREA_PROJECTION", rtcAlgorithm::RTC_AREA_PROJECTION)
        ;
}
