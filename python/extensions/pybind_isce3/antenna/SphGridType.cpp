#include "SphGridType.h"

namespace py = pybind11;
using isce3::antenna::SphGridType;

void addbinding(py::enum_<SphGridType>& pySphGridType)
{
    pySphGridType.value("THETA_PHI", SphGridType::THETA_PHI)
            .value("EL_AND_AZ", SphGridType::EL_AND_AZ)
            .value("EL_OVER_AZ", SphGridType::EL_OVER_AZ)
            .value("AZ_OVER_EL", SphGridType::AZ_OVER_EL);
}
