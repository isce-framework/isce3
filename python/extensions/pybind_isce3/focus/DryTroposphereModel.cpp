#include "DryTroposphereModel.h"

using isce3::focus::DryTroposphereModel;

void addbinding(pybind11::enum_<DryTroposphereModel> & pyDryTropoModel)
{
    pyDryTropoModel
        .value("NoDelay", DryTroposphereModel::NoDelay)
        .value("TSX", DryTroposphereModel::TSX);
}
