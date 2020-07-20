#include "LookSide.h"

using isce3::core::LookSide;

void addbinding(pybind11::enum_<isce3::core::LookSide> & pyLookSide)
{
    pyLookSide
        .value("Left", LookSide::Left)
        .value("Right", LookSide::Right);
}
