#include "LookSide.h"

using isce::core::LookSide;

void addbinding(pybind11::enum_<isce::core::LookSide> & pyLookSide)
{
    pyLookSide
        .value("Left", LookSide::Left)
        .value("Right", LookSide::Right);
}
