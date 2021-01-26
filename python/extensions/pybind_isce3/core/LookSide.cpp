#include "LookSide.h"

using isce3::core::LookSide;

void addbinding(pybind11::enum_<isce3::core::LookSide> & pyLookSide)
{
    pyLookSide
        .value("Left", LookSide::Left)
        .value("Right", LookSide::Right);
}

LookSide duck_look_side(pybind11::object pySide)
{
    if (pybind11::isinstance<pybind11::str>(pySide)) {
        auto s = pySide.cast<std::string>();
        return isce3::core::parseLookSide(s);
    }
    return pySide.cast<LookSide>();
}
