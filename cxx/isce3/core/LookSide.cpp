// -*- C++ -*-
// -*- coding: utf-8 -*-

#include "LookSide.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <string>

#include <isce3/except/Error.h>

using isce3::core::LookSide;

LookSide isce3::core::parseLookSide(const std::string & inputLook)
{
    // Convert to lowercase
    std::string look(inputLook);
    std::transform(look.begin(), look.end(), look.begin(),
        [](unsigned char c) { return std::tolower(c); });
    // Validate look string before setting
    if (look.compare(0, 5, "right") == 0) {
        return LookSide::Right;
    } else if (look.compare(0, 4, "left") != 0) {
        std::string msg = "Expected look side string in {'left', 'right'} "
            "but got '" + look + "'";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), msg);
    }
    return LookSide::Left;
}

std::string isce3::core::to_string(LookSide d)
{
    if (d == LookSide::Left) {
        return std::string("left");
    }
    assert(d == LookSide::Right);
    return std::string("right");
}

std::ostream & isce3::core::operator<<(std::ostream & out, const LookSide d)
{
    return out << isce3::core::to_string(d);
}
