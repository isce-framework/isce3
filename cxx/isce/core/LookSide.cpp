// -*- C++ -*-
// -*- coding: utf-8 -*-

#include "LookSide.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <pyre/journal.h>

using isce::core::LookSide;

LookSide isce::core::parseLookSide(const std::string & inputLook)
{
    // Convert to lowercase
    std::string look(inputLook);
    std::transform(look.begin(), look.end(), look.begin(),
        [](unsigned char c) { return std::tolower(c); });
    // Validate look string before setting
    if (look.compare("right") == 0) {
        return LookSide::Right;
    } else if (look.compare("left") != 0) {
        pyre::journal::error_t error("isce.core");
        error
            << pyre::journal::at(__HERE__)
            << "Could not successfully set look direction."
            << "  Must be \"right\" or \"left\"."
            << pyre::journal::endl;
    }
    return LookSide::Left;
}

std::string isce::core::to_string(LookSide d)
{
    if (d == LookSide::Left) {
        return std::string("left");
    }
    assert(d == LookSide::Right);
    return std::string("right");
}

std::ostream & isce::core::operator<<(std::ostream & out, const LookSide d)
{
    return out << isce::core::to_string(d);
}
