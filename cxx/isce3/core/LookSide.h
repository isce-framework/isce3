// -*- C++ -*-
// -*- coding: utf-8 -*-

#pragma once

#include "forward.h"
#include <iostream>
#include <string>

namespace isce3 { namespace core {

/** Side that radar looks at, Left or Right. */
enum class LookSide {
    // NOTE choice of +-1 is deliberate and used for arithmetic. Do not change!
    Left = 1,   /**< Radar points to left/port side of vehicle. */
    Right = -1  /**< Radar points to right/starboard side of vehicle. */
};

/** Parse string (e.g., "left" or "right") to enum LookSide. */
LookSide parseLookSide(const std::string & str);

/** Convert enum LookSide to string ("left" or "right"). */
std::string to_string(LookSide d);

std::ostream & operator<<(std::ostream & out, const LookSide d);

}} // isce3::core
