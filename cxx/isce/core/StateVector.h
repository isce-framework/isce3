#pragma once

#include "forward.h"

#include "DateTime.h"
#include "Vector.h"

namespace isce { namespace core {

struct StateVector {
    DateTime datetime;
    Vec3 position;
    Vec3 velocity;
};

inline
bool operator==(const StateVector & lhs, const StateVector & rhs)
{
    return lhs.datetime == rhs.datetime &&
           lhs.position == rhs.position &&
           lhs.velocity == rhs.velocity;
}

inline
bool operator!=(const StateVector & lhs, const StateVector & rhs)
{
    return !(lhs == rhs);
}

}}
