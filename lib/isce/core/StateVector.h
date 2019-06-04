#pragma once
#ifndef ISCE_CORE_STATEVECTOR_H
#define ISCE_CORE_STATEVECTOR_H

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

#endif

