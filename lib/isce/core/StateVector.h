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

}}

#endif

