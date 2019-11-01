#pragma once

#include <vector>

#include "../DateTime.h"
#include "../Linspace.h"
#include "../StateVector.h"
#include "../Vector.h"

namespace isce { namespace core { namespace detail {

Linspace<double>
getOrbitTime(const std::vector<StateVector> & statevecs, const DateTime & reference_epoch);

std::vector<Vec3>
getOrbitPosition(const std::vector<StateVector> & statevecs);

std::vector<Vec3>
getOrbitVelocity(const std::vector<StateVector> & statevecs);

}}}
