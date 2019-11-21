#include "BuildOrbit.h"

#include <limits>
#include <stdexcept>
#include <string>

#include "../TimeDelta.h"

namespace isce { namespace core { namespace detail {

Linspace<double>
getOrbitTime(const std::vector<StateVector> & statevecs, const DateTime & reference_epoch)
{
    // convert size() to int, check for overflow
    constexpr static std::size_t max_int = std::numeric_limits<int>::max();
    if (statevecs.size() > max_int) {
        throw std::overflow_error("number of state vectors exceeds max orbit size");
    }
    int size = statevecs.size();

    // estimate state vector spacing (need at least two samples)
    if (size < 2) {
        throw std::invalid_argument("at least two state vectors are required");
    }
    TimeDelta spacing = statevecs[1].datetime - statevecs[0].datetime;

    // check that state vectors are uniformly sampled in time
    for (int i = 0; i < size-1; ++i) {
        DateTime t1 = statevecs[i].datetime;
        DateTime t2 = statevecs[i+1].datetime;
        if (!t2.isClose(t1 + spacing)) {
            std::string errmsg =
                "non-uniform spacing between state vectors encountered - "
                "interval between state vector at position " + std::to_string(i) + " "
                "and state vector at position " + std::to_string(i+1) + " "
                "is " + std::to_string( (t2 - t1).getTotalSeconds() ) + " s, "
                "expected " + std::to_string(spacing.getTotalSeconds()) + " s";
            throw std::invalid_argument(errmsg);
        }
    }

    // time of first state vector relative to reference epoch
    TimeDelta starttime = statevecs[0].datetime - reference_epoch;

    return {starttime.getTotalSeconds(), spacing.getTotalSeconds(), size};
}

std::vector<Vec3>
getOrbitPosition(const std::vector<StateVector> & statevecs)
{
    std::vector<Vec3> pos(statevecs.size());
    for (std::size_t i = 0; i < statevecs.size(); ++i) {
        pos[i] = statevecs[i].position;
    }
    return pos;
}

std::vector<Vec3>
getOrbitVelocity(const std::vector<StateVector> & statevecs)
{
    std::vector<Vec3> vel(statevecs.size());
    for (std::size_t i = 0; i < statevecs.size(); ++i) {
        vel[i] = statevecs[i].velocity;
    }
    return vel;
}

}}}
