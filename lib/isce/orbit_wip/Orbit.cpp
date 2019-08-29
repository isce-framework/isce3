#include <isce/except/Error.h>

#include "Orbit.h"

namespace isce { namespace orbit_wip {

OrbitPoint & OrbitPoint::operator=(const StateVector & statevec)
{
    if (!statevec.datetime.isClose(datetime)) {
        std::string errmsg = "input state vector datetime (which is " +
                statevec.datetime.isoformat() + ") did not match datetime "
                "of the orbit point (which is " + datetime.isoformat() + ")";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    position = statevec.position;
    velocity = statevec.velocity;

    return *this;
}

OrbitPoint::OrbitPoint(const DateTime & datetime, Vec3 & position, Vec3 & velocity)
:
    datetime {datetime},
    position {position},
    velocity {velocity}
{}

bool operator==(const OrbitPoint & lhs, const StateVector & rhs)
{
    return lhs.datetime == rhs.datetime &&
           lhs.position == rhs.position &&
           lhs.velocity == rhs.velocity;
}

bool operator==(const StateVector & rhs, const OrbitPoint & lhs)
{
    return lhs.datetime == rhs.datetime &&
           lhs.position == rhs.position &&
           lhs.velocity == rhs.velocity;
}

bool operator!=(const OrbitPoint & lhs, const StateVector & rhs)
{
    return !(lhs == rhs);
}

bool operator!=(const StateVector & lhs, const OrbitPoint & rhs)
{
    return !(lhs == rhs);
}

Orbit Orbit::from_statevectors(const std::vector<StateVector> & statevecs)
{
    if (statevecs.size() < 2) {
        throw isce::except::InvalidArgument(ISCE_SRCINFO(),
                "at least 2 state vectors are required");
    }

    DateTime refepoch = statevecs[0].datetime;
    TimeDelta spacing = statevecs[1].datetime - statevecs[0].datetime;
    int size = statevecs.size();

    Orbit orbit {refepoch, spacing, size};

    for (int i = 0; i < size; ++i) {
        orbit[i] = statevecs[i];
    }

    return orbit;
}

void Orbit::set_statevectors(const std::vector<StateVector> & statevecs)
{
    if (statevecs.size() < 2) {
        throw isce::except::InvalidArgument(ISCE_SRCINFO(),
                "at least 2 state vectors are required");
    }

    TimeDelta spacing = statevecs[1].datetime - statevecs[0].datetime;
    int size = statevecs.size();
  
    //Resize the containers
    resize(size);
    _refepoch = statevecs[0].datetime;
    _time.spacing( spacing.getTotalSeconds() ); 


    for (int i = 0; i < size; ++i) {
        (*this)[i] = statevecs[i];
    }
}

Orbit::Orbit(const DateTime & refepoch, const TimeDelta & spacing, int size)
:
    _refepoch {refepoch},
    _time {0., spacing.getTotalSeconds(), size},
    _position (std::size_t(size)),
    _velocity (std::size_t(size))
{}

Orbit::operator OrbitView()
{
    return {_refepoch, _time, _position.data(), _velocity.data()};
}

void Orbit::push_back(const StateVector & statevec)
{
    DateTime expected = _refepoch + spacing() * size();
    if (!statevec.datetime.isClose(expected)) {
        std::string errmsg = "input state vector datetime (which is " +
                statevec.datetime.isoformat() + ") did not match expected "
                "datetime of the next orbit point (which is " +
                expected.isoformat() + ")";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    _time.resize(size() + 1);

    _position.push_back(statevec.position);
    _velocity.push_back(statevec.velocity);
}

void Orbit::resize(int size)
{
    _time.resize(size);
    _position.resize(std::size_t(size));
    _velocity.resize(std::size_t(size));
}

std::vector<StateVector> Orbit::to_statevectors() const
{
    std::vector<StateVector> statevecs;

    for (int i = 0; i < size(); ++i) {
        StateVector sv = operator[](i);
        statevecs.push_back(sv);
    }

    return statevecs;
}

OrbitView Orbit::subinterval(int start, int stop)
{
    if (start > stop) {
        std::string errmsg = "stop index (which is " + std::to_string(stop) +
                ") must be >= start index (which is " + std::to_string(start) + ")";
        throw isce::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }
    if (start >= size() || stop > size()) {
        throw isce::except::InvalidArgument(ISCE_SRCINFO(),
                "out of range start or stop index");
    }

    return { _refepoch + spacing() * start,
             _time.subinterval(start, stop),
             &_position[start],
             &_velocity[start] };
}

bool operator==(const Orbit & lhs, const Orbit & rhs)
{
    return lhs.refepoch() == rhs.refepoch() &&
           lhs.time() == rhs.time() &&
           lhs.position() == rhs.position() &&
           lhs.velocity() == rhs.velocity();
}

bool operator!=(const Orbit & lhs, const Orbit & rhs)
{
    return !(lhs == rhs);
}

}}

