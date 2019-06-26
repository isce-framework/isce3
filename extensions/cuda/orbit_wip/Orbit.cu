#include <isce/except/Error.h>
#include <thrust/host_vector.h>

#include "Orbit.h"

namespace isce { namespace cuda { namespace orbit_wip {

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

OrbitPoint::OrbitPoint(const DateTime & datetime,
        thrust::device_reference<Vec3> position,
        thrust::device_reference<Vec3> velocity)
:
    datetime {datetime},
    position {position},
    velocity {velocity}
{}

bool operator==(const OrbitPoint & lhs, const StateVector & rhs)
{
    Vec3 lhs_pos = lhs.position;
    Vec3 lhs_vel = lhs.velocity;
    return lhs.datetime == rhs.datetime &&
           lhs_pos == rhs.position &&
           lhs_vel == rhs.velocity;
}

bool operator==(const StateVector & lhs, const OrbitPoint & rhs)
{
    Vec3 rhs_pos = rhs.position;
    Vec3 rhs_vel = rhs.velocity;
    return lhs.datetime == rhs.datetime &&
           lhs.position == rhs_pos &&
           lhs.velocity == rhs_vel;
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

    thrust::host_vector<Vec3> position_h (statevecs.size());
    thrust::host_vector<Vec3> velocity_h (statevecs.size());

    int size = statevecs.size();

    for (int i = 0; i < size; ++i) {
        position_h[i] = statevecs[i].position;
        velocity_h[i] = statevecs[i].velocity;
    }

    DateTime refepoch = statevecs[0].datetime;
    TimeDelta spacing = statevecs[1].datetime - statevecs[0].datetime;

    Orbit orbit {refepoch, spacing, size};

    orbit._position = position_h;
    orbit._velocity = velocity_h;

    return orbit;
}

Orbit::Orbit(const DateTime & refepoch, const TimeDelta & spacing, int size)
:
    _refepoch {refepoch},
    _time {0., spacing.getTotalSeconds(), size},
    _position (std::size_t(size)),
    _velocity (std::size_t(size))
{}

Orbit::Orbit(const isce::orbit_wip::Orbit & orbit)
:
    _refepoch {orbit.refepoch()},
    _time {orbit.time()},
    _position ( thrust::host_vector<Vec3>(orbit.position()) ),
    _velocity ( thrust::host_vector<Vec3>(orbit.velocity()) )
{}

Orbit & Orbit::operator=(const isce::orbit_wip::Orbit & orbit)
{
    _refepoch = orbit.refepoch();
    _time = orbit.time();
    _position = thrust::host_vector<Vec3>(orbit.position());
    _velocity = thrust::host_vector<Vec3>(orbit.velocity());
    return *this;
}

Orbit::operator OrbitView()
{
    return {_refepoch, _time, _position.data().get(), _velocity.data().get()};
}

Orbit::operator isce::orbit_wip::Orbit() const
{
    std::vector<StateVector> statevecs = to_statevectors();
    return isce::orbit_wip::Orbit::from_statevectors(statevecs);
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
    _position.resize(size);
    _velocity.resize(size);
}

std::vector<StateVector> Orbit::to_statevectors() const
{
    // copy device vectors to host
    thrust::host_vector<Vec3> position_h = _position;
    thrust::host_vector<Vec3> velocity_h = _velocity;

    std::vector<StateVector> statevecs;

    for (int i = 0; i < size(); ++i) {
        StateVector sv = {_refepoch + _time[i], position_h[i], velocity_h[i]};
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
             (&_position[start]).get(),
             (&_velocity[start]).get() };
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

}}}

