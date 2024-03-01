#include "Orbit.h"

#include <isce3/error/ErrorCode.h>
#include <isce3/except/Error.h>

#include "detail/BuildOrbit.h"
#include "detail/InterpolateOrbit.h"

using isce3::error::ErrorCode;
using isce3::error::getErrorString;

namespace isce3 { namespace core {

Orbit::Orbit(const std::vector<StateVector> & statevecs,
             OrbitInterpMethod interp_method)
:
    Orbit(statevecs, statevecs.at(0).datetime, interp_method)
{}

Orbit::Orbit(const std::vector<StateVector> & statevecs,
             const DateTime & reference_epoch,
             OrbitInterpMethod interp_method)
:
    _reference_epoch(reference_epoch),
    _time(detail::getOrbitTime(statevecs, reference_epoch)),
    _position(detail::getOrbitPosition(statevecs)),
    _velocity(detail::getOrbitVelocity(statevecs)),
    _interp_method(interp_method)
{}

std::vector<StateVector> Orbit::getStateVectors() const
{
    std::vector<StateVector> statevecs(size());
    for (int i = 0; i < size(); ++i) {
        statevecs[i].datetime = _reference_epoch + TimeDelta(_time[i]);
        statevecs[i].position = _position[i];
        statevecs[i].velocity = _velocity[i];
    }
    return statevecs;
}

void Orbit::setStateVectors(const std::vector<StateVector> & statevecs)
{
    _time = detail::getOrbitTime(statevecs, _reference_epoch);
    _position = detail::getOrbitPosition(statevecs);
    _velocity = detail::getOrbitVelocity(statevecs);
}

void Orbit::referenceEpoch(const DateTime & reference_epoch)
{
    DateTime old_refepoch = _reference_epoch;
    double old_starttime = _time.first();

    double dt = (old_refepoch - reference_epoch).getTotalSeconds();
    double starttime = old_starttime + dt;

    _time.first(starttime);
    _reference_epoch = reference_epoch;
}

ErrorCode Orbit::interpolate(Vec3* position, Vec3* velocity, double t,
                                  OrbitInterpBorderMode border_mode) const {
    // interpolate
    ErrorCode status =
            detail::interpolateOrbit(position, velocity, *this, t, border_mode);

    // check for errors
    if (status != ErrorCode::Success and
            border_mode == OrbitInterpBorderMode::Error) {

        std::string errmsg = getErrorString(status);
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }

    return status;
}

bool operator==(const Orbit & lhs, const Orbit & rhs)
{
    return lhs.referenceEpoch() == rhs.referenceEpoch() &&
           lhs.time() == rhs.time() &&
           lhs.position() == rhs.position() &&
           lhs.velocity() == rhs.velocity() &&
           lhs.interpMethod() == rhs.interpMethod();
}

bool operator!=(const Orbit & lhs, const Orbit & rhs)
{
    return !(lhs == rhs);
}

Orbit Orbit::crop(const DateTime& start, const DateTime& end, int npad) const
{
    const double tstart = (start - _reference_epoch).getTotalSeconds();
    const double tend = (end - _reference_epoch).getTotalSeconds();

    if (not this->contains(tstart)) {
        std::string errmsg = "Requested start time " + start.isoformat() +
                             " does not fall in orbit time interval [" +
                             startDateTime().isoformat() + ", " +
                             endDateTime().isoformat() + "].";
        throw isce3::except::DomainError(ISCE_SRCINFO(), errmsg);
    }
    if (not this->contains(tend)) {
        std::string errmsg = "Requested end time " + end.isoformat() +
                             " does not fall in orbit time interval [" +
                             startDateTime().isoformat() + ", " +
                             endDateTime().isoformat() + "].";
        throw isce3::except::DomainError(ISCE_SRCINFO(), errmsg);
    }
    if (npad < 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "npad must be positive");
    }

    const int istart = std::max(_time.search(tstart) - 1 - npad, 0);
    const int iend = std::min(_time.search(tend) + 1 + npad, _time.size());
    const int n = iend - istart;

    std::vector<StateVector> statevecs(n);
    for (int i = 0; i < n; ++i) {
        const int k = i + istart;
        const DateTime t = _reference_epoch + TimeDelta(_time[k]);
        statevecs[i] = {t, _position[k], _velocity[k]};
    }
    return Orbit(statevecs, _reference_epoch, _interp_method);
}

}}
