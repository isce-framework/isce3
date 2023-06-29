#include "Attitude.h"

#include <algorithm>
#include <exception>

#include <isce3/except/Error.h>
#include <pyre/journal.h>

#include "TimeDelta.h"

static bool isStrictlyIncreasing(const std::vector<double>& time)
{
    auto last = time[0];
    for (size_t i = 1; i < time.size(); ++i) {
        if (time[i] <= last) {
            return false;
        }
        last = time[i];
    }
    return true;
}

namespace isce3 { namespace core {

Attitude::Attitude(const std::vector<double>& time,
                   const std::vector<Quaternion>& quaternions,
                   const DateTime& epoch)
    : _reference_epoch(epoch), _time(time), _quaternions(quaternions)
{
    pyre::journal::error_t errorChannel("isce.core.Attitude");
    if (time.size() < 2) {
        errorChannel << pyre::journal::at(__HERE__)
                     << "Require at least two attitudes."
                     << pyre::journal::endl;
        // FIXME remove once pyre errors are fatal
        throw std::invalid_argument("Require at least two attitudes.");
    }
    if (time.size() != quaternions.size()) {
        errorChannel << pyre::journal::at(__HERE__)
                     << "Time and quaternion vector sizes don't match."
                     << pyre::journal::endl;
        throw std::invalid_argument("size mismatch");
    }
    if (not ::isStrictlyIncreasing(time)) {
        errorChannel << pyre::journal::at(__HERE__)
                     << "Time must be strictly increasing."
                     << pyre::journal::endl;
        throw std::invalid_argument("time must be strictly increasing");
    }
}

Quaternion Attitude::interpolate(double t) const
{
    // Check time bounds; error if out of bonds
    const int n = size();
    if (t < _time[0] || t > _time[n - 1]) {
        pyre::journal::error_t errorChannel("isce.core.Attitude");
        errorChannel << pyre::journal::at(__HERE__)
                     << "Requested out-of-bounds time." << pyre::journal::endl;
        throw std::domain_error("time out of bounds");
    }

    // Find interval containing desired point.
    // _time setter guarantees monotonic.
    // Offsets at start and end implement extrapolation w/o explicit logic.
    auto it = std::lower_bound(_time.begin() + 1, _time.end() - 1, t);
    auto i = it - _time.begin();

    // Slerp between the nearest data points.
    const double tq = (t - _time[i - 1]) / (_time[i] - _time[i - 1]);
    auto q0 = _quaternions[i - 1];
    auto q1 = _quaternions[i];
    return q0.slerp(tq, q1);
}

void Attitude::referenceEpoch(const DateTime& epoch)
{
    std::transform(_time.begin(), _time.end(), _time.begin(), [&](double t) {
        DateTime dateTime = _reference_epoch + TimeDelta(t);
        return (dateTime - epoch).getTotalSeconds();
    });
    _reference_epoch = epoch;
}

Attitude Attitude::crop(const DateTime& start, const DateTime& end, int npad) const
{
    const double tstart = (start - _reference_epoch).getTotalSeconds();
    const double tend = (end - _reference_epoch).getTotalSeconds();

    if (not this->contains(tstart)) {
        std::string errmsg = "Requested start time " + start.isoformat() +
                             " does not fall in attitude time interval [" +
                             startDateTime().isoformat() + ", " +
                             endDateTime().isoformat() + "].";
        throw isce3::except::DomainError(ISCE_SRCINFO(), errmsg);
    }
    if (not this->contains(tend)) {
        std::string errmsg = "Requested end time " + end.isoformat() +
                             " does not fall in attitude time interval [" +
                             startDateTime().isoformat() + ", " +
                             endDateTime().isoformat() + "].";
        throw isce3::except::DomainError(ISCE_SRCINFO(), errmsg);
    }
    if (tend < tstart) {
        std::string errmsg = "Expected start time <= end time";
        throw isce3::except::DomainError(ISCE_SRCINFO(), errmsg);
    }
    if (npad < 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "npad must be positive");
    }

    auto it = std::lower_bound(_time.begin(), _time.end(), tstart);
    auto istart = std::distance(_time.begin(), it) - npad - 1;
    it = std::upper_bound(it, _time.end(), tend);
    auto iend = std::distance(_time.begin(), it) + npad + 1;

    using T = decltype(istart);
    istart = std::max<T>(0, istart);
    iend = std::min<T>(_time.size(), iend);
    const auto n = iend - istart;

    std::vector<double> times(n);
    std::vector<Quaternion> quaternions(n);
    for (T i = 0; i < n; ++i) {
        const auto k = i + istart;
        times[i] = _time[k];
        quaternions[i] = _quaternions[k];
    }
    return Attitude(times, quaternions, _reference_epoch);
}

}} // namespace isce3::core
