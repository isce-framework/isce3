#include "GapMask.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <isce/except/Error.h>

namespace isce { namespace focus {

GapMask::GapMask(const std::vector<double> & azimuth_time, int samples,
    double range_window_start, double range_sampling_rate,
    double chirp_duration, double guard)
:
    t(azimuth_time),
    n(samples),
    dwp(range_window_start),
    fs(range_sampling_rate),
    chirplen(chirp_duration),
    guard(guard)
{
    if (t.size() > INT_MAX) {
        throw isce::except::InvalidArgument(ISCE_SRCINFO(),
            "require azimuth_time.size() <= INT_MAX");
    }
    for (int i = 1; i < t.size(); ++i) {
        if (t[i - 1] > t[i]) {
            std::string errmsg = "azimuth time must be monotonically increasing";
            throw isce::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
        }
    }
    using isce::except::DomainError;
    if (n <= 0) {
        throw DomainError(ISCE_SRCINFO(), "require range samples > 0");
    }
    if (dwp <= 0.) {
        throw DomainError(ISCE_SRCINFO(), "range window start time must be > 0");
    }
    if (fs <= 0.) {
        throw DomainError(ISCE_SRCINFO(), "range sampling rate must be > 0");
    }
    if (chirplen <= 0.) {
        throw DomainError(ISCE_SRCINFO(), "require chirp duration > 0");
    }
    if (guard < 0.) {
        throw DomainError(ISCE_SRCINFO(), "require guard band >= 0");
    }
}

std::vector<std::pair<int, int>>
GapMask::gaps(int pulse) const
{
    if ((pulse < 0) || (pulse >= t.size())) {
        throw isce::except::DomainError(ISCE_SRCINFO(), "pulse out of bounds");
    }
    std::vector<std::pair<int, int>> g;
    const double t0 = t[pulse] + dwp;
    const double t1 = t0 + n / fs;
    // Loop over pulses in the air.
    for (int i = pulse; i < t.size(); ++i) {
        // Done when pulses occur after end of RX window.
        // Typically 10-20 pulses for NISAR.
        if ((t[i] - guard) > t1) {
            break;
        }
        int j0 = static_cast<int>(std::lround((t[i] - t0 - guard) * fs));
        int j1 = static_cast<int>(std::lround(
            (t[i] + chirplen + guard - t0) * fs));
        // TX[i] overlaps RX[pulse]
        if ((j0 <= n) && (j1 >= 0)) {
            j0 = std::max(0, j0);
            j1 = std::min(n, j1);
            g.push_back(std::make_pair(j0, j1));
        }
    }
    return g;
}

std::vector<bool>
GapMask::mask(int pulse) const
{
    // convert pairs of [start, stop) intervals to boolean mask.
    std::vector<bool> mask(n, false);
    for (const auto & gap : gaps(pulse)) {
        for (auto i = gap.first; i < gap.second; ++i) {
            mask[i] = true;
        }
    }
    return mask;
}

}} // namespace isce::focus
