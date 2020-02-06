#include "Chirp.h"

#include <cmath>
#include <limits>

#include <isce/except/Error.h>

namespace isce { namespace focus {

std::vector<std::complex<float>>
formLinearChirp(double chirprate,
                double duration,
                double samplerate,
                double centerfreq,
                double amplitude,
                double phi)
{
    // sanity checks
    if (duration <= 0.) {
        throw isce::except::DomainError(ISCE_SRCINFO(), "chirp duration must be > 0");
    }
    if (samplerate <= 0.) {
        throw isce::except::DomainError(ISCE_SRCINFO(), "sampling rate must be > 0");
    }
    if (amplitude <= 0.) {
        throw isce::except::DomainError(ISCE_SRCINFO(), "amplitude must be > 0");
    }

    // check for possible overflow before double -> int conversion
    double d_size = samplerate * duration;
    double d_maxsize = std::numeric_limits<int>::max();
    if (d_size > d_maxsize) {
        throw isce::except::OverflowError(ISCE_SRCINFO(), "chirp size exceeds max int value");
    }

    // number of samples (rounded to nearest odd integer)
    int size = std::floor(d_size);
    if (size % 2 == 0) { size++; }

    double spacing = 1. / samplerate;
    double startfreq = centerfreq - 0.5 * chirprate * duration;

    // evaluate time-domain LFM chirp samples
    std::vector<std::complex<float>> chirp(size);
    for (int i = 0; i < size; ++i) {
        double tau = spacing * i;
        double phase = phi + 2. * M_PI * (startfreq + 0.5 * chirprate * tau) * tau;

        chirp[i] = std::polar(amplitude, phase);
    }

    return chirp;
}

}}
