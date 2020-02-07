#pragma once

#include <complex>
#include <vector>

namespace isce { namespace focus {

/**
 * Construct a time-domain LFM chirp waveform
 *
 * \param[in] chirpslope    Signed frequency slope (Hz/s)
 * \param[in] duration      Chirp duration (s)
 * \param[in] samplerate    Sampling rate (Hz)
 * \param[in] centerfreq    Center frequency (Hz)
 * \param[in] amplitude     Amplitude
 * \param[in] phi           Phase offset at center of chirp (rad)
 * \returns                 Time-domain I/Q samples
 */
std::vector<std::complex<float>>
formLinearChirp(double chirprate,
                double duration,
                double samplerate,
                double centerfreq = 0.,
                double amplitude = 1.,
                double phi = 0.);

}}
