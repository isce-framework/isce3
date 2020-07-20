#pragma once

#include <utility>
#include <vector>

namespace isce3 { namespace focus {

/** Determine location of blind ranges in SweepSAR systems. */
class GapMask {
public:
    /** Constructor
     *
     * @param[in] azimuth_time          Transmit time of each pulse (seconds
     *                                  relative to an arbitrary epoch).
     * @param[in] samples               Range samples
     * @param[in] range_window_start    Delay between TX and RX (s)
     * @param[in] range_sampling_rate   Sample rate (Hz)
     * @param[in] chirp_duration        Length of TX pulse (s)
     * @param[in] guard                 Additional guard band to blank around
     *                                  pulse (s)
     */
    GapMask(const std::vector<double> & azimuth_time, int samples,
            double range_window_start, double range_sampling_rate,
            double chirp_duration, double guard = 0.0);

    /** Compute gap locations for a given pulse.
     *
     * @param[in] pulse Index of desired range line
     * @returns List of [start, stop) range indices blocked by transmit events.
     */
    std::vector<std::pair<int, int>>
    gaps(int pulse) const;

    /** Compute gap locations for a given pulse.
     *
     * @param[in] pulse Index of desired range line
     * @returns Gap mask, true for samples blocked by transmit events.
     */
    std::vector<bool>
    mask(int pulse) const;

private:
    std::vector<double> t;
    int n;
    double dwp;
    double fs;
    double chirplen;
    double guard;
};

}} // namespace isce3::focus
