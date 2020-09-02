#pragma once
#define EIGEN_MPL2_ONLY
#include "forward.h"

#include <vector>

#include <Eigen/Geometry>

#include "Attitude.h"
#include "DateTime.h"
#include "Quaternion.h"
#include "TimeDelta.h"

namespace isce3 { namespace core {

/** Store and interpolate attitude measurements. */
class Attitude {

public:
    /** Constructor
     *
     * @param[in] time          Time tags, seconds since some epoch.
     *                          Must be strictly increasing.
     * @param[in] quaternions   Unit quaternions representing antenna to XYZ
     *                          (ECEF) rotation.
     * @param[in] epoch         Reference epoch (UTC) for time tags.
     */
    Attitude(const std::vector<double>& time,
             const std::vector<Quaternion>& quaternions, const DateTime& epoch);

    Attitude() = default;

    /** Return quaternion interpolated at requested time. */
    Quaternion interpolate(double t) const;

    /** Return data vector of time */
    const std::vector<double>& time() const { return _time; }

    /** Return data vector of quaternions */
    const std::vector<Quaternion>& quaternions() const { return _quaternions; };

    /** Return number of epochs */
    int size() const { return _time.size(); }

    /** Get reference epoch (UTC) for time tags. */
    const DateTime& referenceEpoch() const { return _reference_epoch; }

    /** Set reference epoch (UTC)
     *
     * Updates contents of time() so that
     *      referenceEpoch() + TimeDelta(time()[i])
     * remains the invariant.
     */
    void referenceEpoch(const DateTime& epoch);

    /** Time of first measurement relative to reference epoch (s) */
    double startTime() const { return _time[0]; }

    /** Time of last measurement relative to reference epoch (s) */
    double endTime() const { return _time[size() - 1]; }

    /** UTC time of first measurement */
    DateTime startDateTime() const
    {
        return _reference_epoch + TimeDelta(startTime());
    }

    /** UTC time of last measurement */
    DateTime endDateTime() const
    {
        return _reference_epoch + TimeDelta(endTime());
    }

private:
    DateTime _reference_epoch;
    std::vector<double> _time;
    std::vector<Quaternion> _quaternions;
};

}} // namespace isce3::core
