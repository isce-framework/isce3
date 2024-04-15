#pragma once

#include <isce3/error/ErrorCode.h>
#include <vector>
#include <string>

#include "DateTime.h"
#include "Linspace.h"
#include "StateVector.h"
#include "TimeDelta.h"
#include "Vector.h"

namespace isce3 { namespace core {

/**
 * Orbit interpolation method
 *
 * See <a href="overview_geometry.html#orbitinterp">the documentation</a>
 * for a description of each method.
 */
enum class OrbitInterpMethod {
    Hermite,  /**< Third-order Hermite polynomial interpolation */
    Legendre, /**< Eighth-order Legendre polynomial interpolation */
};

/** Mode determining how interpolation outside of orbit domain is handled */
enum class OrbitInterpBorderMode {
    Error,       /**< Raise error if interpolation attempted outside orbit domain */
    Extrapolate, /**< Allow extrapolation to points outside orbit domain */
    FillNaN,     /**< Output NaN for interpolation points outside orbit domain */
};

/**
 * Sequence of platform ephemeris samples (state vectors) with uniform temporal
 * spacing, supporting efficient lookup and interpolation
 *
 * DateTimes are expected to be UTC. Positions & velocities are in meters and
 * meters per second respectively and in ECEF coordinates w.r.t WGS84 ellipsoid.
 *
 * Platform position and velocity are stored at timepoints relative to a
 * reference epoch. For best accuracy, reference epoch should be within 24
 * hours of orbit time tags.
 */
class Orbit {
public:

    Orbit() = default;

    /**
     * Construct from list of state vectors
     *
     * Reference epoch defaults to time of first state vector
     *
     * \param[in] statevecs State vectors
     * \param[in] interp_method Interpolation method
     * \param[in] type Orbit ephemeris precision type
     */
    Orbit(const std::vector<StateVector> & statevecs,
          OrbitInterpMethod interp_method = OrbitInterpMethod::Hermite,
          const std::string& type = "");

    /**
     * Construct from list of state vectors and orbit type
     *
     * Reference epoch defaults to time of first state vector
     *
     * \param[in] statevecs State vectors
     * \param[in] type Orbit ephemeris precision type
     */
    Orbit(const std::vector<StateVector> & statevecs,
          const std::string& type);

    /**
     * Construct from list of state vectors and reference epoch
     *
     * \param[in] statevecs State vectors
     * \param[in] reference_epoch Reference epoch
     * \param[in] interp_method Interpolation method
     * \param[in] orbit_type Orbit ephemeris precision type
     */
    Orbit(const std::vector<StateVector> & statevecs,
          const DateTime & reference_epoch,
          OrbitInterpMethod interp_method = OrbitInterpMethod::Hermite,
          const std::string& type = "");

    /** Create a new Orbit containing data in the requested interval
     *
     * \param[in] start Beginning of time interval
     * \param[in] end   End of time interval
     * \param[in] npad  Minimal number of state vectors to include past each of
     *                  the given time bounds (useful to guarantee adequate
     *                  support for interpolation).
     * \returns Orbit object with data containing start & end times.  The
     *          reference epoch and interpolation method are preserved.
     */
    Orbit crop(const DateTime& start, const DateTime& end, int npad = 0) const;

    /** Export list of state vectors */
    std::vector<StateVector> getStateVectors() const;

    /** Set orbit state vectors */
    void setStateVectors(const std::vector<StateVector> &);

    /** Reference epoch (UTC) */
    const DateTime & referenceEpoch() const { return _reference_epoch; }

    /** Set reference epoch (UTC)
     *
     * Also updates relative time tags so that
     *      referenceEpoch() + TimeDelta(time()[i])
     * results in the same absolute time tags before and after this call.
     */
    void referenceEpoch(const DateTime &);

    /** Interpolation method */
    OrbitInterpMethod interpMethod() const { return _interp_method; }

    /** Set interpolation method */
    void interpMethod(OrbitInterpMethod interp_method) { _interp_method = interp_method; }

    /** Orbit ephemeris precision type */
    const std::string& type() const { return _type; }

    /** Set the orbit ephemeris precision type */
    void type(const std::string& orbit_type) { _type = orbit_type; }

    /** Time of first state vector relative to reference epoch (s) */
    double startTime() const { return _time[0]; }

    /** Time of center of orbit relative to reference epoch (s) */
    double midTime() const { return startTime() + 0.5 * (size() - 1) * spacing(); }

    /** Time of last state vector relative to reference epoch (s) */
    double endTime() const { return _time[size()-1]; }

    /** UTC time of first state vector */
    DateTime startDateTime() const { return _reference_epoch + TimeDelta(startTime()); }

    /** UTC time of center of orbit */
    DateTime midDateTime() const { return _reference_epoch + TimeDelta(midTime()); }

    /** UTC time of last state vector */
    DateTime endDateTime() const { return _reference_epoch + TimeDelta(endTime()); }

    /** Check if time falls in the valid interpolation domain. */
    bool contains(double time) const {
        return (startTime() <= time) && (time <= endTime());
    }

    /** Time interval between state vectors (s) */
    double spacing() const { return _time.spacing(); }

    /** Number of state vectors in orbit */
    int size() const { return _time.size(); }

    /** Get state vector times relative to reference epoch (s) */
    const Linspace<double> & time() const { return _time; }

    /** Get state vector positions in ECEF coordinates (m) */
    const std::vector<Vec3> & position() const { return _position; }

    /** Get state vector velocities in ECEF coordinates (m/s) */
    const std::vector<Vec3> & velocity() const { return _velocity; }

    /** Get the specified state vector time relative to reference epoch (s) */
    double time(int idx) const { return _time[idx]; }

    /** Get the specified state vector position in ECEF coordinates (m) */
    const Vec3 & position(int idx) const { return _position[idx]; }

    /** Get the specified state vector velocity in ECEF coordinates (m/s) */
    const Vec3 & velocity(int idx) const { return _velocity[idx]; }

    /**
     * Interpolate platform position and/or velocity
     *
     * If either \p position or \p velocity is a null pointer, that output will
     * not be computed. This may improve runtime by avoiding unnecessary
     * operations.
     *
     * \param[out] position Interpolated position
     * \param[out] velocity Interpolated velocity
     * \param[in] t Interpolation time
     * \param[in] border_mode Mode for handling interpolation outside orbit
     * domain
     * \return Error code indicating exit status
     */
    isce3::error::ErrorCode
    interpolate(Vec3* position, Vec3* velocity, double t,
                   OrbitInterpBorderMode border_mode =
                           OrbitInterpBorderMode::Error) const;

private:
    DateTime _reference_epoch;
    Linspace<double> _time;
    std::vector<Vec3> _position;
    std::vector<Vec3> _velocity;
    OrbitInterpMethod _interp_method = OrbitInterpMethod::Hermite;
    std::string _type = "";
};

bool operator==(const Orbit &, const Orbit &);
bool operator!=(const Orbit &, const Orbit &);

/**
 * Get minimum number of orbit state vectors required for interpolation with
 * specified method
 */
constexpr
int minStateVecs(OrbitInterpMethod method)
{
    switch (method) {
        case OrbitInterpMethod::Hermite  : return 4;
        case OrbitInterpMethod::Legendre : return 9;
    }

    return -1;
}

}}
