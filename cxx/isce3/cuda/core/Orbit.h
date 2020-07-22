#pragma once

#include <thrust/device_vector.h>
#include <vector>

#include <isce3/core/DateTime.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/StateVector.h>
#include <isce3/core/TimeDelta.h>
#include <isce3/core/Vector.h>

namespace isce3 { namespace cuda { namespace core {

/**
 * CUDA counterpart of isce3::core::Orbit
 *
 * This class exports a host-side interface for managing and interacting with
 * orbit data residing in device memory. It is not intended for use in device
 * code - use OrbitView instead.
 */
class Orbit {
public:

    Orbit() = default;

    /** Construct from isce3::core::Orbit (copy from host to device) */
    Orbit(const isce3::core::Orbit &);

    /**
     * Construct from list of state vectors
     *
     * Reference epoch defaults to time of first state vector
     *
     * \param[in] statevecs State vectors
     * \param[in] interp_method Interpolation method
     */
    Orbit(const std::vector<isce3::core::StateVector> & statevecs,
          isce3::core::OrbitInterpMethod interp_method = isce3::core::OrbitInterpMethod::Hermite);

    /**
     * Construct from list of state vectors and reference epoch
     *
     * \param[in] statevecs State vectors
     * \param[in] reference_epoch Reference epoch
     * \param[in] interp_method Interpolation method
     */
    Orbit(const std::vector<isce3::core::StateVector> & statevecs,
          const isce3::core::DateTime & reference_epoch,
          isce3::core::OrbitInterpMethod interp_method = isce3::core::OrbitInterpMethod::Hermite);

    /** Export list of state vectors */
    std::vector<isce3::core::StateVector> getStateVectors() const;

    /** Set orbit state vectors */
    void setStateVectors(const std::vector<isce3::core::StateVector> &);

    /** Reference epoch (UTC) */
    const isce3::core::DateTime & referenceEpoch() const { return _reference_epoch; }

    /** Set reference epoch (UTC) */
    void referenceEpoch(const isce3::core::DateTime &);

    /** Interpolation method */
    isce3::core::OrbitInterpMethod interpMethod() const { return _interp_method; }

    /** Set interpolation method */
    void interpMethod(isce3::core::OrbitInterpMethod interp_method) { _interp_method = interp_method; }

    /** Time of first state vector relative to reference epoch (s) */
    double startTime() const { return _time[0]; }

    /** Time of center of orbit relative to reference epoch (s) */
    double midTime() const { return startTime() + 0.5 * (size() - 1) * spacing(); }

    /** Time of last state vector relative to reference epoch (s) */
    double endTime() const { return _time[size()-1]; }

    /** UTC time of first state vector */
    isce3::core::DateTime startDateTime() const { return _reference_epoch + isce3::core::TimeDelta(startTime()); }

    /** UTC time of center of orbit */
    isce3::core::DateTime midDateTime() const { return _reference_epoch + isce3::core::TimeDelta(midTime()); }

    /** UTC time of last state vector */
    isce3::core::DateTime endDateTime() const { return _reference_epoch + isce3::core::TimeDelta(endTime()); }

    /** Time interval between state vectors (s) */
    double spacing() const { return _time.spacing(); }

    /** Number of state vectors in orbit */
    int size() const { return _time.size(); }

    /** Get state vector times relative to reference epoch (s) */
    const isce3::core::Linspace<double> & time() const { return _time; }

    /** Get state vector positions in ECEF coordinates (m) */
    const thrust::device_vector<isce3::core::Vec3> & position() const { return _position; }

    /** Get state vector velocities in ECEF coordinates (m/s) */
    const thrust::device_vector<isce3::core::Vec3> & velocity() const { return _velocity; }

    /** Get the specified state vector time relative to reference epoch (s) */
    double time(int idx) const { return _time[idx]; }

    /** Get the specified state vector position in ECEF coordinates (m) */
    isce3::core::Vec3 position(int idx) const { return _position[idx]; }

    /** Get the specified state vector velocity in ECEF coordinates (m/s) */
    isce3::core::Vec3 velocity(int idx) const { return _velocity[idx]; }

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
     * \param[in] border_mode Mode for handling interpolation outside orbit domain
     * \return Error code indicating exit status
     */
    isce3::error::ErrorCode
    interpolate(isce3::core::Vec3* position, isce3::core::Vec3* velocity,
                double t,
                isce3::core::OrbitInterpBorderMode border_mode =
                        isce3::core::OrbitInterpBorderMode::Error) const;

private:
    isce3::core::DateTime _reference_epoch;
    isce3::core::Linspace<double> _time;
    thrust::device_vector<isce3::core::Vec3> _position;
    thrust::device_vector<isce3::core::Vec3> _velocity;
    isce3::core::OrbitInterpMethod _interp_method = isce3::core::OrbitInterpMethod::Hermite;
};

bool operator==(const Orbit &, const Orbit &);
bool operator!=(const Orbit &, const Orbit &);

}}}
