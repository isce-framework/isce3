#pragma once

#include "forward.h"

#include <isce3/core/Common.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Vector.h>
#include <isce3/error/ErrorCode.h>

namespace isce { namespace cuda { namespace core {

/**
 * Non-owning reference to Orbit
 *
 * This class exports a device code interface for operating on orbit data.
 */
class OrbitView {
public:

    OrbitView() = default;

    /** Create a non-owning view of the specified orbit object */
    OrbitView(const Orbit &);

    /** Interpolation method */
    CUDA_DEV
    isce::core::OrbitInterpMethod interpMethod() const { return _interp_method; }

    /** Time of first state vector relative to reference epoch (s) */
    CUDA_DEV
    double startTime() const { return _time[0]; }

    /** Time of center of orbit relative to reference epoch (s) */
    CUDA_DEV
    double midTime() const { return startTime() + 0.5 * (size() - 1) * spacing(); }

    /** Time of last state vector relative to reference epoch (s) */
    CUDA_DEV
    double endTime() const { return _time[size()-1]; }

    /** Time interval between state vectors (s) */
    CUDA_DEV
    double spacing() const { return _time.spacing(); }

    /** Number of state vectors in orbit */
    CUDA_DEV
    int size() const { return _time.size(); }

    /** Get state vector times relative to reference epoch (s) */
    CUDA_DEV
    const isce::core::Linspace<double> & time() const { return _time; }

    /** Get state vector positions in ECEF coordinates (m) */
    CUDA_DEV
    const isce::core::Vec3 * position() const { return _position; }

    /** Get state vector velocities in ECEF coordinates (m/s) */
    CUDA_DEV
    const isce::core::Vec3 * velocity() const { return _velocity; }

    /** Get the specified state vector time relative to reference epoch (s) */
    CUDA_DEV
    double time(int idx) const { return _time[idx]; }

    /** Get the specified state vector position in ECEF coordinates (m) */
    CUDA_DEV
    const isce::core::Vec3 & position(int idx) const { return _position[idx]; }

    /** Get the specified state vector velocity in ECEF coordinates (m/s) */
    CUDA_DEV
    const isce::core::Vec3 & velocity(int idx) const { return _velocity[idx]; }

    /**
     * Interpolate platform position and/or velocity
     *
     * \param[out] position Interpolated position
     * \param[out] velocity Interpolated velocity
     * \param[in] t Interpolation time
     * \param[in] border_mode Mode for handling interpolation outside orbit domain
     * \return Error code indicating exit status
     */
    CUDA_DEV
    isce::error::ErrorCode
    interpolate(isce::core::Vec3 * position,
                isce::core::Vec3 * velocity,
                double t,
                isce::core::OrbitInterpBorderMode border_mode = isce::core::OrbitInterpBorderMode::Error) const;

private:
    const isce::core::Linspace<double> _time;
    const isce::core::Vec3 * _position;
    const isce::core::Vec3 * _velocity;
    const isce::core::OrbitInterpMethod _interp_method = isce::core::OrbitInterpMethod::Hermite;
};

}}}

#define ISCE_CUDA_CORE_ORBITVIEW_ICC
#include "OrbitView.icc"
#undef ISCE_CUDA_CORE_ORBITVIEW_ICC
