#pragma once
#ifndef ISCE_ORBITWIP_ORBIT_H
#define ISCE_ORBITWIP_ORBIT_H

#include <isce/core/Common.h>
#include <isce/core/DateTime.h>
#include <isce/core/Linspace.h>
#include <isce/core/StateVector.h>
#include <isce/core/TimeDelta.h>
#include <isce/core/Vector.h>
#include <vector>

#include "InterpOrbit.h"

namespace isce { namespace orbit_wip {

using isce::core::DateTime;
using isce::core::TimeDelta;
using isce::core::Linspace;
using isce::core::StateVector;
using isce::core::Vec3;

// Proxy for StateVector in an Orbit
struct OrbitPoint {
    OrbitPoint & operator=(const StateVector &);

    operator StateVector() const { return {datetime, position, velocity}; }

    const DateTime datetime;
    Vec3 & position;
    Vec3 & velocity;

private:
    OrbitPoint(const DateTime & datetime, Vec3 & position, Vec3 & velocity);

    friend class Orbit;
    friend class OrbitView;
};

bool operator==(const OrbitPoint &, const StateVector &);
bool operator==(const StateVector &, const OrbitPoint &);
bool operator!=(const OrbitPoint &, const StateVector &);
bool operator!=(const StateVector &, const OrbitPoint &);

/** Non-owning reference to Orbit or sub-interval of an Orbit. */
class OrbitView {
public:
    OrbitView() = default;

    CUDA_HOSTDEV
    OrbitView(const DateTime & refepoch,
              const Linspace<double> & time,
              Vec3 * position,
              Vec3 * velocity)
    :
        _refepoch {refepoch},
        _time {time},
        _position {position},
        _velocity {velocity}
    {}

    CUDA_HOSTDEV
    OrbitPoint operator[](int idx)
    {
        return {_refepoch + _time[idx], _position[idx], _velocity[idx]};
    }

    CUDA_HOSTDEV
    StateVector operator[](int idx) const
    {
        return {_refepoch + _time[idx], _position[idx], _velocity[idx]};
    }

    /** Return datetime of the first state vector. */
    CUDA_HOSTDEV
    const DateTime & refepoch() const { return _refepoch; }

    /** Return time interval between state vectors. */
    CUDA_HOSTDEV
    double spacing() const { return _time.spacing(); }

    /** Return number of state vectors. */
    CUDA_HOSTDEV
    int size() const { return _time.size(); }

    /** Return sequence of timepoints relative to reference epoch. */
    CUDA_HOSTDEV
    const Linspace<double> & time() const { return _time; }

    /** Return sequence of platform positions at each timepoint. */
    CUDA_HOSTDEV
    const Vec3 * position() const { return _position; }

    /** Return sequence of platform velocities at each timepoint. */
    CUDA_HOSTDEV
    const Vec3 * velocity() const { return _velocity; }

    /** Check if there are no state vectors in the sequence. */
    CUDA_HOSTDEV
    bool empty() const { return size() == 0; }

private:
    DateTime _refepoch;
    Linspace<double> _time;
    Vec3 * _position;
    Vec3 * _velocity;
};

/**
 * Container for a set of platform position and velocity state vectors
 * sampled uniformly in time.
 */
class Orbit {
public:
    /** Create Orbit from vector of uniformly spaced StateVectors. */
    static
    Orbit from_statevectors(const std::vector<StateVector> &);

    Orbit() = default;

    /**
     * Constructor
     *
     * @param[in] refepoch datetime of the initial state vector
     * @param[in] spacing time interval between state vectors
     * @param[in] size number of state vectors
     */
    Orbit(const DateTime & refepoch, const TimeDelta & spacing, int size = 0);

    /** Return a non-owning view of the Orbit. */
    operator OrbitView();

    OrbitPoint operator[](int idx)
    {
        return {_refepoch + _time[idx], _position[idx], _velocity[idx]};
    }

    StateVector operator[](int idx) const
    {
        return {_refepoch + _time[idx], _position[idx], _velocity[idx]};
    }

    /** Return datetime of the first state vector. */
    const DateTime & refepoch() const { return _refepoch; }

    /** Return time interval between state vectors. */
    TimeDelta spacing() const { return _time.spacing(); }

    /** Return number of state vectors. */
    int size() const { return _time.size(); }

    /** Return sequence of timepoints relative to reference epoch. */
    const Linspace<double> & time() const { return _time; }

    /** Return sequence of platform positions at each timepoint. */
    const std::vector<Vec3> & position() const { return _position; }

    /** Return sequence of platform velocities at each timepoint. */
    const std::vector<Vec3> & velocity() const { return _velocity; }

    /** Append a new state vector. */
    void push_back(const StateVector &);

    /** Resize the container. */
    void resize(int size);

    /** Check if there are no state vectors in the sequence. */
    bool empty() const { return size() == 0; }

    /** Convert to vector of StateVectors. */
    std::vector<StateVector> to_statevectors() const;

    /**
     * Return a view of a subspan of the Orbit over the half-open
     * interval [start, stop).
     *
     * @param[in] start start index
     * @param[in] stop end index (not included in interval)
     */
    OrbitView subinterval(int start, int stop);

private:
    DateTime _refepoch;
    Linspace<double> _time;
    std::vector<Vec3> _position;
    std::vector<Vec3> _velocity;
};

bool operator==(const Orbit &, const Orbit &);
bool operator!=(const Orbit &, const Orbit &);

}}

#endif

