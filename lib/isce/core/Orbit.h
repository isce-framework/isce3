//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_ORBIT_H
#define ISCE_CORE_ORBIT_H

#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Constants.h"
#include "StateVector.h"

// Declaration
namespace isce {
    namespace core {
        struct Orbit;
        void orbitHermite(const std::vector<cartesian_t>&,
                          const std::vector<cartesian_t>&,
                          const std::vector<double>&,double,cartesian_t&,cartesian_t&);
    }
}

/** Data structure to represent ECEF state vectors of imaging platform
 *
 *  There are two representations of the orbit information that are currently
 *  carried by this data structure. A vector of isce::core::StateVector for a 
 *  serializable representation and vectors of time-stamps, positions and 
 *  velocities to speed up computations. 
 *  All Time stamps are assumed to be UTC.
 *  All positions are in meters and in ECEF coordinates w.r.t WGS84 ellipsoid
 *  All velocities are in meters/sec and in ECEF coordinates w.r.t WGS84 Ellipsoid */
struct isce::core::Orbit {

    /** Number of state vectors */
    int nVectors;
    /** Linearized vector of isce::core::DateTime from contained State Vectors*/
    std::vector<DateTime> epochs;
    /** Linearized vecor of UTC times corresponding to State Vectors*/
    std::vector<double> UTCtime;
    /** Linearized position values of contained State Vectors*/
    std::vector<double> position;
    /** Linearized velocity values of contained State Vectors*/
    std::vector<double> velocity;
    /** Vector of isce::core::StateVector*/
    std::vector<StateVector> stateVectors;

    /** Reformat the orbit and convert datetime to seconds since epoch*/
    void reformatOrbit(const DateTime &);
    /** Only convert datetime to seconds since epoch*/
    void updateUTCTimes(const DateTime &);
    /** If no epoch provided, use minimum datetime as epoch*/
    void reformatOrbit();

    /** \brief Constructor number of state vectors
     *
     * @param[in] nv Number of state vectors*/
    Orbit(int nv) : nVectors(nv), epochs(nv,MIN_DATE_TIME),
                    UTCtime(nv,0.), position(3*nv,0.), velocity(3*nv,0.) {}

    /** Empty constructor*/
    Orbit() : Orbit(0) {}

    /** Copy constructor
     *
     * @param[in] o isce::core::Orbit object to copy*/
    Orbit(const Orbit &o) : nVectors(o.nVectors), epochs(o.epochs),
                            UTCtime(o.UTCtime), position(o.position), velocity(o.velocity),
                            stateVectors(o.stateVectors) {}

    /** Assignment operator*/
    inline Orbit& operator=(const Orbit &o);

    /** Increment operator*/
    inline Orbit& operator+=(const Orbit &o);

    /** Addition operator*/
    inline const Orbit operator+(const Orbit &o) const;

    /** Get state vector by index
     *
     * @param[in] ind Index of the state vector to return
     * @param[out] t Time since the reference epoch in seconds
     * @param[out] pos Position (m) of the state vector
     * @param[out] vel Velocity (m/s) of the state vector
     *
     * Returns values from linearized vectors and not stateVectors*/
    inline void getStateVector(int ind, double &t, cartesian_t &pos, cartesian_t &vel) const;

    /** Set state vector by index
     *
     * @param[in] ind Index of the state vector to set
     * @param[in] t Time since reference epoch in seconds for state vector
     * @param[in] pos Position (m) of the state vector
     * @param[in] vel Velocity (m/s) of the state vector
     *
     * Sets values in linearized vectors and not stateVectors*/
    inline void setStateVector(int ind, double t, const cartesian_t &pos, const cartesian_t &vel);

    /** Adds a state vector to orbit
     *
     * @param[in] t Time since reference epoch in seconds for state vector
     * @param[in] pos Position (m) of the state vector
     * @param[in] vel Velocity (m/s) of the state vector
     *
     * Sets values in linearized vectors and not stateVectors. Index to insert is determined 
     * using the "t" value. Internally, linearized vectors are sorted by time. */
    inline void addStateVector(double t, const cartesian_t &pos, const cartesian_t &vel);

    /** Interpolate orbit using specified method.
     *
     * @param[in] t Time since reference epoch in seconds
     * @param[out] sv StateVector object
     * @param[in] method Method to use for interpolation*/
    int interpolate(double t, StateVector &sv, orbitInterpMethod method) const;

    /** Interpolate orbit using specified method.
     *
     * @param[in] t Time since reference epoch in seconds
     * @param[out] pos Interpolated position (m)
     * @param[out] vel Interpolated velocity (m/s)
     * @param[in] method Method to use for interpolation
     *
     * Returns non-zero status on error*/
    int interpolate(double t, cartesian_t &pos, cartesian_t &vel, orbitInterpMethod method) const;

    /** Interpolate orbit using Hermite polynomial.
     *
     * @param[in] t Time since reference epoch in seconds
     * @param[out] pos Interpolated position (m)
     * @param[out] vel Interpolated position (m/s)
     *
     * Returns non-zero status on error*/
    int interpolateWGS84Orbit(double t, cartesian_t &pos, cartesian_t &vel) const;

    /** Interpolated orbit using Legendre polynomial.
     *
     * @param[in] t Time since reference epoch in seconds
     * @param[out] pos Interpolated position (m)
     * @param[out] vel Interpolated position (m/s)
     *
     * Returns non-zero status on error*/
    int interpolateLegendreOrbit(double, cartesian_t &,cartesian_t &) const;

    /** Interpolate orbit using Linear weights.
     *
     * @param[in] t Time since reference epoch in seconds
     * @param[out] pos Interpolated position (m)
     * @param[out] vel Interpolated velocity (m/s)
     *
     * Returns non-zero status on error*/
    int interpolateSCHOrbit(double t, cartesian_t &pos, cartesian_t &vel) const;
    
    /** Compute acceleration numerically at given epoch.
     *
     * @param[in] t Time since reference epoch in seconds
     * @param[out] acc Acceleration in m/s^2
     *
     * An interval of +/- 0.01 seconds is used for numerical computations*/ 
    int computeAcceleration(double t, cartesian_t &acc) const;

    /** Compute Heading (clockwise w.r.t North) in degrees at given epoch
     *
     * @param[in] t Time since reference epoch in seconds*/
    double getENUHeading(double t) const;

    /** Debug print function */
    void printOrbit() const;

    /** Utility function to load orbit from HDR file */
    void loadFromHDR(const char*);

    /** Utility function to dump orbit to HDR file */
    void dumpToHDR(const char*) const;

    /** \brief Reference epoch for the orbit object.
     *
     * Defaults to min DateTime. This value is used to reference DateTime tags
     * to double precision seconds. Ideally should be within a day of time tags*/
    DateTime refEpoch;
};

isce::core::Orbit & isce::core::Orbit::
operator=(const Orbit &rhs) {
    nVectors = rhs.nVectors;
    UTCtime = rhs.UTCtime;
    epochs = rhs.epochs;
    position = rhs.position;
    velocity = rhs.velocity;
    stateVectors = rhs.stateVectors;
    return *this;
}

isce::core::Orbit & isce::core::Orbit::
operator+=(const Orbit &rhs) {
    cartesian_t t_pos, t_vel;
    for (int i = 0; i < rhs.nVectors; i++) {
        for (int j = 0; j < 3; ++j) {
            t_pos[j] = rhs.position[i*3+j];
            t_vel[j] = rhs.position[i*3+j];
        }
        addStateVector(rhs.UTCtime[i], t_pos, t_vel);
    }
    return *this;
}

const isce::core::Orbit isce::core::Orbit::
operator+(const Orbit &rhs) const {
    return (Orbit(*this) += rhs);
}

void isce::core::Orbit::
getStateVector(int idx, double &t, cartesian_t &pos, cartesian_t &vel) const {
    if ((idx < 0) || (idx >= nVectors)) {
        std::string errstr = "Orbit::getStateVector - Trying to access vector " +
                             std::to_string(idx+1) + " out of " + std::to_string(nVectors) +
                             " possible vectors";
        throw std::out_of_range(errstr);
    }
    t = UTCtime[idx];
    for (int i=0; i<3; i++) {
        pos[i] = position[(3*idx)+i];
        vel[i] = velocity[(3*idx)+i];
    }
}

void isce::core::Orbit::
setStateVector(int idx, double t, const cartesian_t & pos, const cartesian_t & vel) {
    if ((idx < 0) || (idx >= nVectors)) {
        std::string errstr = "Orbit::setStateVector - Trying to access vector " + 
                             std::to_string(idx+1) + " out of " + std::to_string(nVectors) +
                             " possible vectors";
        throw std::out_of_range(errstr);
    }
    UTCtime[idx] = t;
    epochs[idx] = refEpoch + t;
    for (int i=0; i<3; i++) {
        position[3*idx+i] = pos[i];
        velocity[3*idx+i] = vel[i];
    }
}

void isce::core::Orbit::
addStateVector(double t, const cartesian_t & pos, const cartesian_t & vel) {
    int vec_idx = 0;
    while ((vec_idx < nVectors) && (t > UTCtime[vec_idx])) vec_idx++;
    UTCtime.insert(UTCtime.begin()+vec_idx, t);
    epochs.insert(epochs.begin()+vec_idx, refEpoch + t);
    position.insert(position.begin()+(3*vec_idx), pos.begin(), pos.end());
    velocity.insert(velocity.begin()+(3*vec_idx), vel.begin(), vel.end());
    nVectors++;
}

#endif

// end of file
