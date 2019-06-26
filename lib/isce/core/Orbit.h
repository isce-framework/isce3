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

// isce::core
#include "Cartesian.h"
#include "Constants.h"
#include "Utilities.h"
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
    /** \brief Reference epoch for the orbit object.
     *
     * Defaults to MIN_DATE_TIME. This value is used to reference DateTime tags
     * to double precision seconds. Ideally should be within a day of time tags*/
    DateTime refEpoch;

    /** Reformat the orbit and convert datetime to seconds since epoch*/
    void reformatOrbit(const DateTime &epoch);


    /** Only convert datetime to seconds since epoch*/
    void updateUTCTimes(const DateTime &epoch);

    /** If no epoch provided, use MIN_DATE_TIME as epoch*/
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
                            stateVectors(o.stateVectors), refEpoch(o.refEpoch) {
    }

    /** Comparison operator */
    inline bool operator==(const Orbit &o) const;

    /** Assignment operator*/
    inline Orbit& operator=(const Orbit &o);

    /** Increment operator*/
    inline Orbit& operator+=(const Orbit &o);

    /** Addition operator*/
    inline const Orbit operator+(const Orbit &o) const;

    /** Get state vector by index*/
    inline void getStateVector(int ind, double &t, cartesian_t &pos, cartesian_t &vel) const;

    /** Set state vector by index*/
    inline void setStateVector(int ind, double t, const cartesian_t &pos, const cartesian_t &vel);

    /** Adds a state vector to orbit*/
    inline void addStateVector(double t, const cartesian_t &pos, const cartesian_t &vel);

    /** Interpolate orbit using specified method.*/
    int interpolate(double tintp, StateVector &sv, orbitInterpMethod method) const;

    /** Interpolate orbit using specified method. */
    int interpolate(double tintp, cartesian_t &opos, cartesian_t &ovel, orbitInterpMethod intp_type) const;

    /** Interpolate orbit using Hermite polynomial.*/
    int interpolateWGS84Orbit(double tintp, cartesian_t &opos, cartesian_t &ovel) const;

    /** Interpolated orbit using Legendre polynomial.*/
    int interpolateLegendreOrbit(double tintp, cartesian_t &opos,cartesian_t &ovel) const;

    /** Interpolate orbit using Linear weights.*/
    int interpolateSCHOrbit(double tintp, cartesian_t &opos, cartesian_t &ovel) const;

    /** Compute acceleration numerically at given epoch.*/
    int computeAcceleration(double tintp, cartesian_t &acc) const;

    /** Compute Heading (clockwise w.r.t North) in radians at given epoch*/
    double getENUHeading(double aztime) const;

    /** Debug print function */
    void printOrbit() const;

    /** Utility function to load orbit from HDR file.
     *  Will update refEpoch if 2nd argument is true.
     */
    void loadFromHDR(const char *filename, bool update_epoch = true);

    /** Utility function to dump orbit to HDR file */
    void dumpToHDR(const char*) const;

};

// Comparison operator
bool isce::core::Orbit::
operator==(const Orbit & rhs) const {
    // Some easy checks first
    bool equal = nVectors == rhs.nVectors;
    equal *= refEpoch == rhs.refEpoch;
    if (!equal) {
        return false;
    }
    // If we pass the easy checks, check the orbit contents
    for (size_t i = 0; i < nVectors; ++i) {
        equal *= isce::core::compareFloatingPoint(UTCtime[i], rhs.UTCtime[i]);
        for (size_t j = 0; j < 3; ++j) {
            equal *= isce::core::compareFloatingPoint(position[3*i+j], rhs.position[3*i+j]);
            equal *= isce::core::compareFloatingPoint(velocity[3*i+j], rhs.velocity[3*i+j]);
        }
    }
    return equal;
}

isce::core::Orbit & isce::core::Orbit::
operator=(const Orbit &rhs) {
    nVectors = rhs.nVectors;
    UTCtime = rhs.UTCtime;
    epochs = rhs.epochs;
    position = rhs.position;
    velocity = rhs.velocity;
    stateVectors = rhs.stateVectors;
    refEpoch = rhs.refEpoch;
    return *this;
}

isce::core::Orbit & isce::core::Orbit::
operator+=(const Orbit &rhs) {
    cartesian_t t_pos, t_vel;
    for (int i = 0; i < rhs.nVectors; i++) {
        for (int j = 0; j < 3; ++j) {
            t_pos[j] = rhs.position[i*3+j];
            t_vel[j] = rhs.velocity[i*3+j];
        }
        addStateVector(rhs.UTCtime[i], t_pos, t_vel);
    }
    return *this;
}

const isce::core::Orbit isce::core::Orbit::
operator+(const Orbit &rhs) const {
    return (Orbit(*this) += rhs);
}




/** @param[in] ind Index of the state vector to return
 * @param[out] t Time since the reference epoch in seconds
 * @param[out] pos Position (m) of the state vector
 * @param[out] vel Velocity (m/s) of the state vector
 *
 * Returns values from linearized vectors and not stateVectors*/
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

/** @param[in] ind Index of the state vector to set
 * @param[in] t Time since reference epoch in seconds for state vector
 * @param[in] pos Position (m) of the state vector
 * @param[in] vel Velocity (m/s) of the state vector
 *
 * Sets values in linearized vectors and not stateVectors*/
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


/** @param[in] t Time since reference epoch in seconds for state vector
 * @param[in] pos Position (m) of the state vector
 * @param[in] vel Velocity (m/s) of the state vector
 *
 * Sets values in linearized vectors and not stateVectors. Index to insert is determined
 * using the "t" value. Internally, linearized vectors are sorted by time. */
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
