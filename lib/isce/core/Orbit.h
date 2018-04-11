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

// Orbit declaration
struct isce::core::Orbit {

    // Should be deprecated (unused)
    int basis;
    // Number of State Vectors
    int nVectors;
    // Linearized UTC time values of contained State Vectors
    std::vector<double> UTCtime;
    // Linearized position values of contained State Vectors
    std::vector<double> position;
    // Linearized velocity values of contained State Vectors
    std::vector<double> velocity;
    // Vector of StateVectors
    std::vector<StateVector> stateVectors;

    Orbit(int bs, int nv) : basis(bs), nVectors(nv), UTCtime(nv,0.), position(3*nv,0.), 
                            velocity(3*nv,0.) {}
    Orbit() : Orbit(0,0) {}
    Orbit(const Orbit &o) : basis(o.basis), nVectors(o.nVectors), UTCtime(o.UTCtime), 
                            position(o.position), velocity(o.velocity) {}
    inline Orbit& operator=(const Orbit&);
    inline Orbit& operator+=(const Orbit&);
    inline const Orbit operator+(const Orbit&) const;
    void getPositionVelocity(double,cartesian_t&,cartesian_t&) const;
    inline void getStateVector(int,double&,cartesian_t&,cartesian_t&) const;
    inline void setStateVector(int,double,const cartesian_t&,
                               const cartesian_t&);
    inline void addStateVector(double,const cartesian_t &,const cartesian_t &);
    int interpolate(double,cartesian_t&,cartesian_t&,orbitInterpMethod) const;
    int interpolateWGS84Orbit(double,cartesian_t&,cartesian_t&) const;
    int interpolateLegendreOrbit(double,cartesian_t&,cartesian_t&) const;
    int interpolateSCHOrbit(double,cartesian_t&,cartesian_t&) const;
    int computeAcceleration(double,cartesian_t&) const;
    double getENUHeading(double);
    void printOrbit() const;
    void loadFromHDR(const char*,int);
    void dumpToHDR(const char*) const;
};

isce::core::Orbit & isce::core::Orbit::
operator=(const Orbit &rhs) {
    basis = rhs.basis;
    nVectors = rhs.nVectors;
    UTCtime = rhs.UTCtime;
    position = rhs.position;
    velocity = rhs.velocity;
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
    position.insert(position.begin()+(3*vec_idx), pos.begin(), pos.end());
    velocity.insert(velocity.begin()+(3*vec_idx), vel.begin(), vel.end());
    nVectors++;
}

#endif

// end of file
