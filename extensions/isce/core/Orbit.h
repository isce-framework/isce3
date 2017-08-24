//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_ORBIT_H__
#define __ISCE_CORE_ORBIT_H__

#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "isce/core/Constants.h"

namespace isce { namespace core {
    struct Orbit {
        int basis;                      // Needs to be deprecated if continued to be unused
        int nVectors;                   // Number of State Vectors
        std::vector<double> UTCtime;    // Linearized UTC time values of contained State Vectors
        std::vector<double> position;   // Linearized position values of contained State Vectors
        std::vector<double> velocity;   // Linearized velocity values of contained State Vectors

        Orbit(int bs, int nv) : basis(bs), nVectors(nv), UTCtime(nv,0.), position(3*nv,0.), velocity(3*nv,0.) {}                        // Value constructor
        Orbit() : Orbit(0,0) {}                                                                                                         // Default constructor (delegated)
        Orbit(const Orbit &o) : basis(o.basis), nVectors(o.nVectors), UTCtime(o.UTCtime), position(o.position), velocity(o.velocity) {} // Copy constructor
        inline Orbit& operator=(const Orbit&);
        inline Orbit& operator+=(const Orbit&);
        inline const Orbit operator+(const Orbit&) const;
        void getPositionVelocity(double,std::vector<double>&,std::vector<double>&);
        inline void getStateVector(int,double&,std::vector<double>&,std::vector<double>&);
        inline void setStateVector(int,double,std::vector<double>&,std::vector<double>&);
        inline void addStateVector(double,std::vector<double>&,std::vector<double>&);
        int interpolate(double,std::vector<double>&,std::vector<double>&,orbitInterpMethod);
        int interpolateWGS84Orbit(double,std::vector<double>&,std::vector<double>&);
        int interpolateLegendreOrbit(double,std::vector<double>&,std::vector<double>&);
        int interpolateSCHOrbit(double,std::vector<double>&,std::vector<double>&);
        int computeAcceleration(double,std::vector<double>&);
        void printOrbit();
        void loadFromHDR(const char*,int);
        void dumpToHDR(const char*);
    };

    void orbitHermite(std::vector<std::vector<double>>&,std::vector<std::vector<double>>&,std::vector<double>&,double,std::vector<double>&,std::vector<double>&);

    inline Orbit& Orbit::operator=(const Orbit &rhs) {
        basis = rhs.basis;
        nVectors = rhs.nVectors;
        UTCtime = rhs.UTCtime;
        position = rhs.position;
        velocity = rhs.velocity;
        return *this;
    }

    inline Orbit& Orbit::operator+=(const Orbit &rhs) {
        std::vector<double> t_pos(3), t_vel(3);
        for (int i=0; i<rhs.nVectors; i++) {
            t_pos.assign(rhs.position.begin()+(3*i), rhs.position.begin()+(3*(i+1)));
            t_vel.assign(rhs.velocity.begin()+(3*i), rhs.velocity.begin()+(3*(i+1)));
            addStateVector(rhs.UTCtime[i], t_pos, t_vel);
        }
        return *this;
    }

    inline const Orbit Orbit::operator+(const Orbit &rhs) const {
        return (Orbit(*this) += rhs);
    }

    inline void Orbit::getStateVector(int idx, double &t, std::vector<double> &pos, std::vector<double> &vel) {
        if ((idx < 0) || (idx >= nVectors)) {
            std::string errstr = "Orbit::getStateVector - Trying to access vector "+std::to_string(idx+1)+" out of "+std::to_string(nVectors)+" possible vectors";
            throw std::out_of_range(errstr);
        }
        checkVecLen(pos,3);
        checkVecLen(vel,3);
        t = UTCtime[idx];
        pos.assign(position.begin()+(3*idx), position.begin()+(3*idx)+3);
        vel.assign(velocity.begin()+(3*idx), position.begin()+(3*idx)+3);
    }

    inline void Orbit::setStateVector(int idx, double t, std::vector<double> &pos, std::vector<double> &vel) {
        if ((idx < 0) || (idx >= nVectors)) {
            std::string errstr = "Orbit::setStateVector - Trying to access vector "+std::to_string(idx+1)+" out of "+std::to_string(nVectors)+" possible vectors";
            throw std::out_of_range(errstr);
        }
        checkVecLen(pos,3);
        checkVecLen(vel,3);
        UTCtime[idx] = t;
        for (int i=0; i<3; i++) {
            position[3*idx+i] = pos[i];
            velocity[3*idx+i] = pos[i];
        }
    }

    inline void Orbit::addStateVector(double t, std::vector<double> &pos, std::vector<double> &vel) {
        checkVecLen(pos,3);
        checkVecLen(vel,3);
        int vec_idx = 0;
        while ((vec_idx < nVectors) && (t > UTCtime[vec_idx])) vec_idx++;
        UTCtime.insert(UTCtime.begin()+vec_idx, t);
        position.insert(position.begin()+(3*vec_idx), pos.begin(), pos.end());
        velocity.insert(velocity.begin()+(3*vec_idx), vel.begin(), vel.end());
        nVectors++;
    }
}}

#endif
