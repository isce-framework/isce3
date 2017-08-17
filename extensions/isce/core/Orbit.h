//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_ORBIT_H
#define ISCELIB_ORBIT_H

#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "isceLibConstants.h"

namespace isceLib {
    struct Orbit {
        int basis;
        int nVectors;
        std::vector<double> UTCtime;
        std::vector<double> position;
        std::vector<double> velocity;

        Orbit(int bs, int nv) : basis(bs), nVectors(nv), UTCtime(nv,0.), position(3*nv,0.), velocity(3*nv,0.) {}
        Orbit() : Orbit(0,0) {}
        Orbit(const Orbit &o) : basis(o.basis), nVectors(o.nVectors), UTCtime(o.UTCtime), position(o.position), velocity(o.velocity) {}
        inline Orbit& operator=(const Orbit&);
        inline Orbit& operator+=(const Orbit&);

        void getPositionVelocity(double,std::vector<double>&,std::vector<double>&);
        inline void getStateVector(int,double&,std::vector<double>&,std::vector<double>&);
        inline void setStateVector(int,double,std::vector<double>&,std::vector<double>&);
        inline void addStateVector(double,std::vector<double>&,std::vector<double>&);
        int interpolate(double,std::vector<double>&,std::vector<double>&,int);
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
    #ifdef __CUDACC__
    /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
     *
     *      GPU-related Structs/Methods - Please note that the following code is prototype design code for the CUDA algorithms and is highly experimental in nature.
     *
     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
    
    struct gpuOrbit {
        int basis;
        int nVectors;
        double *UTCtime;
        double *position;
        double *velocity;
    
        __host__ __device__ gpuOrbit(int bs, int nv) : basis(bs), nVectors(nv), UTCtime(nullptr), position(nullptr), velocity(nullptr) {}
        __host__ __device__ gpuOrbit() : gpuOrbit(0,0) {}
        __host__ __device__ gpuOrbit(const gpuOrbit &o) : basis(o.basis), nVectors(o.nVectors) {
                                                            UTCtime = new double[o.nVectors];
                                                            position = new double[3*o.nVectors];
                                                            velocity = new double[3*o.nVectors];
                                                            for (int i=0; i<o.nVectors; i++) {
                                                                UTCtime[i] = o.UTCtime[i];
                                                                position[3*i] = o.position[3*i];
                                                                position[3*i+1] = o.position[3*i+1];
                                                                position[3*i+2] = o.position[3*i+2];
                                                                velocity[3*i] = o.velocity[3*i];
                                                                velocity[3*i+1] = o.velocity[3*i+1];
                                                                velocity[3*i+2] = o.velocity[3*i+2];
                                                            }
                                                        }
        __host__ __device__ gpuOrbit(const Orbit &o) : basis(o.basis), nVector(o.nVectors) {
                                                            UTCtime = new double[o.nVectors];
                                                            position = new double[3*o.nVectors];
                                                            velocity = new double[3*o.nVectors];
                                                            for (int i=0; i<o.nVectors; i++) {
                                                                UTCtime[i] = o.UTCtime[i];
                                                                position[3*i] = o.position[3*i];
                                                                position[3*i+1] = o.position[3*i+1];
                                                                position[3*i+2] = o.position[3*i+2];
                                                                velocity[3*i] = o.velocity[3*i];
                                                                velocity[3*i+1] = o.velocity[3*i+1];
                                                                velocity[3*i+2] = o.velocity[3*i+2];
                                                            }
                                                        }
        __host__ __device__ ~gpuOrbit() { delete[] UTCtime; delete[] position; delete[] velocity; }

        __host__ __device__ inline void getStateVector(int,double&,double*,double*);
        __host__ __device__ inline void setStateVector(int,double,double*,double*);
        __host__ __device__ int interpolateWGS84Orbit(double,double*,double*);
        __host__ __device__ int interpolateLegendreOrbit(double,double*,double*);
        __host__ __device__ int interpolateSCHOrbit(double,double*,double*);
        __host__ __device__ int computeAcceleration(double,double*);
    };

    __host__ __device__
    inline void gpuOrbit::getStateVector(int idx, double &t, double *pos, double *vel) {
        // Note: We cannot really bounds-check this since printing serializes threads (so print-debugging would always slow down the code even if never tripped).
        if ((idx < 0) || (idx >= nVectors)) return;
        t = UTCtime[idx];
        for (int i=0; i<3; i++) {
            pos[i] = position[3*idx+i];
            vel[i] = velocity[3*idx+i];
        }
    }

    __host__ __device__
    inline void gpuOrbit::setStateVector(int idx, double t, double *pos, double *vel) {
        // Note: We cannot really bounds-check this since printing serializes threads (so print-debugging would always slow down the code even if never tripped).
        if ((idx < 0) || (idx >= nVectors)) return;
        UTCtime[idx] = t;
        for (int i=0; i<3; i++) {
            position[3*idx+i] = pos[i];
            velocity[3*idx+i] = vel[i];
        }
    }
    #endif
}

#endif
