#cython: language_level=3
#
# Author: Joshua Cohen
# Copyright 2017
#

from libcpp.vector cimport vector

cdef extern from "Orbit.h" namespace "isceLib":
    cdef cppclass Orbit:
        int basis
        int nVectors
        vector[double] position
        vector[double] velocity
        vector[double] UTCtime

        Orbit() except +
        Orbit(int,int) except +
        Orbit(const Orbit&) except +
        void getPositionVelocity(double,vector[double]&,vector[double]&)
        void getStateVector(int,double&,vector[double]&,vector[double]&)
        void setStateVector(int,double,vector[double]&,vector[double]&)
        void addStateVector(double,vector[double]&,vector[double]&)
        int interpolate(double,vector[double]&,vector[double]&,int)
        int interpolateWGS84Orbit(double,vector[double]&,vector[double]&)
        int interpolateLegendreOrbit(double,vector[double]&,vector[double]&)
        int interpolateSCHOrbit(double,vector[double]&,vector[double]&)
        int computeAcceleration(double,vector[double]&)
        void printOrbit()
        void loadFromHDR(const char*,int)
        void dumpToHDR(const char*)

