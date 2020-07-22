#cython: language_level=3
#
# Author: Bryan Riel
# Copyright 2017-2019
#

from libcpp cimport bool
from libcpp.string cimport string
from DateTime cimport DateTime
from Matrix cimport valarray, Matrix
from Interpolator cimport dataInterpMethod
from IH5 cimport IGroup

# LUT2d
cdef extern from "isce3/core/LUT2d.h" namespace "isce3::core":
    cdef cppclass LUT2d[T]:

        # Constructors
        LUT2d() except +
        LUT2d(double xstart, double ystart, double dx, double dy,
              const Matrix[T] & data, dataInterpMethod method) except +
        LUT2d(const valarray[double] & xcoord,
              const valarray[double] & ycoord,
              const Matrix[T] & data,
              dataInterpMethod method) except +
        LUT2d(const LUT2d[T] &) except +
        
        # Evaluation
        T eval(double y, double x)

        double xStart()
        bool boundsError()
        void boundsError(bool val)

# Wrapper around isce3::core serialization defined in <isce/core/Serialization.h
cdef extern from "isce3/core/Serialization.h" namespace "isce3::core":
    void loadCalGrid(IGroup & group, const string & dsetName, LUT2d[double] & lut)
    void saveCalGrid(IGroup & group,
                     const string & dsetName,
                     const LUT2d[double] & lut,
                     const DateTime & refEpoch,
                     const string & units)
