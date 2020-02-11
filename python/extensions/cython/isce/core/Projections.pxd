#cython: language_level=3
#
# Author: Piyush Agram
# Copyright 2017-2020
#

from Cartesian cimport cartesian_t
from Ellipsoid cimport Ellipsoid

cdef extern from "isce/core/Projections.h" namespace "isce::core":
    cdef cppclass ProjectionBase:

        #Print for debugging
        void print()

        #Get underlying ellipsoid
        Ellipsoid& ellipsoid()

        #Get EPSG code
        int code()

        #Forward method
        int forward(cartesian_t&, cartesian_t&)

        #Inverse method
        int inverse(cartesian_t&, cartesian_t&)

    #Factory 
    ProjectionBase* createProj(int)

# end of file
