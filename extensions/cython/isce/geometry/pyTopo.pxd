#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Topo cimport Topo
from Orbit cimport orbitInterpMethod
from Interpolator cimport dataInterpMethod

cdef class pyTopo:
    cdef Topo * c_topo
    cdef bool __owner

    # Orbit interpolation methods
    orbitInterpMethods = {
        'hermite': orbitInterpMethod.HERMITE_METHOD,
        'sch' :  orbitInterpMethod.SCH_METHOD,
        'legendre': orbitInterpMethod.LEGENDRE_METHOD
    }

    # DEM interpolation methods
    demInterpMethods = {
        'sinc': dataInterpMethod.SINC_METHOD,
        'bilinear': dataInterpMethod.BILINEAR_METHOD,
        'bicubic': dataInterpMethod.BICUBIC_METHOD,
        'nearest': dataInterpMethod.NEAREST_METHOD,
        'akima': dataInterpMethod.AKIMA_METHOD,
        'biquintic': dataInterpMethod.BIQUINTIC_METHOD
    }
        
# end of file
