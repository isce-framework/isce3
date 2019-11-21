#cython: language_level=3
# 
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from Metadata cimport Metadata

cdef class pyMetadata:

    # C++ class
    cdef Metadata * c_metadata
    cdef bool __owner

    # Cython class members
    cdef pyOrbit py_orbit
    cdef pyEulerAngles py_attitude
    cdef pyProcessingInformation py_procInfo
   
# end of file 
