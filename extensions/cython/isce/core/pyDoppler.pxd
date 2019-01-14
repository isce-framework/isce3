#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2018
#

from libcpp.string cimport string
from libcpp cimport bool
from Doppler cimport Doppler
from isceextension cimport pyEulerAngles
from isceextension cimport pyQuaternion

cdef class pyDoppler:
    cdef Doppler * c_doppler
    cdef int side
    cdef bool precession
    cdef string frame
    cdef bool __owner 

cdef class pyDopplerEuler(pyDoppler):
    cdef pyEulerAngles eulerangles
    
cdef class pyDopplerQuaternion(pyDoppler):
    cdef pyQuaternion quaternion

# end of file
