from libcpp cimport bool
from LUT1d cimport LUT1d

cdef class pyLUT1d:
    cdef LUT1d[double] * c_lut
    cdef bool __owner
