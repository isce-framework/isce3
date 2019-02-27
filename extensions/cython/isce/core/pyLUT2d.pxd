from libcpp cimport bool
from LUT2d cimport LUT2d

cdef class pyLUT2d:
    cdef LUT2d[double] * c_lut
    cdef bool __owner

    @staticmethod
    cdef cbind(LUT2d[double])
