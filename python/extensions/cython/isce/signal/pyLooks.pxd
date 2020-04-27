#cython: language_level=3

from libcpp cimport bool
from Looks cimport Looks

cdef class pyLooksBase:
    cdef Looks * c_looks
    cdef bool __owner

cdef class pyLooksFloat(pyLooksBase):
    pass

cdef class pyLooksDouble(pyLooksBase):
    pass
