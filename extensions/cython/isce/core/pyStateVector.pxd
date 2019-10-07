#cython: language_level=3

from StateVector cimport StateVector

cdef class pyStateVector:
    cdef StateVector c_statevector
