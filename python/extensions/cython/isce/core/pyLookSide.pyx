#cython: language_level=3
from LookSide cimport *

cdef LookSide pyParseLookSide(s):
    cdef string cs = str(s).encode('UTF-8')
    return parseLookSide(cs)
