#cython: language_level=3
#
#
#
from GeoGridParameters cimport GeoGridParameters

cdef class pyGeoGridParameters:
    cdef GeoGridParameters * c_geogrid
    cdef bool __owner
