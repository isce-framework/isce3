#cython: language_level=3
#
# Author: Bryan V. Riel
# Copyright 2017-2018
#

from libcpp cimport bool
from GDAL cimport GDALDataset

cdef class pyGDALDataset:
    cdef GDALDataset * c_dataset
    cdef bool __owner

# end of file
